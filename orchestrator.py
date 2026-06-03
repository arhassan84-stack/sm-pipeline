#!/usr/bin/env python3
"""
orchestrator.py
End-to-end pipeline orchestration:
  1. Select experiments that need processing (via bookkeeper)
  2. Sync selected experiment data to the cluster
  3. Submit one SLURM job per experiment
  4. Monitor jobs — copy back each experiment's output as it finishes
  5. Update the bookkeeper registry

Usage:
    python orchestrator.py <parent_dir> [options]

Examples:
    # Process everything not yet completed with version 2.0
    python orchestrator.py /path/to/experiments --version 2.0

    # Reprocess all experiments regardless of prior runs
    python orchestrator.py /path/to/experiments --version 2.0 --all
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bookkeeper import survey, print_table
from version    import PIPELINE_SCRIPT, PIPELINE_VERSION

SLURM_SCRIPT     = 'job_full_pipeline_with_plots.sh'
DEFAULT_CLUSTER  = 'bouchet.ycrc.yale.edu'
DEFAULT_REMOTE   = '/nfs/roberts/project/pi_sah46/ah2286/new_'
DEFAULT_POLL     = 60   # seconds


# ── Cluster helpers ────────────────────────────────────────────────────────────

def ssh(host, cmd, check=True):
    """Run cmd on host via SSH. Returns (stdout, returncode)."""
    result = subprocess.run(
        ['ssh', host, cmd],
        capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        sys.exit(f'SSH command failed:\n  cmd : {cmd}\n  err : {result.stderr.strip()}')
    return result.stdout.strip(), result.returncode


def rsync_file_up(host, local_file, remote_dir):
    """Upload a single local file to host:remote_dir/, skipping if already present."""
    subprocess.run(
        ['rsync', '-av', '--update', str(local_file), f'{host}:{remote_dir}/'],
        check=True)


def rsync_files_up(host, local_files, remote_dir):
    """Upload a list of local files to host:remote_dir/ in one rsync call.
    Files unchanged since the last upload are skipped (--update)."""
    subprocess.run(
        ['rsync', '-av', '--update'] + [str(f) for f in local_files] + [f'{host}:{remote_dir}/'],
        check=True)


def rsync_up(host, local_dir, remote_dir, exclude=None):
    """Upload local_dir/ to host:remote_dir/, skipping already-present files.
    Optionally exclude glob patterns (e.g. 'processed_on_*')."""
    cmd = ['rsync', '-av', '--update']
    for pat in (exclude or []):
        cmd += ['--exclude', pat]
    cmd += [f'{local_dir}/', f'{host}:{remote_dir}/']
    subprocess.run(cmd, check=True)


def rsync_outputs_down(host, remote_dir, local_dir):
    """Download only processed_on_* subdirs from host:remote_dir/ → local_dir/.
    Skips files already present locally."""
    cmd = [
        'rsync', '-av', '--update',
        '--include=processed_on_*/',
        '--include=processed_on_*/**',
        '--exclude=*',
        f'{host}:{remote_dir}/',
        f'{local_dir}/',
    ]
    subprocess.run(cmd, check=True)


def list_remote_processed_folders(host, remote_exp_dir):
    """Return the set of processed_on_* folder names present on the cluster."""
    out, rc = ssh(host, f'ls -d {remote_exp_dir}/processed_on_* 2>/dev/null', check=False)
    if rc != 0 or not out.strip():
        return set()
    return {Path(p.strip()).name for p in out.strip().splitlines() if p.strip()}


def rsync_new_outputs_down(host, remote_exp_dir, local_exp_dir, pre_folders):
    """Download only processed_on_* folders that are NEW since pre_folders snapshot.

    pre_folders: set of folder names that existed before the current job ran.
    Only folders not in pre_folders are rsynced, preventing stale outputs from
    previous cluster runs being pulled down.
    Falls back to rsync_outputs_down if pre_folders is None (legacy sessions).
    """
    if pre_folders is None:
        rsync_outputs_down(host, remote_exp_dir, local_exp_dir)
        return
    current = list_remote_processed_folders(host, remote_exp_dir)
    new_folders = sorted(current - pre_folders)
    if not new_folders:
        return
    local_exp_dir = Path(local_exp_dir)
    for folder in new_folders:
        local_folder = local_exp_dir / folder
        local_folder.mkdir(parents=True, exist_ok=True)
        cmd = [
            'rsync', '-av', '--update',
            f'{host}:{remote_exp_dir}/{folder}/',
            f'{local_folder}/',
        ]
        subprocess.run(cmd, check=True)


def active_job_ids(host, job_ids):
    """Return the subset of job_ids still present in the SLURM queue."""
    if not job_ids:
        return set()
    ids_str = ','.join(sorted(job_ids))
    out, _  = ssh(host, f'squeue -j {ids_str} -h -o "%i" 2>/dev/null', check=False)
    return {line.strip() for line in out.splitlines() if line.strip()}


def job_final_state(host, job_id):
    """Return the final SLURM state string for a completed job.

    Queries sacct and returns the state of the main job record
    (e.g. 'COMPLETED', 'TIMEOUT', 'FAILED', 'CANCELLED').
    Retries up to 3 times with a 5-second pause in case sacct
    hasn't recorded the job yet.
    """
    for _ in range(3):
        out, _ = ssh(
            host,
            f'sacct -j {job_id} --format=State --noheader -P 2>/dev/null'
            f' | grep -v "^$" | head -1',
            check=False)
        state = out.strip().split()[0].upper() if out.strip() else ''
        if state:
            return state
        time.sleep(5)
    return 'UNKNOWN'


# ── Experiment selection ───────────────────────────────────────────────────────

def select_experiments(registry, target_version, run_all):
    """Return list of (exp_name, last_status) for experiments that need processing."""
    selected = []
    for exp_name, info in sorted(registry['experiments'].items()):
        runs = info['runs']
        if run_all:
            last = runs[-1]['status'] if runs else 'not processed'
            selected.append((exp_name, last))
            continue
        # Include if no *completed* run with the target version exists
        already_done = any(
            r['status'] == 'completed' and
            (target_version is None or r['pipeline_version'] == target_version)
            for r in runs
        )
        if not already_done:
            last = runs[-1]['status'] if runs else 'not processed'
            selected.append((exp_name, last))
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FCS pipeline orchestrator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('parent_dir',
                        help='Local parent directory containing experiment folders')
    parser.add_argument('--version', default=None,
                        help='Pipeline version to check against (e.g. 2.0)')
    parser.add_argument('--all', action='store_true', dest='run_all',
                        help='Reprocess all experiments, including already-completed ones')
    parser.add_argument('--cluster', default=DEFAULT_CLUSTER,
                        help='Cluster SSH hostname')
    parser.add_argument('--remote-base', default=DEFAULT_REMOTE,
                        help='Remote directory containing the pipeline scripts and models')
    parser.add_argument('--remote-experiments', default=None,
                        help='Remote experiments directory '
                             '(default: <remote-base>/experiments)')
    parser.add_argument('--poll-interval', type=int, default=DEFAULT_POLL,
                        help='Seconds between squeue status checks')
    parser.add_argument('--time-limit', default=None,
                        help='Override SLURM wall-time limit (e.g. 4:00:00). '
                             'Blank = use the script default (currently 2:00:00)')
    args = parser.parse_args()

    parent_dir     = Path(args.parent_dir).resolve()
    cluster        = args.cluster
    remote_base    = args.remote_base.rstrip('/')
    remote_exp_dir = (args.remote_experiments or f'{remote_base}/experiments').rstrip('/')

    if not parent_dir.is_dir():
        sys.exit(f'ERROR: directory not found: {parent_dir}')

    # ── Phase 1 — Select ──────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('Phase 1 — Selecting experiments')
    print(f'{"="*60}')

    registry = survey(parent_dir)
    selected = select_experiments(registry, args.version, args.run_all)

    if not selected:
        tag = f'version {args.version}' if args.version else 'any version'
        print(f'\nNothing to do — all experiments already completed ({tag}).')
        print('Use --all to reprocess.')
        return

    print(f'\n  {len(selected)} experiment(s) to process:\n')
    for exp_name, last_status in selected:
        print(f'  {"[" + last_status + "]":<20}  {exp_name}')

    to_process = [name for name, _ in selected]

    # ── Phase 2 — Ensure pipeline script is on the cluster ───────────────────
    print(f'\n{"="*60}')
    print('Phase 2 — Checking pipeline script on cluster')
    print(f'{"="*60}')

    local_script  = Path(__file__).parent / PIPELINE_SCRIPT
    remote_script = f'{remote_base}/{PIPELINE_SCRIPT}'

    if not local_script.exists():
        sys.exit(f'ERROR: pipeline script not found locally: {local_script}')

    out, rc = ssh(cluster, f'test -f {remote_script} && echo exists', check=False)
    if 'exists' in out:
        print(f'  {PIPELINE_SCRIPT} already present on cluster — skipping upload')
    else:
        print(f'  {PIPELINE_SCRIPT} not found on cluster — uploading...')
        rsync_file_up(cluster, local_script, remote_base)
        print(f'  uploaded {PIPELINE_SCRIPT} → {remote_base}/')

    # ── Phase 3 — Sync experiment data to cluster ─────────────────────────────
    print(f'\n{"="*60}')
    print('Phase 3 — Syncing experiment data to cluster')
    print(f'{"="*60}')

    ssh(cluster, f'mkdir -p {remote_exp_dir}')

    for exp_name in to_process:
        local_exp  = parent_dir / exp_name
        remote_exp = f'{remote_exp_dir}/{exp_name}'
        print(f'\n  ↑  {exp_name}')
        ssh(cluster, f'mkdir -p {remote_exp}')
        rsync_up(
            host=cluster,
            local_dir=local_exp,
            remote_dir=remote_exp,
            exclude=['processed_on_*'],   # never upload old outputs
        )

    # ── Phase 4 — Submit SLURM jobs ───────────────────────────────────────────
    print(f'\n{"="*60}')
    print('Phase 4 — Submitting SLURM jobs')
    print(f'{"="*60}\n')

    job_map  = {}   # job_id (str) → exp_name
    failed_submissions = []

    for exp_name in to_process:
        remote_exp = f'{remote_exp_dir}/{exp_name}'
        time_opt = f'--time={args.time_limit} ' if args.time_limit else ''
        cmd = (
            f'cd {remote_base} && '
            f'sbatch {time_opt}'
            f'--export=DATA_FOLDER={remote_exp},'
            f'PIPELINE_SCRIPT={PIPELINE_SCRIPT} {SLURM_SCRIPT}'
        )
        out, rc = ssh(cluster, cmd, check=False)
        if rc != 0 or 'Submitted' not in out:
            print(f'  ERROR  {exp_name}: {out}')
            failed_submissions.append(exp_name)
            continue
        job_id = out.strip().split()[-1]
        job_map[job_id] = exp_name
        print(f'  job {job_id}  →  {exp_name}')

    if not job_map:
        sys.exit('\nNo jobs submitted successfully — aborting.')
    if failed_submissions:
        print(f'\n  WARNING: submission failed for: {failed_submissions}')

    # ── Phase 5 — Monitor + rolling copy-back ────────────────────────────────
    print(f'\n{"="*60}')
    print('Phase 5 — Monitoring + copying back as jobs finish')
    print(f'{"="*60}\n')

    pending    = set(job_map.keys())
    n_total    = len(job_map)
    n_done     = 0

    while pending:
        active       = active_job_ids(cluster, pending)
        just_finished = pending - active

        for job_id in sorted(just_finished):
            exp_name  = job_map[job_id]
            local_exp = parent_dir / exp_name
            remote_exp = f'{remote_exp_dir}/{exp_name}'
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] job {job_id} finished  ↓  {exp_name}')
            rsync_outputs_down(cluster, remote_exp, local_exp)
            n_done += 1
            print(f'         copied back → {local_exp}  '
                  f'({n_done}/{n_total} done)')

        pending = active

        if pending:
            ts = datetime.now().strftime('%H:%M:%S')
            running_ids = ', '.join(sorted(pending))
            print(f'[{ts}] {len(pending)} job(s) still running '
                  f'[{running_ids}] — next check in {args.poll_interval}s')
            time.sleep(args.poll_interval)

    # ── Final: update registry ─────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('All jobs finished — updating bookkeeper registry')
    print(f'{"="*60}\n')

    final_registry = survey(parent_dir)
    reg_path = parent_dir / 'registry.json'
    with open(reg_path, 'w') as f:
        json.dump(final_registry, f, indent=2)
    print(f'Registry updated → {reg_path}\n')

    print_table(final_registry)


if __name__ == '__main__':
    main()
