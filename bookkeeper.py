#!/usr/bin/env python3
"""
bookkeeper.py
Survey all experiment folders inside a parent directory, read pipeline_log.json
from every processed_on_* subfolder, and maintain a registry.json summary.

Usage:
    python bookkeeper.py <parent_directory>

Example:
    python bookkeeper.py /nfs/roberts/project/pi_sah46/ah2286/new_/experiments
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

RUN_FOLDER_PREFIX = 'processed_on_'
LOG_FILENAME      = 'pipeline_log.json'
REGISTRY_FILENAME = 'registry.json'


def survey(parent_dir: Path) -> dict:
    """Scan parent_dir, return registry dict."""
    experiments = {}

    candidates = sorted(p for p in parent_dir.iterdir() if p.is_dir()
                        and not p.name.startswith('.'))

    for exp_dir in candidates:
        runs = []
        run_dirs = sorted(p for p in exp_dir.iterdir()
                          if p.is_dir() and p.name.startswith(RUN_FOLDER_PREFIX))

        for run_dir in run_dirs:
            log_path = run_dir / LOG_FILENAME
            if not log_path.exists():
                # Run folder exists but no log — record as unknown
                runs.append({
                    'run_folder':  run_dir.name,
                    'status':      'unknown',
                    'started_at':  None,
                    'finished_at': None,
                    'pipeline_version': None,
                    'n_measurements': None,
                    'n_events':    None,
                })
                continue

            with open(log_path) as f:
                log = json.load(f)

            runs.append({
                'run_folder':       run_dir.name,
                'status':           log.get('status'),
                'started_at':       log.get('started_at'),
                'finished_at':      log.get('finished_at'),
                'pipeline_version': log.get('pipeline_version'),
                'n_measurements':   log.get('n_measurements'),
                'n_events':         log.get('n_events'),
                'error':            log.get('error'),   # only present on failure
            })

        experiments[exp_dir.name] = {'runs': runs}

    return {
        'parent_dir':    str(parent_dir),
        'last_surveyed': datetime.now().isoformat(timespec='seconds'),
        'experiments':   experiments,
    }


def print_table(registry: dict) -> None:
    """Print a human-readable summary table."""
    experiments = registry['experiments']
    if not experiments:
        print('  (no experiments found)')
        return

    # Column widths
    W_EXP  = max(len(k) for k in experiments) + 2
    W_RUN  = 30
    W_VER  = 9
    W_STAT = 12
    W_MEAS = 6
    W_EVT  = 7

    header = (f'{"Experiment":<{W_EXP}}  {"Run folder":<{W_RUN}}  '
              f'{"Version":<{W_VER}}  {"Status":<{W_STAT}}  '
              f'{"Files":>{W_MEAS}}  {"Events":>{W_EVT}}')
    sep    = '-' * len(header)

    print(sep)
    print(header)
    print(sep)

    for exp_name, info in sorted(experiments.items()):
        runs = info['runs']

        if not runs:
            print(f'{exp_name:<{W_EXP}}  {"—":<{W_RUN}}  '
                  f'{"—":<{W_VER}}  {"not processed":<{W_STAT}}  '
                  f'{"—":>{W_MEAS}}  {"—":>{W_EVT}}')
        else:
            for i, run in enumerate(runs):
                exp_col    = exp_name if i == 0 else ''
                ver        = run['pipeline_version'] or '—'
                status     = run['status'] or '—'
                meas       = str(run['n_measurements']) if run['n_measurements'] is not None else '—'
                evts       = str(run['n_events'])       if run['n_events']       is not None else '—'
                status_str = 'FAILED' if status == 'failed' else status

                print(f'{exp_col:<{W_EXP}}  {run["run_folder"]:<{W_RUN}}  '
                      f'{ver:<{W_VER}}  {status_str:<{W_STAT}}  '
                      f'{meas:>{W_MEAS}}  {evts:>{W_EVT}}')

        print()   # blank line between experiments

    print(sep)
    n_runs = sum(len(v['runs']) for v in experiments.values())
    print(f'  {len(experiments)} experiment(s)  |  {n_runs} run(s) total')
    print(f'  Surveyed: {registry["last_surveyed"]}')


def main():
    parser = argparse.ArgumentParser(description='FCS pipeline bookkeeper')
    parser.add_argument('parent_dir',
                        help='Parent directory containing experiment folders')
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir).resolve()
    if not parent_dir.is_dir():
        sys.exit(f'ERROR: directory not found: {parent_dir}')

    print(f'\nSurveying: {parent_dir}\n')
    registry = survey(parent_dir)

    # Write / overwrite registry.json
    registry_path = parent_dir / REGISTRY_FILENAME
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f'Registry written → {registry_path}\n')

    print_table(registry)


if __name__ == '__main__':
    main()
