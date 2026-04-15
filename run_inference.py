"""
run_inference.py — Batch FCS inference runner
═══════════════════════════════════════════════
Processes all .fcs files in a given data directory using the SM-FCS pipeline.

Usage (on Bouchet):
    python run_inference.py \
        --data_dir  /nfs/roberts/.../01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF \
        --model_dir /nfs/roberts/.../new_ \
        --out_dir   ./results_01082026

Outputs per-file:
    <out_dir>/<stem>_<channel>_events.csv
    <out_dir>/<stem>_<channel>_trace.png
    <out_dir>/<stem>_<channel>_d_scatter.png
    <out_dir>/<stem>_<channel>_d_histogram.png

Plus a combined summary:
    <out_dir>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt= '%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch FCS inference runner')
    parser.add_argument('--data_dir',  required=True,
                        help='Directory containing .fcs (and .raw) files')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing model .pt and .npy files')
    parser.add_argument('--out_dir',   required=True,
                        help='Output directory for CSVs and plots')
    parser.add_argument('--no_plots',  action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--channel',   default=None,
                        help='Process only this channel (e.g. S1). Default: all.')
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model registry ───────────────────────────────────────────────────
    from model_registry import ModelRegistry
    registry = ModelRegistry(
        noise_json     = model_dir / 'noise_models.json',
        diffusion_json = model_dir / 'diffusion_models.json',
        model_dir      = model_dir,
    )
    log.info('Loading models...')
    registry.load_all()

    # ── Discover FCS files ────────────────────────────────────────────────────
    fcs_files = sorted(data_dir.glob('*.fcs'))
    if not fcs_files:
        log.error(f'No .fcs files found in {data_dir}')
        sys.exit(1)

    log.info(f'Found {len(fcs_files)} FCS files in {data_dir}')

    # ── Run pipeline on each file ─────────────────────────────────────────────
    from measurement import load_measurement, MeasurementError
    from pipeline    import run_pipeline

    raw_opts = dict(bits=16, n_header=48, clock_rate_hz=15_000_000)

    all_events  = []   # accumulated across all files for summary CSV
    file_stats  = []   # per-file summary row

    for i, fcs_path in enumerate(fcs_files):
        stem = fcs_path.stem
        log.info(f'[{i+1}/{len(fcs_files)}] Processing {stem}')
        t0 = time.time()

        try:
            meas = load_measurement(
                fcs_path = fcs_path,
                data_dir = data_dir,
                registry = registry,
                raw_opts = raw_opts,
            )
        except MeasurementError as e:
            log.warning(f'  {stem}: load_measurement failed — {e}')
            file_stats.append({
                'file': stem, 'status': 'load_error', 'error': str(e),
                'n_events': 0, 'n_channels': 0,
                'duration_ms': 0, 'has_raw': False,
            })
            continue
        except Exception as e:
            log.error(f'  {stem}: unexpected error in load_measurement:\n{traceback.format_exc()}')
            file_stats.append({
                'file': stem, 'status': 'load_error', 'error': str(e),
                'n_events': 0, 'n_channels': 0,
                'duration_ms': 0, 'has_raw': False,
            })
            continue

        log.info(f'  duration={meas.total_duration_ms:.0f}ms  '
                 f'has_raw={meas.has_raw}  channels={meas.channel_ids}')

        try:
            results = run_pipeline(
                meas        = meas,
                registry    = registry,
                channel_id  = args.channel,
                out_dir     = out_dir,
                run_id      = stem,
                save_csv    = True,
                save_plots  = not args.no_plots,
            )
        except Exception as e:
            log.error(f'  {stem}: run_pipeline failed:\n{traceback.format_exc()}')
            file_stats.append({
                'file': stem, 'status': 'pipeline_error', 'error': str(e),
                'n_events': 0, 'n_channels': 0,
                'duration_ms': meas.total_duration_ms, 'has_raw': meas.has_raw,
            })
            continue

        # Collect results
        n_events_total = sum(r.n_events for r in results)
        dt = time.time() - t0

        for res in results:
            for ev in res.events:
                ev_row = dict(ev)
                ev_row['file']    = stem
                ev_row['channel'] = res.channel_id
                # flatten list fields
                ev_row['d_hat_all'] = ','.join(f'{d:.4f}' for d in ev.get('d_hat_all', []))
                ev_row['flags']     = ','.join(ev.get('flags', []))
                all_events.append(ev_row)

        file_stats.append({
            'file':         stem,
            'status':       'ok',
            'error':        '',
            'n_events':     n_events_total,
            'n_channels':   len(results),
            'duration_ms':  meas.total_duration_ms,
            'has_raw':      meas.has_raw,
            'n_abs_kHz':    meas.n_abs,
            'elapsed_s':    round(dt, 1),
        })
        log.info(f'  → {n_events_total} events in {dt:.1f}s')

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = out_dir / 'summary.csv'
    _write_summary(file_stats, summary_path)
    log.info(f'Summary written → {summary_path}')

    # ── Write combined events CSV ─────────────────────────────────────────────
    if all_events:
        combined_path = out_dir / 'all_events.csv'
        _write_all_events(all_events, combined_path)
        log.info(f'All events ({len(all_events)}) → {combined_path}')

    n_ok     = sum(1 for s in file_stats if s['status'] == 'ok')
    n_fail   = len(file_stats) - n_ok
    n_events = sum(s.get('n_events', 0) for s in file_stats)
    log.info(f'Done: {n_ok}/{len(file_stats)} files OK, '
             f'{n_fail} errors, {n_events} total events')


def _write_summary(stats: list, path: Path) -> None:
    if not stats:
        return
    fieldnames = ['file', 'status', 'error', 'n_events', 'n_channels',
                  'duration_ms', 'has_raw', 'n_abs_kHz', 'elapsed_s']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(stats)


def _write_all_events(events: list, path: Path) -> None:
    if not events:
        return
    # collect all keys seen
    keys = list(dict.fromkeys(k for ev in events for k in ev))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(events)


if __name__ == '__main__':
    main()
