#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_submeasurements.py -- FCS quality-control viewer.

For each FCS file, generates a PNG showing the raw (un-subtracted)
1 ms intensity trace for the first N sub-measurements, with detected
events overlaid as shaded regions.

Usage
-----
    python plot_submeasurements.py \
        --fcs_dir  /path/to/fcs/files \
        --res_dir  /path/to/results_01082026 \
        --out_dir  /path/to/output \
        [--n_show 30] \
        [--n_cols 5]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file

# ── aesthetics ──────────────────────────────────────────────────────────────
S1_COLOR    = '#1f77b4'   # blue   (S1 / mCherry2 / ITGA5)
S2_COLOR    = '#d62728'   # red    (S2 / GFP / ITGB1)
BG_COLOR    = 'white'
FG_COLOR    = 'black'
GRID_COLOR  = '#cccccc'
SPINE_LW    = 1.5
FS_SUPTITLE = 13
FS_TITLE    = 8
FS_TICK     = 7


def _load_events(events_csv: Path) -> dict:
    """Return {stem: {'S1': [(t_left, t_right), ...], 'S2': [...]}}."""
    ev = {}
    if not events_csv.exists():
        return ev
    with open(events_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem  = row['file']
            ch    = row['channel']
            # Store (t_peak, W_seg) — shade ±W_seg around each peak only.
            entry = (int(row['t_peak']), int(row['W_seg']),
                     int(row['t_left']), int(row['t_right']))
            ev.setdefault(stem, {'S1': [], 'S2': []})[ch].append(entry)
    return ev


def _count_rates(b, ch_idx: int):
    """Return kHz array for channel ch_idx in FCSBlock b, or None."""
    if ch_idx >= len(b.count_rates):
        return None
    arr = b.count_rates[ch_idx]
    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
        return arr[:, 1] / 1000.0   # Hz → kHz
    if arr.ndim == 1 and arr.shape[0] > 0:
        return arr / 1000.0
    return None


BRACKET_COLOR = 'orange'

def _mark_peak(ax, t_peak_ms, w_seg_ms, t_left_ms, t_right_ms, t_off_ms, bpb, color, kHz=None):
    """
    Draw a vertical line at t_peak and an orange bracket at the peak height
    spanning [t_left, t_right], clipped to [t_off, t_off+bpb].
    Only draws if the peak falls within this sub-measurement.
    """
    if t_peak_ms < t_off_ms or t_peak_ms >= t_off_ms + bpb:
        return
    t_peak_local = (t_peak_ms - t_off_ms) * 1e-3
    ax.axvline(t_peak_local, color=color, lw=1.0, alpha=0.9, zorder=2)

    # Orange bracket at peak height, clipped to panel
    t_l_local = max(0,           (t_left_ms  - t_off_ms) * 1e-3)
    t_r_local = min(bpb * 1e-3, (t_right_ms - t_off_ms) * 1e-3)
    if t_r_local > t_l_local:
        # Determine peak height from the kHz trace; fall back to a fixed height
        peak_idx = t_peak_ms - t_off_ms
        if kHz is not None and 0 <= peak_idx < len(kHz):
            y_top = float(kHz[peak_idx])
        else:
            y_top = ax.get_ylim()[1] * 0.8
        tick_h = y_top * 0.12   # short downward ticks at bracket ends
        # Horizontal bar
        ax.plot([t_l_local, t_r_local], [y_top, y_top],
                color=BRACKET_COLOR, lw=1.5, alpha=0.85,
                solid_capstyle='butt', zorder=3)
        # Left tick
        ax.plot([t_l_local, t_l_local], [y_top, y_top - tick_h],
                color=BRACKET_COLOR, lw=1.5, alpha=0.85, zorder=3)
        # Right tick
        ax.plot([t_r_local, t_r_local], [y_top, y_top - tick_h],
                color=BRACKET_COLOR, lw=1.5, alpha=0.85, zorder=3)


def plot_file(fcs_path: Path, all_events: dict, out_dir: Path,
              n_show: int, n_cols: int) -> None:
    stem = fcs_path.stem
    print(f'  {stem}', end=' ', flush=True)

    fcs = parse_fcs_file(fcs_path)
    if fcs.n_blocks == 0:
        print('(no blocks — skipped)')
        return

    nc       = max(fcs.n_channels, 1)
    first_b  = next((b for b in fcs.blocks if b.count_rates), None)
    if first_b is None:
        print('(no count-rate data — skipped)')
        return

    # Detect layout (same logic as fcs_io.intensity_trace)
    multi_ch = nc > 1 and len(first_b.count_rates) >= nc
    bpb      = first_b.count_rates[0].shape[0]   # bins per sub-measurement

    # Map sub-measurement index → list of FCSBlocks that carry it
    if multi_ch:
        # Each block is one full sub-measurement (all channels inside)
        sub_blocks = [(k, fcs.blocks[k]) for k in range(fcs.n_blocks)]
    else:
        # Interleaved: blocks 0,nc,2nc,... = sub-meas 0; blocks 1,nc+1,... = sub-meas 0 ch1 etc.
        # Group nc consecutive blocks into one sub-measurement
        sub_blocks = [
            (k, fcs.blocks[k * nc: k * nc + nc])
            for k in range(fcs.n_blocks // nc)
        ]

    n_submeas = len(sub_blocks)
    n_plot    = min(n_show, n_submeas)
    n_rows    = (n_plot + n_cols - 1) // n_cols

    ev_s1 = all_events.get(stem, {}).get('S1', [])
    ev_s2 = all_events.get(stem, {}).get('S2', [])

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 1.6),
        facecolor=BG_COLOR,
        squeeze=False,
    )
    fig.suptitle(
        f'{stem}  --  first {n_plot} / {n_submeas} sub-measurements  (raw FCS, no background subtraction)',
        color=FG_COLOR, fontsize=FS_SUPTITLE, y=1.01,
    )

    t_local = np.arange(bpb) * 1e-3   # time axis in seconds

    for plot_idx in range(n_plot):
        row_i, col_i = divmod(plot_idx, n_cols)
        ax = axes[row_i, col_i]
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_edgecolor(FG_COLOR)

        # Sub-measurement absolute offset in the concatenated trace (ms)
        t_off = plot_idx * bpb

        # Pull intensity arrays for each channel
        if multi_ch:
            k, b = sub_blocks[plot_idx]
            ch_data = [
                (0, S1_COLOR, 'S1', ev_s1),
                (1, S2_COLOR, 'S2', ev_s2),
            ]
            for ci, color, ch_label, ev_list in ch_data:
                kHz = _count_rates(b, ci)
                if kHz is not None:
                    n = min(len(kHz), bpb)
                    ax.plot(t_local[:n], kHz[:n], color=color, lw=0.5)
                # Mark each detected peak: vertical line + orange bracket at peak height
                for (t_peak, w_seg, t_left, t_right) in ev_list:
                    _mark_peak(ax, t_peak, w_seg, t_left, t_right, t_off, bpb, color, kHz)
        else:
            # Interleaved
            k, blks = sub_blocks[plot_idx]
            for ci, (color, ev_list) in enumerate([(S1_COLOR, ev_s1), (S2_COLOR, ev_s2)]):
                if ci >= len(blks):
                    break
                kHz = _count_rates(blks[ci], 0)
                if kHz is not None:
                    n = min(len(kHz), bpb)
                    ax.plot(t_local[:n], kHz[:n], color=color, lw=0.5)
                for (t_peak, w_seg, t_left, t_right) in ev_list:
                    _mark_peak(ax, t_peak, w_seg, t_left, t_right, t_off, bpb, color, kHz)

        ax.set_title(
            f'k={plot_idx}  (t={t_off / 1000:.0f}s)',
            color=FG_COLOR, fontsize=FS_TITLE, pad=2,
        )
        ax.tick_params(colors=FG_COLOR, labelsize=FS_TICK, length=2, width=0.8)
        ax.set_xlim(0, bpb * 1e-3)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

        if col_i == 0:
            ax.set_ylabel('kHz', color=FG_COLOR, fontsize=FS_TICK)
        else:
            ax.set_yticklabels([])
        if row_i == n_rows - 1 or plot_idx + n_cols >= n_plot:
            ax.set_xlabel('s', color=FG_COLOR, fontsize=FS_TICK)
        else:
            ax.set_xticklabels([])

    # Hide unused axes
    for plot_idx in range(n_plot, n_rows * n_cols):
        axes[plot_idx // n_cols, plot_idx % n_cols].set_visible(False)

    handles = [
        mpatches.Patch(color=S1_COLOR,      label='S1  mCherry2 · ITGA5'),
        mpatches.Patch(color=S2_COLOR,      label='S2  GFP · ITGB1'),
        mpatches.Patch(color=BRACKET_COLOR, label='event extent bracket'),
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=3, fontsize=8,
        facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=FG_COLOR,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = out_dir / f'{stem}_submeasurements.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'→ {out_path.name}')


def main():
    ap = argparse.ArgumentParser(description='FCS sub-measurement viewer')
    ap.add_argument('--fcs_dir', required=True,
                    help='Directory containing .fcs files')
    ap.add_argument('--res_dir', required=True,
                    help='Results directory (must contain all_events.csv)')
    ap.add_argument('--events_csv', default=None,
                    help='Override path to events CSV (default: <res_dir>/all_events.csv)')
    ap.add_argument('--out_dir', required=True,
                    help='Output directory for PNG files')
    ap.add_argument('--n_show', type=int, default=30,
                    help='Max sub-measurements to show per file (default 30)')
    ap.add_argument('--n_cols', type=int, default=5,
                    help='Number of columns in the grid (default 5)')
    args = ap.parse_args()

    fcs_dir    = Path(args.fcs_dir)
    res_dir    = Path(args.res_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_csv = Path(args.events_csv) if args.events_csv else res_dir / 'all_events.csv'
    all_events = _load_events(events_csv)
    print(f'Loaded events for {len(all_events)} files from {events_csv.name}')

    fcs_files = sorted(fcs_dir.glob('*.fcs'))
    print(f'Found {len(fcs_files)} FCS files\n')

    for fcs_path in fcs_files:
        plot_file(fcs_path, all_events, out_dir, args.n_show, args.n_cols)

    print(f'\nDone.  PNGs written to {out_dir}')


if __name__ == '__main__':
    main()
