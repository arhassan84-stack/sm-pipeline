# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
plot_events_filtered.py

Reload existing pipeline CSVs, look up peak amplitude from the FCS trace,
discard events below 50% of the global peak amplitude, and replot
2-second panels (same layout as plot_events_2s.py).
"""

import csv
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
RES_DIR  = PROJECT / 'results_01082026' / 'pipeline_test'

sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file

# ── Aesthetics (MEMORY.md) ────────────────────────────────────────────────────
CURVE_COLOR = '#007700'   # dark green readable on white
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 11
FS_LABEL    = 13
FS_TICK     = 10

# ── Config ────────────────────────────────────────────────────────────────────
FILES             = ['nt_dorsal_1', 'nt_dorsal_10', 'nt_dorsal_15']
CHANNELS          = ['S1', 'S2']
WIN_MS            = 2000
N_COLS            = 3
N_ROWS            = 6
AMPLITUDE_FRAC    = 0.50     # keep events >= this fraction of global peak max
AMPLITUDE_ABS_CAP = 50.0    # threshold is capped at this absolute value (kHz)
                             # so bright files with high global_max don't over-exclude
D_VMIN, D_VMAX    = -2.5, -0.5


def load_events(csv_path):
    events = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    't_left':  float(row['t_left_ms']),
                    't_right': float(row['t_right_ms']),
                    't_peak':  float(row['t_peak_ms']),
                    'log10_D': float(row['log10_D']),
                })
            except (ValueError, KeyError):
                pass
    return events


def make_figure(stem, ch_id, i100, events, n_total, out_path):
    N    = len(i100)
    t_ms = np.arange(N, dtype=float)
    cmap = cm.get_cmap('RdYlBu_r')
    norm = Normalize(vmin=D_VMIN, vmax=D_VMAX)

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                              figsize=(N_COLS * 5, N_ROWS * 2),
                              facecolor='white')
    fig.suptitle(
        f'{stem} / {ch_id}   {len(events)} events after filter  '
        f'(was {n_total}, -{n_total - len(events)} removed)',
        fontsize=FS_SUPTITLE, color='black',
    )

    for idx, ax in enumerate(axes.flat):
        t0, t1 = idx * WIN_MS, (idx + 1) * WIN_MS
        if t0 >= N:
            ax.set_visible(False); continue

        seg   = i100[t0 : min(t1, N)]
        t_seg = t_ms[t0 : t0 + len(seg)]

        ax.set_facecolor('white')
        ax.plot(t_seg * 1e-3, seg, color=CURVE_COLOR, lw=0.6, alpha=0.9)

        for ev in events:
            if ev['t_right'] < t0 or ev['t_left'] > t1:
                continue
            sl = max(ev['t_left'],  t0)
            sr = min(ev['t_right'], t1)
            c  = cmap(norm(ev['log10_D']))
            ax.axvspan(sl * 1e-3, sr * 1e-3, alpha=0.35, color=c, lw=0)
            if t0 <= ev['t_peak'] <= t1:
                ax.axvline(ev['t_peak'] * 1e-3, color=c, lw=0.9, alpha=0.8, ls='--')

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW); sp.set_color('black')
        ax.tick_params(colors='black', labelsize=FS_TICK)
        ax.set_xlim(t0 * 1e-3, t1 * 1e-3)
        ax.set_title(f'{t0/1000:.0f}–{t1/1000:.0f} s',
                     fontsize=FS_TITLE, color='black', pad=2)
        if idx % N_COLS == 0:
            ax.set_ylabel('kHz', fontsize=FS_TICK, color='black')
        if idx // N_COLS == N_ROWS - 1:
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path.name}')


# ── Main ─────────────────────────────────────────────────────────────────────
for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    if not fcs_path.exists():
        print(f'Missing FCS: {fcs_path}'); continue
    print(f'\n{stem}')
    fcs = parse_fcs_file(fcs_path)

    for ch_idx, ch_id in enumerate(CHANNELS):
        csv_path = RES_DIR / f'{stem}_{ch_id}_events.csv'
        if not csv_path.exists():
            print(f'  Missing CSV: {csv_path.name}'); continue

        i100   = fcs.intensity_trace(channel=ch_idx)
        events = load_events(csv_path)

        # Look up peak amplitude from trace for each event.
        # Use max over ±10ms window to tolerate the ~4ms timing offset
        # between the RAW-binned trace (used by the pipeline) and the
        # FCS-stored count-rate trace (used here for display).
        W_LOOKUP = 10
        N_tr = len(i100)
        for ev in events:
            tp  = int(round(ev['t_peak']))
            sl  = slice(max(0, tp - W_LOOKUP), min(N_tr, tp + W_LOOKUP + 1))
            ev['h_peak'] = float(i100[sl].max()) if tp < N_tr else 0.0

        # Global max across all detected peaks
        h_all      = np.array([ev['h_peak'] for ev in events])
        global_max = float(h_all.max()) if len(h_all) else 1.0
        threshold  = AMPLITUDE_FRAC * global_max

        threshold  = min(AMPLITUDE_FRAC * global_max, AMPLITUDE_ABS_CAP)

        n_before    = len(events)
        events_filt = [ev for ev in events if ev['h_peak'] >= threshold]

        print(f'  {ch_id}: {n_before} -> {len(events_filt)} events '
              f'(threshold {threshold:.1f} kHz = min({AMPLITUDE_FRAC*100:.0f}% '
              f'of {global_max:.1f}, {AMPLITUDE_ABS_CAP:.0f}) kHz)')

        out = RES_DIR / f'{stem}_{ch_id}_events_filtered_2s.png'
        make_figure(stem, ch_id, i100, events_filt, n_before, out)

print('\nDone.')
