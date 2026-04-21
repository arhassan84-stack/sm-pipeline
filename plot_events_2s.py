# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
plot_events_2s.py

For each file × channel, produce a multi-panel figure showing consecutive
2-second windows of the raw FCS intensity trace with detected events shaded
by log10(D̂).

Layout: 3 columns × N_ROWS rows, each panel = 2 seconds.
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
CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 11
FS_LABEL    = 13
FS_TICK     = 10

# ── Config ────────────────────────────────────────────────────────────────────
FILES    = ['nt_dorsal_1', 'nt_dorsal_10', 'nt_dorsal_15']
CHANNELS = ['S1', 'S2']
WIN_MS   = 2000    # 2 seconds per panel
N_COLS   = 3       # panels per row
N_ROWS   = 6       # rows → 18 panels × 2 s = 36 s of data shown
D_VMIN   = -2.5
D_VMAX   = -0.5


def load_events(csv_path: Path) -> list:
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


def make_figure(stem: str, ch_id: str, i100: np.ndarray, events: list,
                out_path: Path) -> None:
    N      = len(i100)
    t_ms   = np.arange(N, dtype=float)
    n_pan  = N_COLS * N_ROWS

    cmap = cm.get_cmap('RdYlBu_r')
    norm = Normalize(vmin=D_VMIN, vmax=D_VMAX)

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(N_COLS * 5, N_ROWS * 2),
        facecolor='black',
    )
    fig.suptitle(
        f'{stem} / {ch_id}   ({len(events)} events)',
        fontsize=FS_SUPTITLE, color='white',
    )

    for idx, ax in enumerate(axes.flat):
        t0 = idx * WIN_MS
        t1 = t0 + WIN_MS
        if t0 >= N:
            ax.set_visible(False)
            continue

        seg   = i100[t0 : min(t1, N)]
        t_seg = t_ms[t0 : t0 + len(seg)]

        ax.set_facecolor('black')
        ax.plot(t_seg * 1e-3, seg, color=CURVE_COLOR, lw=0.6, alpha=0.9)

        # Shade events that overlap this 2-second window
        for ev in events:
            el, er = ev['t_left'], ev['t_right']
            if er < t0 or el > t1:
                continue
            sl = max(el, t0)
            sr = min(er, t1)
            c  = cmap(norm(ev['log10_D']))
            ax.axvspan(sl * 1e-3, sr * 1e-3, alpha=0.40, color=c, lw=0)
            if t0 <= ev['t_peak'] <= t1:
                ax.axvline(ev['t_peak'] * 1e-3,
                           color=c, lw=0.9, alpha=0.8, ls='--')

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_color('white')
        ax.tick_params(colors='white', labelsize=FS_TICK)
        ax.set_xlim(t0 * 1e-3, t1 * 1e-3)
        ax.set_title(f'{t0/1000:.0f}–{t1/1000:.0f} s',
                     fontsize=FS_TITLE, color='white', pad=2)

        col = idx % N_COLS
        row = idx // N_COLS
        if col == 0:
            ax.set_ylabel('kHz', fontsize=FS_TICK, color='white')
        if row == N_ROWS - 1:
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='white')

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label('log₁₀ D̂  (µm²/s)', fontsize=FS_LABEL, color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=FS_TICK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path.name}')


# ── Main ──────────────────────────────────────────────────────────────────────
for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    if not fcs_path.exists():
        print(f'Missing FCS: {fcs_path}')
        continue

    print(f'\n{stem}')
    fcs = parse_fcs_file(fcs_path)

    for ch_idx, ch_id in enumerate(CHANNELS):
        csv_path = RES_DIR / f'{stem}_{ch_id}_events.csv'
        if not csv_path.exists():
            print(f'  Missing CSV: {csv_path.name}')
            continue

        i100   = fcs.intensity_trace(channel=ch_idx)   # kHz, 1ms bins
        events = load_events(csv_path)
        print(f'  {ch_id}: {len(i100)} ms, {len(events)} events')

        out = RES_DIR / f'{stem}_{ch_id}_events_2s.png'
        make_figure(stem, ch_id, i100, events, out)

print('\nDone.')
