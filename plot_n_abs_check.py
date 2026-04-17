#!/usr/bin/env python3
"""
plot_n_abs_check.py
Plot raw 1ms intensity traces for a few FCS files with the n_abs background
level (from summary.csv) overlaid as a dashed horizontal line.
Helps visually assess whether n_abs is a good background estimate.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file

# ── Aesthetics (from MEMORY.md) ───────────────────────────────────────────────
S1_COLOR    = '#00FF00'   # bright green — S1 / mCherry2
S2_COLOR    = '#FF4444'   # red          — S2 / GFP
BG_COLOR    = 'black'
FG_COLOR    = 'white'
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12

# ── Paths ─────────────────────────────────────────────────────────────────────
FCS_DIR     = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
SUMMARY_CSV = PROJECT / 'results_01082026' / 'summary.csv'
OUT_PATH    = PROJECT / 'results_01082026' / 'n_abs_check.png'

# Files to plot
FILES_TO_PLOT = ['nt_dorsal_1', 'nt_dorsal_5', 'nt_dorsal_10',
                 'nt_dorsal_11', 'nt_dorsal_15', 'nt_dorsal_18']

# ── Load n_abs from summary ───────────────────────────────────────────────────
n_abs_map = {}
with open(SUMMARY_CSV) as f:
    for row in csv.DictReader(f):
        n_abs_map[row['file']] = float(row['n_abs_kHz'])

# ── Build full 1ms trace from FCS blocks ─────────────────────────────────────
def build_trace(fcs_path, ch_idx):
    """Concatenate count_rates across all blocks for channel ch_idx. Returns kHz array."""
    fcs = parse_fcs_file(fcs_path)
    segments = []
    nc = max(fcs.n_channels, 1)
    multi_ch = nc > 1 and len(fcs.blocks[0].count_rates) >= nc if fcs.n_blocks > 0 else False

    for b in fcs.blocks:
        if not b.count_rates:
            continue
        arr = b.count_rates[ch_idx] if multi_ch else (b.count_rates[0] if b.count_rates else None)
        if arr is None:
            continue
        if arr.ndim == 2 and arr.shape[1] >= 2:
            segments.append(arr[:, 1] / 1000.0)
        elif arr.ndim == 1:
            segments.append(arr / 1000.0)
    return np.concatenate(segments) if segments else np.array([])

# ── Plot ──────────────────────────────────────────────────────────────────────
n_files = len(FILES_TO_PLOT)
fig, axes = plt.subplots(n_files, 2,
                         figsize=(18, n_files * 2.5),
                         facecolor=BG_COLOR,
                         squeeze=False)

fig.suptitle('Raw intensity traces  —  dashed line = n_abs background estimate',
             color=FG_COLOR, fontsize=FS_SUPTITLE, y=1.01)

for row_i, stem in enumerate(FILES_TO_PLOT):
    fcs_path = FCS_DIR / f'{stem}.fcs'
    n_abs    = n_abs_map.get(stem, None)

    for col_i, (ch_idx, color, ch_label) in enumerate([
            (0, S1_COLOR, 'S1  mCherry2'),
            (1, S2_COLOR, 'S2  GFP'),
    ]):
        ax = axes[row_i, col_i]
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_edgecolor(FG_COLOR)

        trace = build_trace(fcs_path, ch_idx)
        if len(trace) == 0:
            ax.set_visible(False)
            continue

        t_s = np.arange(len(trace)) * 1e-3   # ms → s
        ax.plot(t_s, trace, color=color, lw=0.4, alpha=0.85)

        if n_abs is not None:
            ax.axhline(n_abs, color='orange', lw=1.2, ls='--',
                       label=f'n_abs = {n_abs:.3f} kHz')
            ax.legend(fontsize=FS_TICK - 1, framealpha=0.3,
                      labelcolor=FG_COLOR, facecolor=BG_COLOR,
                      edgecolor=FG_COLOR)

        ax.set_title(f'{stem}  {ch_label}', color=FG_COLOR, fontsize=FS_TITLE)
        ax.tick_params(colors=FG_COLOR, labelsize=FS_TICK)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('kHz', color=FG_COLOR, fontsize=FS_LABEL)
        if row_i == n_files - 1:
            ax.set_xlabel('Time (s)', color=FG_COLOR, fontsize=FS_LABEL)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig)
print(f'Saved → {OUT_PATH}')
