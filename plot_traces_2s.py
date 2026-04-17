#!/usr/bin/env python3
"""
plot_traces_2s.py
Plot one 2-second sub-measurement window per file × channel.
6 files × 2 channels → 6-row × 2-col grid, each panel = 2000 ms.

Baseline is estimated per 512ms sliding window (stride 64ms) as the
KDE mode of intensities within each window, drawn as a cyan step function.
The final baseline (median of per-window estimates) is a dashed cyan line.
n_abs (autofluorescence, from noise model) is shown as a dotted red line.

Output: results_01082026/traces_2s.png
"""

import sys, json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file

# ── Paths ──────────────────────────────────────────────────────────────────────
FCS_DIR    = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_PATH   = PROJECT / 'results_01082026' / 'traces_2s.png'
NOISE_JSON = PROJECT / 'results_01082026' / 'noise_levels.json'

FILES = ['nt_dorsal_1', 'nt_dorsal_5', 'nt_dorsal_10',
         'nt_dorsal_11', 'nt_dorsal_15', 'nt_dorsal_18']
CHANNELS = [('S1', 0), ('S2', 1)]

# Which 2-second window to show (0 = first, 1 = second, ...)
WINDOW_IDX = 0
WIN_BINS    = 2000   # 2s at 1ms resolution

# Sliding-window baseline geometry (matches noise model)
SW_MS    = 512   # window length in bins (= ms at 1ms resolution)
STRIDE   = SW_MS // 8   # 64 ms


# ── Baseline estimator ─────────────────────────────────────────────────────────

def kde_mode(seg):
    """Mode of the KDE of seg — most common intensity value."""
    kde = gaussian_kde(seg, bw_method='silverman')
    x   = np.linspace(seg.min(), seg.max(), 2000)
    return float(x[np.argmax(kde(x))])


def sliding_baseline(seg):
    """
    Slide a SW_MS window across seg (stride STRIDE).
    Returns:
      t_centers  : (M,) window-centre times in seconds
      bl_hats    : (M,) per-window KDE-mode baseline (kHz)
    """
    N = len(seg)
    t_centers = []
    bl_hats   = []
    for t0 in range(0, N - SW_MS + 1, STRIDE):
        w = seg[t0 : t0 + SW_MS]
        t_centers.append((t0 + SW_MS / 2) * 1e-3)   # s
        bl_hats.append(kde_mode(w))
    return np.array(t_centers), np.array(bl_hats)


# ── Noise levels (optional) ────────────────────────────────────────────────────
noise_db = {}
if NOISE_JSON.exists():
    with open(NOISE_JSON) as f:
        noise_db = json.load(f)
    print(f'Loaded noise levels from {NOISE_JSON}')
else:
    print('No noise_levels.json found — n_abs lines omitted.')

# ── Aesthetics ─────────────────────────────────────────────────────────────────
BG_COLOR    = 'black'
FG_COLOR    = 'white'
S1_COLOR    = '#00FF00'
S2_COLOR    = '#FF4444'
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12

# ── Plot ───────────────────────────────────────────────────────────────────────
n_files = len(FILES)
fig, axes = plt.subplots(n_files, 2,
                         figsize=(14, n_files * 2.4),
                         facecolor=BG_COLOR, squeeze=False)

fig.suptitle('FCS intensity traces — 2 s windows  |  '
             'cyan: 512 ms sliding baseline  |  red·: n_abs',
             color=FG_COLOR, fontsize=FS_SUPTITLE - 2, y=1.01)

for row_i, stem in enumerate(FILES):
    fcs = parse_fcs_file(FCS_DIR / f'{stem}.fcs')
    print(f'\n{stem}')

    for col_i, (ch_label, ch_idx) in enumerate(CHANNELS):
        ax = axes[row_i, col_i]
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_edgecolor(FG_COLOR)
        ax.tick_params(colors=FG_COLOR, labelsize=FS_TICK)

        color = S1_COLOR if col_i == 0 else S2_COLOR

        i_khz = fcs.intensity_trace(channel=ch_idx)   # full trace, kHz
        if len(i_khz) == 0:
            ax.text(0.5, 0.5, 'No data', color=FG_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{stem}  {ch_label}', color=FG_COLOR, fontsize=FS_TITLE)
            continue

        # Slice the requested 2-second window
        t0  = WINDOW_IDX * WIN_BINS
        t1  = min(t0 + WIN_BINS, len(i_khz))
        seg = i_khz[t0:t1]
        t_s = np.arange(len(seg)) * 1e-3   # ms → s

        ax.plot(t_s, seg, color=color, lw=0.5, alpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_xlim(t_s[0], t_s[-1])

        # ── Sliding-window baseline ───────────────────────────────────────────
        t_centers, bl_hats = sliding_baseline(seg)
        half_stride = STRIDE / 2 * 1e-3   # s

        # Step function: one horizontal segment per window
        for tc, bh in zip(t_centers, bl_hats):
            ax.plot([tc - half_stride, tc + half_stride], [bh, bh],
                    color='cyan', lw=1.5, alpha=0.85, solid_capstyle='butt')

        # Final baseline = median of per-window estimates
        final_bl = float(np.median(bl_hats))
        ax.axhline(final_bl, color='cyan', lw=1.2, ls='--', alpha=0.6,
                   label=f'baseline = {final_bl:.1f} kHz')

        print(f'  {ch_label}: bl range [{bl_hats.min():.1f}, {bl_hats.max():.1f}]  '
              f'median = {final_bl:.1f} kHz')

        # ── n_abs (autofluorescence) ──────────────────────────────────────────
        n_abs = noise_db.get(stem, {}).get(ch_label)
        if n_abs is not None:
            ax.axhline(n_abs, color='red', lw=1.2, ls=':',
                       label=f'n_abs = {n_abs:.1f} kHz')

        ax.legend(fontsize=FS_TICK - 1, framealpha=0.35,
                  labelcolor=FG_COLOR, facecolor=BG_COLOR, edgecolor=FG_COLOR)

        ax.set_title(f'{stem}  {ch_label}', color=FG_COLOR, fontsize=FS_TITLE)
        ax.set_ylabel('kHz', color=FG_COLOR, fontsize=FS_LABEL)
        ax.yaxis.label.set_color(FG_COLOR)
        if row_i == n_files - 1:
            ax.set_xlabel('Time (s)', color=FG_COLOR, fontsize=FS_LABEL)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig)
print(f'\nSaved → {OUT_PATH}')
