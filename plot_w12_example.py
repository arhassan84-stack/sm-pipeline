# -*- coding: utf-8 -*-
"""
plot_w12_example.py
Show the w12 width measurement on the first few Cat-1 peaks of nt_dorsal_1 / S1.

For each selected peak, plots a ±3s window containing:
  - Raw i1ms (green)
  - 50ms box-smoothed trace (black solid)
  - Baseline (black dashed)
  - Horizontal I_cut line (orange) at baseline[tp] + 0.60*(smooth[tp] - baseline[tp])
  - Vertical lines at tp-w12 (blue) and tp+w12 (red)
  - Title showing raw peak, smooth peak, baseline, I_cut, w12, and category
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_DIR  = PROJECT / 'results_01082026' / 'pipeline_test'
sys.path.insert(0, str(PROJECT))

from measurement import load_measurement
from pipeline    import _stage1_baseline, _stage2_subtract

# ── Aesthetics ────────────────────────────────────────────────────────────────
SPINE_LW    = 1.5
FS_TITLE    = 11
FS_LABEL    = 12
FS_TICK     = 10

# ── Stage 3 constants (mirror stage3_v2.py) ───────────────────────────────────
TRNST_LVL        = 0.60
CAT1_MIN_WIDTH_MS = 200
W12_SMOOTH_MS     = 50
W12_MAX_MS        = 2048
PEAK_DISTANCE_MS  = 50
AMP_THRESH_KHZ    = 50.0   # S1

HALF_WIN_MS = 3000   # ±3 s around each peak
N_PEAKS     = 6      # how many Cat-1 peaks to show


class MockRegistry:
    class _DiffModels:
        def best_for(self, **kw): return None
    class _NoiseModels:
        _wrappers = []
        def best_for(self, *a, **kw): return None
    diffusion_models = _DiffModels()
    noise_models     = _NoiseModels()
    dt_min           = 0.1


# ── Load data ────────────────────────────────────────────────────────────────
RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)
meas = load_measurement(
    fcs_path = DATA_DIR / 'nt_dorsal_1.fcs',
    data_dir = DATA_DIR,
    registry = MockRegistry(),
    raw_opts = RAW_OPTS,
)
ch_data  = meas.channels['S1']
i100     = ch_data.i100_full
baseline = _stage1_baseline(i100)
N        = len(i100)


# ── Replicate _step1 peak finding ────────────────────────────────────────────
kernel        = np.ones(W12_SMOOTH_MS, dtype=np.float64) / W12_SMOOTH_MS
signal_smooth = np.convolve(i100, kernel, 'same')

peak_idx, _ = find_peaks(i100, height=baseline, distance=PEAK_DISTANCE_MS)
h_arr       = i100[peak_idx]
peak_idx    = peak_idx[h_arr >= AMP_THRESH_KHZ]


def compute_w12(smooth, tp, bl):
    bl_tp = float(bl[tp])
    I_cut = bl_tp + TRNST_LVL * max(float(smooth[tp]) - bl_tp, 0.0)
    left_idx  = np.where(smooth[:tp] < I_cut)[0]
    t_left_w  = int(left_idx[-1]) if len(left_idx) > 0 else 0
    right_idx = np.where(smooth[tp + 1:] < I_cut)[0]
    t_right_w = int(right_idx[0]) + tp + 1 if len(right_idx) > 0 else N - 1
    w12 = max((t_right_w - t_left_w) // 2, 1)
    return min(w12, W12_MAX_MS), I_cut, t_left_w, t_right_w


cat1_peaks = []
for tp in peak_idx:
    w12, I_cut, tlw, trw = compute_w12(signal_smooth, int(tp), baseline)
    if 2 * w12 >= CAT1_MIN_WIDTH_MS:
        cat1_peaks.append((int(tp), w12, I_cut, tlw, trw))

# Sort by descending amplitude and pick first N_PEAKS
cat1_peaks.sort(key=lambda x: i100[x[0]], reverse=True)
selected = cat1_peaks[:N_PEAKS]
print(f'Found {len(cat1_peaks)} Cat-1 peaks; plotting top {len(selected)}')

# ── Plot ─────────────────────────────────────────────────────────────────────
ncols = 2
nrows = int(np.ceil(len(selected) / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 7, nrows * 3.5),
                          facecolor='white')
fig.suptitle('nt_dorsal_1 / S1 — Cat-1 w12 examples\n'
             'green=raw  black=50ms smooth  black-dashed=baseline  '
             'orange=I_cut  blue/red=±w12',
             fontsize=12, color='black')

t_ms = np.arange(N, dtype=float)

for ax, (tp, w12, I_cut, tlw, trw) in zip(axes.flat, selected):
    t0 = max(0,     tp - HALF_WIN_MS)
    t1 = min(N - 1, tp + HALF_WIN_MS)
    sl = slice(t0, t1 + 1)

    t_seg  = t_ms[sl] * 1e-3
    raw    = i100[sl]
    smooth = signal_smooth[sl]
    bl     = baseline[sl]

    # Raw
    ax.plot(t_seg, raw,    color='#007700', lw=0.8, alpha=0.9, label='raw')
    # Smoothed
    ax.plot(t_seg, smooth, color='black',   lw=1.5, label='50ms smooth')
    # Baseline
    ax.plot(t_seg, bl,     color='black',   lw=1.2, ls='--', label='baseline')
    # I_cut line drawn between the two actual crossing points (t_left_w, t_right_w)
    ax.hlines(I_cut, tlw * 1e-3, trw * 1e-3,
              colors='darkorange', lw=2.0, ls='-', label=f'I_cut={I_cut:.1f}kHz')
    # Dots at the crossing points
    ax.plot([tlw * 1e-3, trw * 1e-3], [I_cut, I_cut],
            'o', color='darkorange', ms=6, zorder=5)
    # Seed window boundaries (tp ± w12, derived from the crossing span)
    ax.axvline((tp - w12) * 1e-3, color='blue', lw=1.8, ls='-')
    ax.axvline((tp + w12) * 1e-3, color='red',  lw=1.8, ls='-')
    # Peak marker
    ax.axvline(tp * 1e-3, color='gray', lw=1.0, ls=':')

    ax.set_facecolor('white')
    ax.set_xlim(t0 * 1e-3, t1 * 1e-3)
    ax.set_ylim(bottom=0, top=max(raw.max(), smooth.max()) * 1.15)
    ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('kHz',      fontsize=FS_LABEL, color='black')
    ax.tick_params(colors='black', labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')

    bl_tp     = float(baseline[tp])
    raw_peak  = float(i100[tp])
    smo_peak  = float(signal_smooth[tp])
    capped    = '(capped)' if w12 == W12_MAX_MS else ''
    ax.set_title(
        f't={tp/1000:.2f}s  raw={raw_peak:.1f}kHz  smooth={smo_peak:.1f}kHz  '
        f'bl={bl_tp:.1f}kHz\n'
        f'I_cut={I_cut:.1f}kHz  w12={w12}ms{capped}  2w12={2*w12}ms',
        fontsize=FS_TITLE, color='black'
    )
    ax.legend(fontsize=8, loc='upper right')

# Hide unused axes
for ax in axes.flat[len(selected):]:
    ax.set_visible(False)

fig.tight_layout()
out = OUT_DIR / 'w12_cat1_examples.png'
fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {out}')
