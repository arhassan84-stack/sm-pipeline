# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
plot_step1_diagnostic.py

For each file x channel, shows consecutive 2-second panels of the intensity
trace.  Every detected peak is drawn with its claimed (or hypothetical) window
shaded:
  - Survived peaks   : coloured by amplitude (viridis)
  - Excluded peaks   : light grey

White background.  No GPU required.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import find_peaks

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_DIR  = PROJECT / 'results_01082026' / 'pipeline_test'

sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file

# ── Algorithm constants (must match stage3_v2.py) ────────────────────────────
TRNST_LVL         = 0.60
CAT1_MIN_WIDTH_MS = 200
W_START2_MS       = 128
PEAK_DISTANCE_MS  = 50
W12_SMOOTH_MS     = 50

# ── Baseline (inlined from pipeline.py) ──────────────────────────────────────
def _histogram_mode(seg, n_bins=128):
    counts, edges = np.histogram(seg, bins=n_bins)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))

def _stage1_baseline(i100, window_ms=512, stride_ms=64):
    N = len(i100)
    t_centers, bl_hats = [], []
    for t0 in range(0, N - window_ms + 1, stride_ms):
        bl_hats.append(_histogram_mode(i100[t0 : t0 + window_ms]))
        t_centers.append(t0 + window_ms / 2)
    if not bl_hats:
        return np.full(N, _histogram_mode(i100))
    return np.interp(np.arange(N, dtype=float),
                     np.array(t_centers), np.array(bl_hats),
                     left=bl_hats[0], right=bl_hats[-1])

# ── w12 (inlined from stage3_v2.py) ──────────────────────────────────────────
def _compute_w12(signal, tp):
    I_cut = TRNST_LVL * signal[tp]
    N     = len(signal)
    left  = np.where(signal[:tp] < I_cut)[0]
    right = np.where(signal[tp + 1:] < I_cut)[0]
    t_l   = int(left[-1])  if len(left)  > 0 else 0
    t_r   = int(right[0]) + tp + 1 if len(right) > 0 else N - 1
    return max((t_r - t_l) // 2, 1)

# ── Step 1 with full diagnostics ─────────────────────────────────────────────
def run_step1_diag(signal, baseline):
    """
    Runs the full Step 1 claiming loop and returns two lists:

    all_peaks : list of dicts, one per detected peak, with keys:
        tp, h_peak, w12, category (1 or 2),
        cl_s, cl_e  (hypothetical claimed window),
        survived    (bool)

    claimed   : bool (N,) array
    """
    N = len(signal)
    peak_idx, _ = find_peaks(signal, height=baseline,
                              distance=PEAK_DISTANCE_MS)
    if len(peak_idx) == 0:
        return [], np.zeros(N, dtype=bool)

    # w12 measured on box-smoothed trace (matches stage3_v2.py)
    kernel        = np.ones(W12_SMOOTH_MS, dtype=float) / W12_SMOOTH_MS
    signal_smooth = np.convolve(signal, kernel, 'same')

    # Classify and build peak list
    peaks = []
    for pid, tp in enumerate(peak_idx):
        h  = float(signal[tp])
        w12 = _compute_w12(signal_smooth, int(tp))
        cat = 1 if 2 * w12 >= CAT1_MIN_WIDTH_MS else 2
        cl_s = max(0, tp - w12) if cat == 1 else max(0, tp - W_START2_MS // 2)
        cl_e = min(N-1, tp + w12) if cat == 1 else min(N-1, tp + W_START2_MS // 2)
        peaks.append(dict(id=pid, tp=int(tp), h_peak=h, w12=w12,
                          category=cat, cl_s=cl_s, cl_e=cl_e, survived=False))

    cat1_raw = sorted([p for p in peaks if p['category'] == 1],
                      key=lambda e: e['h_peak'], reverse=True)
    cat2_raw = sorted([p for p in peaks if p['category'] == 2],
                      key=lambda e: e['h_peak'], reverse=True)

    excluded = set()
    claimed  = np.zeros(N, dtype=bool)

    # Cat-1 claiming
    for ev in cat1_raw:
        if ev['id'] in excluded:
            continue
        claimed[ev['cl_s'] : ev['cl_e'] + 1] = True
        ev['survived'] = True
        for other in cat1_raw + cat2_raw:
            if other['id'] != ev['id'] and ev['cl_s'] <= other['tp'] <= ev['cl_e']:
                excluded.add(other['id'])

    # Cat-2 claiming (vectorised exclusion)
    cat2_tp = np.array([e['tp'] for e in cat2_raw], dtype=np.int64)
    cat2_id = np.array([e['id'] for e in cat2_raw], dtype=np.int64)
    for ev in cat2_raw:
        if ev['id'] in excluded:
            continue
        claimed[ev['cl_s'] : ev['cl_e'] + 1] = True
        ev['survived'] = True
        mask = (cat2_tp >= ev['cl_s']) & (cat2_tp <= ev['cl_e']) & (cat2_id != ev['id'])
        excluded.update(int(i) for i in cat2_id[mask])

    print(f'    peaks={len(peaks)}  cat1={len(cat1_raw)}  cat2={len(cat2_raw)}  '
          f'survived={sum(p["survived"] for p in peaks)}  '
          f'excluded={sum(not p["survived"] for p in peaks)}')
    return peaks, claimed

# ── Plotting ─────────────────────────────────────────────────────────────────
SPINE_LW   = 1.5
FS_TITLE   = 10
FS_LABEL   = 11
FS_TICK    = 9
N_COLS     = 3
N_ROWS     = 6      # 18 panels x 2 s = 36 s
WIN_MS     = 2000
FILES      = ['nt_dorsal_1', 'nt_dorsal_10', 'nt_dorsal_15']
CHANNELS   = ['S1', 'S2']


def make_diag_figure(stem, ch_id, i100, peaks, out_path):
    N     = len(i100)
    t_ms  = np.arange(N, dtype=float)

    # Colour survived peaks by amplitude
    surv = [p for p in peaks if p['survived']]
    if surv:
        h_vals = np.array([p['h_peak'] for p in surv])
        cmap   = cm.viridis
        norm   = Normalize(vmin=h_vals.min(), vmax=h_vals.max())
    else:
        cmap, norm = cm.viridis, Normalize(0, 1)

    n_surv = sum(p['survived'] for p in peaks)
    n_excl = len(peaks) - n_surv

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                              figsize=(N_COLS * 5, N_ROWS * 2),
                              facecolor='white')
    fig.suptitle(
        f'{stem} / {ch_id}  —  {n_surv} survived  |  {n_excl} excluded',
        fontsize=13, color='black',
    )

    for idx, ax in enumerate(axes.flat):
        t0 = idx * WIN_MS
        t1 = t0 + WIN_MS
        if t0 >= N:
            ax.set_visible(False)
            continue

        seg   = i100[t0 : min(t1, N)]
        t_seg = t_ms[t0 : t0 + len(seg)]

        ax.set_facecolor('white')

        # --- shade windows for ALL peaks in this time range ---
        for p in peaks:
            if p['cl_e'] < t0 or p['cl_s'] > t1:
                continue
            sl = max(p['cl_s'], t0)
            sr = min(p['cl_e'], t1)
            if p['survived']:
                c     = cmap(norm(p['h_peak']))
                alpha = 0.30
            else:
                c     = (0.7, 0.7, 0.7)   # grey
                alpha = 0.20
            ax.axvspan(sl * 1e-3, sr * 1e-3, color=c, alpha=alpha, lw=0)

        # --- peak markers ---
        for p in peaks:
            if not (t0 <= p['tp'] <= t1):
                continue
            if p['survived']:
                c  = cmap(norm(p['h_peak']))
                lw = 0.9
                ls = '--'
            else:
                c  = (0.55, 0.55, 0.55)
                lw = 0.6
                ls = ':'
            ax.axvline(p['tp'] * 1e-3, color=c, lw=lw, ls=ls, alpha=0.85)

        # --- intensity trace on top ---
        ax.plot(t_seg * 1e-3, seg, color='black', lw=0.7, zorder=5)

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_color('black')
        ax.tick_params(colors='black', labelsize=FS_TICK)
        ax.set_xlim(t0 * 1e-3, t1 * 1e-3)
        ax.set_title(f'{t0/1000:.0f}–{t1/1000:.0f} s',
                     fontsize=FS_TITLE, color='black', pad=2)

        col = idx % N_COLS
        row = idx // N_COLS
        if col == 0:
            ax.set_ylabel('kHz', fontsize=FS_TICK, color='black')
        if row == N_ROWS - 1:
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')

    # Legend + colorbar
    grey_patch  = mpatches.Patch(color=(0.7, 0.7, 0.7), alpha=0.5,
                                  label='excluded peak window')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label('Peak amplitude (kHz)', fontsize=FS_LABEL, color='black')
    cbar.ax.yaxis.set_tick_params(color='black', labelsize=FS_TICK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')
    fig.legend(handles=[grey_patch], fontsize=FS_TICK,
               loc='lower left', framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path.name}')


# ── Main ─────────────────────────────────────────────────────────────────────
for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    if not fcs_path.exists():
        print(f'Missing: {fcs_path}'); continue
    print(f'\n{stem}')
    fcs = parse_fcs_file(fcs_path)

    for ch_idx, ch_id in enumerate(CHANNELS):
        print(f'  {ch_id}', end=' ... ', flush=True)
        i100     = fcs.intensity_trace(channel=ch_idx)
        baseline = _stage1_baseline(i100)
        peaks, _ = run_step1_diag(i100, baseline)

        out = OUT_DIR / f'{stem}_{ch_id}_step1_diag.png'
        make_diag_figure(stem, ch_id, i100, peaks, out)

print('\nDone.')
