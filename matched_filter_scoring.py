"""
matched_filter_scoring.py
─────────────────────────
Multi-scale matched filter bank applied to FCS intensity traces.

For each D-bin template T_k (unit-energy normalised):
    score_k(t) = Σ_τ  i100_clean[t + τ] × T̃_k[τ]
               = ||window_t|| × cos(angle between window and T̃_k)

High score requires BOTH:
  • large local amplitude  (real molecular signal)
  • shape match to template (correct D-bin)

Combined response:  R(t)   = max_k  score_k(t)
Winning bin:        k*(t)  = argmax_k score_k(t)

Demo: one representative trace per D-bin from training data.
Each row shows:
  Left  – intensity trace with detected event region shaded
  Right – 5 filter response curves + combined max + detected peak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE   = BASE / 'cache_i_train.npy'
D_FILE   = BASE / 'cache_d_train.npy'
TPL_FILE = BASE / 'burst_templates.npy'
OUT_FIG  = BASE / 'matched_filter_demo.png'

# ── D-bin config ───────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024,         512,        256,        128,       64]

SIGMA_SMOOTH  = 10     # ms — for peak finding in demo traces
PEAK_MIN_DIST = 128    # ms — minimum distance between detected peaks
SEED          = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
})
FS_SUPTITLE=20; FS_TITLE=13; FS_LABEL=13; FS_YLABEL=12
FS_TICK=11;     FS_LEGEND=10; SPINE_LW=1.5

def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=FS_TICK, colors='black')

# ── load templates ─────────────────────────────────────────────────────────────
print('Loading templates …')
templates = np.load(TPL_FILE, allow_pickle=True).item()

# unit-energy normalise each template mean
T_norm = {}
for k, label in enumerate(D_LABELS):
    if label not in templates:
        continue
    t = templates[label]['mean'].copy()
    T_norm[k] = t / np.linalg.norm(t)

# ── matched filter function ────────────────────────────────────────────────────
def run_filter_bank(signal):
    """
    Parameters
    ----------
    signal : 1-D array  (i100_clean, dt=1ms)

    Returns
    -------
    scores   : (n_bins, N) array — per-D-bin score at each time bin (NaN outside valid range)
    combined : (N,) array        — max score across D-bins at each time bin
    winning  : (N,) int array    — index of winning D-bin at each time bin
    """
    N = len(signal)
    n_bins = len(D_LABELS)
    scores = np.full((n_bins, N), np.nan)

    for k in range(n_bins):
        if k not in T_norm:
            continue
        T  = T_norm[k]
        W  = W_HALVES[k]
        # scipy correlate mode='valid': output length = N - len(T) + 1
        raw      = correlate(signal, T, mode='valid')
        t_start  = W                       # first valid center bin
        t_end    = t_start + len(raw)
        scores[k, t_start:t_end] = raw

    combined = np.nanmax(scores, axis=0)   # NaN where ALL templates are NaN

    # nanargmax raises on all-NaN columns; replace with 0 where no template valid
    any_valid = np.any(~np.isnan(scores), axis=0)
    winning   = np.zeros(N, dtype=int)
    winning[any_valid] = np.nanargmax(scores[:, any_valid], axis=0)

    return scores, combined, winning

# ── select one representative trace per D-bin ──────────────────────────────────
print('Loading training data …')
i_all = np.load(I_FILE, mmap_mode='r')
d_all = np.load(D_FILE, mmap_mode='r')

rng = np.random.default_rng(SEED)

demo_traces = []   # list of (trace, d_val, d_bin_idx, trace_idx)
for k in range(len(D_LABELS)):
    mask = np.where(
        (d_all >= D_EDGES[k]) & (d_all < D_EDGES[k+1])
    )[0]
    # pick trace with highest peak intensity — cleanest single event
    candidates = rng.choice(mask, size=min(500, len(mask)), replace=False)
    best = candidates[np.argmax([i_all[c].max() for c in candidates])]
    demo_traces.append((i_all[best].astype(np.float64), d_all[best], k, best))
    print(f'  D={D_LABELS[k]}  trace #{best}  D={d_all[best]:.3f} µm²/s  '
          f'peak={i_all[best].max()/1e3:.1f} kHz')

# ── figure ─────────────────────────────────────────────────────────────────────
n_rows = len(demo_traces)
fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.2 * n_rows))
fig.suptitle('Multi-scale matched filter bank — demo on training traces',
             fontsize=FS_SUPTITLE, color='black', y=1.01)

t_ms = np.arange(4096)   # time axis in ms

for row, (trace, d_val, k_true, idx) in enumerate(demo_traces):
    ax_tr  = axes[row, 0]   # left:  intensity trace
    ax_sc  = axes[row, 1]   # right: filter scores

    # run filter bank
    scores, combined, winning = run_filter_bank(trace)

    # detect peaks in combined response
    threshold = np.nanpercentile(combined, 85)   # adaptive: top 15% of scores
    peaks, _  = find_peaks(combined,
                            height=threshold,
                            distance=PEAK_MIN_DIST)

    # ── left panel: intensity trace ───────────────────────────────────────────
    ax_tr.plot(t_ms, trace / 1e3, color='black', lw=0.8, alpha=0.7)

    # shade detected event regions (±W_HALVES of winning bin at each peak)
    for p in peaks:
        k_win = int(winning[p])
        W_win = W_HALVES[k_win]
        ax_tr.axvspan(max(0, p - W_win), min(4095, p + W_win),
                      color=D_COLORS[k_win], alpha=0.20)
        ax_tr.axvline(p, color=D_COLORS[k_win], lw=1.2, ls='--', alpha=0.8)

    ax_tr.set_xlim(0, 4095)
    ax_tr.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
    ax_tr.set_ylabel('Intensity  (kHz)', fontsize=FS_YLABEL)
    ax_tr.set_title(f'D = {d_val:.3f} µm²/s  (bin: {D_LABELS[k_true]})',
                    fontsize=FS_TITLE, color=D_COLORS[k_true])
    style_ax(ax_tr)

    # ── right panel: filter scores ────────────────────────────────────────────
    for k in range(len(D_LABELS)):
        s = scores[k]
        valid = ~np.isnan(s)
        if valid.sum() == 0:
            continue
        ax_sc.plot(t_ms[valid], s[valid],
                   color=D_COLORS[k], lw=1.2, alpha=0.75,
                   label=f'T: D={D_LABELS[k]}  (±{W_HALVES[k]} ms)')

    # combined max response
    valid_c = ~np.isnan(combined)
    ax_sc.plot(t_ms[valid_c], combined[valid_c],
               color='black', lw=1.8, ls='-', alpha=0.9,
               label='Combined max', zorder=5)

    # threshold line
    ax_sc.axhline(threshold, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax_sc.text(4050, threshold * 1.02, 'thr', color='gray',
               fontsize=FS_TICK - 1, ha='right')

    # mark detected peaks
    for p in peaks:
        ax_sc.axvline(p, color=D_COLORS[int(winning[p])], lw=1.2, ls='--', alpha=0.8)
        ax_sc.scatter([p], [combined[p]], color=D_COLORS[int(winning[p])],
                      s=50, zorder=6)

    ax_sc.set_xlim(0, 4095)
    ax_sc.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
    ax_sc.set_ylabel('Filter response', fontsize=FS_YLABEL)
    ax_sc.set_title(f'Filter bank scores  —  {len(peaks)} peak(s) detected',
                    fontsize=FS_TITLE)
    ax_sc.legend(fontsize=FS_LEGEND, loc='upper right',
                 framealpha=0.85, edgecolor='gray', ncol=2)
    style_ax(ax_sc)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')

# ── print peak summary ─────────────────────────────────────────────────────────
print('\n── Peak detection summary ────────────────────────────────────────')
for row, (trace, d_val, k_true, idx) in enumerate(demo_traces):
    scores, combined, winning = run_filter_bank(trace)
    threshold = np.nanpercentile(combined, 85)
    peaks, _  = find_peaks(combined, height=threshold, distance=PEAK_MIN_DIST)
    print(f'  D={d_val:.3f}  (true bin: {D_LABELS[k_true]:>10})  '
          f'{len(peaks)} peak(s) detected  '
          f'winning bins: {[D_LABELS[int(winning[p])] for p in peaks]}')
