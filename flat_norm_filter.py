"""
flat_norm_filter.py
───────────────────
Tests the flat-signal-normalised matched filter (Approach C) against
the Baseline MF.

Flat-norm score:
    score_C_k(t) = dot(window_t, T̃_k)  /  (μ_t  ×  sum(T̃_k))

Interpretation: how much better does template k match the window
than a flat (constant) signal would?  Score=1 → flat signal; score>1 → burst.

All templates produce score=1 on any flat signal, regardless of template
length → removes the length-integration bias that plagued the Baseline MF.

Figure:
  Row 0 – score curves, slow demo (D≈0.012)  ×  2 columns [Baseline | Flat-norm]
  Row 1 – score curves, fast demo (D≈7.96)   ×  2 columns
  Row 2 – classification accuracy bar chart (n=10,000 traces)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE   = BASE / 'cache_i_train.npy'
D_FILE   = BASE / 'cache_d_train.npy'
TPL_FILE = BASE / 'burst_templates.npy'
OUT_FIG  = BASE / 'flat_norm_comparison.png'

# ── config ─────────────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]

SIGMA_SMOOTH  = 10
PEAK_MIN_DIST = 128
N_VALID       = 10_000
REG_FRAC      = 0.005    # local-mean regularisation: fraction of trace max
SEED          = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.facecolor':'white','figure.facecolor':'white',
})
FS_SUPTITLE=20; FS_TITLE=13; FS_LABEL=12; FS_YLABEL=11
FS_TICK=10;     FS_LEGEND=9;  SPINE_LW=1.5

def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=FS_TICK, colors='black')

# ── load data ──────────────────────────────────────────────────────────────────
print('Loading …')
i_all     = np.load(I_FILE, mmap_mode='r')
d_all     = np.load(D_FILE, mmap_mode='r')
templates = np.load(TPL_FILE, allow_pickle=True).item()
N, T      = i_all.shape

# pre-compute unit-energy templates and their flat-signal response sum(T̃_k)
T_norm = {}; sum_T  = {}
for k, label in enumerate(D_LABELS):
    if label in templates:
        t          = templates[label]['mean'].copy()
        T_norm[k]  = t / np.linalg.norm(t)
        sum_T[k]   = T_norm[k].sum()          # expected response to unit flat signal

print('sum(T̃_k) per D-bin:')
for k, label in enumerate(D_LABELS):
    if k in sum_T:
        print(f'  D={label}  W=±{W_HALVES[k]}ms  sum_T={sum_T[k]:.4f}')

# ── sliding local mean via cumsum ───────────────────────────────────────────────
def local_mean(signal, W):
    """Mean of signal in a sliding window of length 2W+1."""
    L  = 2 * W + 1
    cs = np.concatenate([[0.], np.cumsum(signal)])
    return (cs[L:] - cs[:-L]) / L     # length = N - 2W

# ── filter bank functions ───────────────────────────────────────────────────────
def _winner(scores):
    N_ = scores.shape[1]
    any_v = np.any(~np.isnan(scores), axis=0)
    comb  = np.nanmax(scores, axis=0)
    win   = np.zeros(N_, dtype=int)
    win[any_v] = np.nanargmax(scores[:, any_v], axis=0)
    return scores, comb, win

def run_baseline(signal):
    """Unit-energy MF: score = dot(window, T̃)."""
    N_ = len(signal)
    scores = np.full((len(D_LABELS), N_), np.nan)
    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W   = W_HALVES[k]
        raw = correlate(signal, T_norm[k], mode='valid')
        scores[k, W: W + len(raw)] = raw
    return _winner(scores)

def run_flat_norm(signal):
    """
    Flat-normalised MF:
        score_k(t) = dot(window_t, T̃_k) / (μ_t × sum(T̃_k))

    μ_t  = local mean of signal in window [t-W, t+W]
    Regularisation: μ_t clipped to REG_FRAC × signal.max() to avoid ÷0
    """
    N_  = len(signal)
    reg = max(signal.max() * REG_FRAC, 1.0)   # minimum denominator floor
    scores = np.full((len(D_LABELS), N_), np.nan)

    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W    = W_HALVES[k]
        raw  = correlate(signal, T_norm[k], mode='valid')      # length N-2W
        mu   = local_mean(signal, W)                            # length N-2W
        mu_r = np.maximum(mu, reg)                              # regularised
        denom = mu_r * sum_T[k]
        scores[k, W: W + len(raw)] = raw / denom

    return _winner(scores)

APPROACHES = [
    ('Baseline MF',  run_baseline,  '#333333'),
    ('Flat-norm MF', run_flat_norm, '#0077BB'),
]

# ── select demo traces ─────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
demos = []
for k_demo in [0, 4]:
    mask  = np.where((d_all >= D_EDGES[k_demo]) & (d_all < D_EDGES[k_demo+1]))[0]
    cands = rng.choice(mask, size=min(500, len(mask)), replace=False)
    best  = cands[np.argmax([i_all[c].max() for c in cands])]
    demos.append((i_all[best].astype(np.float64), d_all[best], k_demo))

# ── quantitative validation ────────────────────────────────────────────────────
print(f'\nValidating on {N_VALID} traces …')
val_idx = rng.choice(N, size=N_VALID, replace=False)
correct = np.zeros((2, len(D_LABELS)))
total   = np.zeros(len(D_LABELS), dtype=int)

for ii, gi in enumerate(val_idx):
    trace  = i_all[gi].astype(np.float64)
    d_val  = d_all[gi]
    k_true = min(np.searchsorted(D_EDGES[1:], d_val), len(D_LABELS)-1)
    total[k_true] += 1
    smooth    = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
    true_peak = int(np.argmax(smooth))
    for ai, (_, fn, _) in enumerate(APPROACHES):
        _, _, win = fn(trace)
        if int(win[true_peak]) == k_true:
            correct[ai, k_true] += 1
    if (ii + 1) % 2000 == 0:
        print(f'  … {ii+1}/{N_VALID}')

acc = correct / np.maximum(total, 1)
print('\nClassification accuracy at true event peak:')
print(f'{"D-bin":>12}  {"Baseline":>10}  {"Flat-norm":>10}  {"Δ":>8}')
for k, label in enumerate(D_LABELS):
    delta = acc[1,k] - acc[0,k]
    print(f'{label:>12}  {acc[0,k]:>10.1%}  {acc[1,k]:>10.1%}  '
          f'{delta:>+8.1%}')
print(f'{"MEAN":>12}  {acc[0].mean():>10.1%}  {acc[1].mean():>10.1%}  '
      f'{acc[1].mean()-acc[0].mean():>+8.1%}')

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85],
                        hspace=0.50, wspace=0.30)
t_ms = np.arange(T, dtype=float)

for demo_row, (trace, d_val, k_true) in enumerate(demos):
    for ai, (a_label, a_fn, a_col) in enumerate(APPROACHES):
        ax = fig.add_subplot(gs[demo_row, ai])

        scores, combined, winning = a_fn(trace)
        threshold = np.nanpercentile(combined, 85)
        peaks, _  = find_peaks(combined, height=threshold,
                                distance=PEAK_MIN_DIST)

        for k in range(len(D_LABELS)):
            s = scores[k]; v = ~np.isnan(s)
            if v.sum() == 0:
                continue
            ax.plot(t_ms[v], s[v], color=D_COLORS[k], lw=1.0, alpha=0.65,
                    label=f'D={D_LABELS[k]}')

        vc = ~np.isnan(combined)
        ax.plot(t_ms[vc], combined[vc], color='black', lw=1.8, zorder=5,
                label='Combined max')
        ax.axhline(threshold, color='gray', lw=0.8, ls=':', alpha=0.7)

        for p in peaks:
            kw = int(winning[p])
            ax.axvline(p, color=D_COLORS[kw], lw=1.2, ls='--',
                       alpha=0.8, zorder=6)
            ax.scatter([p], [combined[p]], s=60, color=D_COLORS[kw], zorder=7)

        ax.set_xlim(0, T-1)
        ax.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
        ax.set_ylabel('Score', fontsize=FS_YLABEL)
        ax.set_title(
            f'{a_label}\n'
            f'D={d_val:.3f} µm²/s  (true: {D_LABELS[k_true]})  '
            f'| {len(peaks)} peak(s)',
            fontsize=FS_TITLE, color=D_COLORS[k_true])
        ax.legend(fontsize=FS_LEGEND - 1, loc='upper right',
                  framealpha=0.85, edgecolor='gray', ncol=2)
        style_ax(ax)

# ── accuracy bar chart ─────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, :])
x  = np.arange(len(D_LABELS))
w  = 0.32
A_COLORS_BAR = ['#333333', '#0077BB']
offsets      = [-w/2, w/2]

for ai, (a_label, _, _) in enumerate(APPROACHES):
    bars = ax_bar.bar(x + offsets[ai], acc[ai] * 100, width=w,
                      color=A_COLORS_BAR[ai], alpha=0.82,
                      label=a_label, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, acc[ai]):
        if v > 0.05:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.2,
                        f'{v:.0%}', ha='center', va='bottom',
                        fontsize=FS_TICK - 1, color='black')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(D_LABELS, fontsize=FS_TICK)
ax_bar.set_ylabel('Classification accuracy  (%)', fontsize=FS_LABEL)
ax_bar.set_xlabel('True D-bin  (µm²/s)', fontsize=FS_LABEL)
ax_bar.set_title(
    f'Winner D-bin classification accuracy at true event peak  (n={N_VALID})',
    fontsize=FS_TITLE + 1)
ax_bar.set_ylim(0, 115)
ax_bar.legend(fontsize=FS_LEGEND + 2, loc='upper right',
              framealpha=0.85, edgecolor='gray')
ax_bar.axhline(100, color='gray', lw=0.5, ls='--', alpha=0.4)
style_ax(ax_bar)

fig.suptitle('Baseline MF vs Flat-signal-normalised MF',
             fontsize=FS_SUPTITLE, color='black', y=1.01)

plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')
