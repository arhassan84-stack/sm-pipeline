"""
compare_filter_approaches.py
────────────────────────────
Compares three matched-filter scoring strategies on the training data.

Approach 0 — Baseline MF
  score_k(t) = dot(window_t, T̃_k)          (unit-energy template)
  winner     = argmax_k score_k(t)

Approach A — NCC  (Normalised Cross-Correlation / Pearson)
  score_k(t) = dot(window_t, T̃_k) / ||window_t||   ∈ [−1, +1]
  winner     = argmax_k score_k(t)
  Pure shape-match: amplitude-invariant

Approach B — Amplitude × NCC
  score_k(t) = NCC_k(t) × amplitude(t)
  amplitude  = gaussian_filter1d(signal, σ=10)[t]   (local smoothed level)
  winner     = argmax_k score_k(t)
  Requires both shape match AND strong local signal

Figure layout
  Row 1–2 : score curves for one slow (D≈0.012) and one fast (D≈7.96) example
             columns = the 3 approaches
  Row 3   : grouped bar chart — classification accuracy per D-bin × approach
             (evaluated on 10,000 randomly sampled training traces)
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
OUT_FIG  = BASE / 'filter_comparison.png'

# ── D-bin config ───────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]

SIGMA_SMOOTH  = 10
PEAK_MIN_DIST = 128
N_VALID       = 10_000
SEED          = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.facecolor':'white','figure.facecolor':'white',
})
FS_SUPTITLE=20; FS_TITLE=12; FS_LABEL=12; FS_YLABEL=11
FS_TICK=10;     FS_LEGEND=9;  SPINE_LW=1.5
A_COLORS = ['#333333', '#E66100', '#5D3A9B']   # baseline, NCC, Amp×NCC
A_LABELS = ['Baseline MF', 'NCC', 'Amplitude × NCC']

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

# pre-compute unit-energy templates
T_norm = {}
for k, label in enumerate(D_LABELS):
    if label in templates:
        t = templates[label]['mean'].copy()
        T_norm[k] = t / np.linalg.norm(t)

# ── filter bank implementations ────────────────────────────────────────────────
def _raw_scores(signal):
    """Unit-energy matched filter scores per D-bin."""
    N_ = len(signal)
    scores = np.full((len(D_LABELS), N_), np.nan)
    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W   = W_HALVES[k]
        raw = correlate(signal, T_norm[k], mode='valid')
        scores[k, W: W + len(raw)] = raw
    return scores

def _local_rms(signal, W):
    """RMS of signal in a sliding window of length 2W+1, centered."""
    sig2   = signal ** 2
    L      = 2 * W + 1
    cs     = np.concatenate([[0.], np.cumsum(sig2)])
    rms    = np.sqrt((cs[L:] - cs[:-L]) / L)   # length N - L + 1
    return rms   # first valid center = W

def run_baseline(signal):
    scores = _raw_scores(signal)
    any_v  = np.any(~np.isnan(scores), axis=0)
    comb   = np.nanmax(scores, axis=0)
    win    = np.zeros(len(signal), dtype=int)
    win[any_v] = np.nanargmax(scores[:, any_v], axis=0)
    return scores, comb, win

def run_ncc(signal):
    """Pearson NCC: divide raw matched filter by local RMS of signal."""
    raw    = _raw_scores(signal)
    N_     = len(signal)
    scores = np.full_like(raw, np.nan)
    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W     = W_HALVES[k]
        rms   = _local_rms(signal, W)        # length = N - 2W
        denom = rms.copy()
        denom[denom < 1e-12] = 1e-12
        valid_slice = slice(W, W + len(rms))
        scores[k, valid_slice] = raw[k, valid_slice] / denom
    any_v  = np.any(~np.isnan(scores), axis=0)
    comb   = np.nanmax(scores, axis=0)
    win    = np.zeros(N_, dtype=int)
    win[any_v] = np.nanargmax(scores[:, any_v], axis=0)
    return scores, comb, win

def run_amp_ncc(signal):
    """Amplitude × NCC: NCC score weighted by local smoothed amplitude."""
    _, ncc_scores, _ = run_ncc(signal)
    amp    = gaussian_filter1d(signal, sigma=SIGMA_SMOOTH)
    amp   /= max(amp.max(), 1e-12)            # normalise to [0,1]
    # rebuild per-bin scores weighted by amplitude for winner selection
    raw    = _raw_scores(signal)
    N_     = len(signal)
    scores = np.full_like(raw, np.nan)
    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W     = W_HALVES[k]
        rms   = _local_rms(signal, W)
        denom = rms.copy(); denom[denom < 1e-12] = 1e-12
        valid_slice = slice(W, W + len(rms))
        ncc_k = raw[k, valid_slice] / denom
        scores[k, valid_slice] = ncc_k * amp[valid_slice]
    any_v  = np.any(~np.isnan(scores), axis=0)
    comb   = np.nanmax(scores, axis=0)
    win    = np.zeros(N_, dtype=int)
    win[any_v] = np.nanargmax(scores[:, any_v], axis=0)
    return scores, comb, win

APPROACHES = [
    ('Baseline MF',       run_baseline,  A_COLORS[0]),
    ('NCC',               run_ncc,       A_COLORS[1]),
    ('Amplitude × NCC',   run_amp_ncc,   A_COLORS[2]),
]

# ── select demo traces: best slow + best fast ──────────────────────────────────
rng = np.random.default_rng(SEED)
demos = []
for k_demo in [0, 4]:   # slowest + fastest D-bin
    mask = np.where((d_all >= D_EDGES[k_demo]) & (d_all < D_EDGES[k_demo+1]))[0]
    cands = rng.choice(mask, size=min(500, len(mask)), replace=False)
    best  = cands[np.argmax([i_all[c].max() for c in cands])]
    demos.append((i_all[best].astype(np.float64), d_all[best], k_demo))
    print(f'Demo trace: D={d_all[best]:.3f}  bin={D_LABELS[k_demo]}  '
          f'peak={i_all[best].max()/1e3:.1f} kHz  idx={best}')

# ── quantitative validation on N_VALID traces ──────────────────────────────────
print(f'\nValidating on {N_VALID} traces …')
val_idx = rng.choice(N, size=N_VALID, replace=False)
# correct[approach_idx][bin_idx] = fraction of correctly classified peaks
correct = np.zeros((3, len(D_LABELS)))
total   = np.zeros(len(D_LABELS), dtype=int)

for ii, gi in enumerate(val_idx):
    trace  = i_all[gi].astype(np.float64)
    d_val  = d_all[gi]
    k_true = min(np.searchsorted(D_EDGES[1:], d_val), len(D_LABELS)-1)
    total[k_true] += 1

    # true event center = peak of smoothed trace
    smooth     = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
    true_peak  = int(np.argmax(smooth))

    for ai, (_, fn, _) in enumerate(APPROACHES):
        _, _, win = fn(trace)
        k_win = int(win[true_peak])
        if k_win == k_true:
            correct[ai, k_true] += 1

    if (ii + 1) % 2000 == 0:
        print(f'  … {ii+1}/{N_VALID}')

acc = correct / np.maximum(total, 1)   # (3, 5)  per-approach per-bin accuracy
print('Done.\n')
print('Classification accuracy at true peak location:')
print(f'{"D-bin":>12}  {"Baseline":>10}  {"NCC":>10}  {"Amp×NCC":>10}')
for k, label in enumerate(D_LABELS):
    print(f'{label:>12}  {acc[0,k]:>10.1%}  {acc[1,k]:>10.1%}  {acc[2,k]:>10.1%}')
print(f'{"MEAN":>12}  {acc[0].mean():>10.1%}  {acc[1].mean():>10.1%}  {acc[2].mean():>10.1%}')

# ── figure ─────────────────────────────────────────────────────────────────────
# Layout: 3 rows × 3 cols
#   rows 0–1: score curves for slow demo (row 0) and fast demo (row 1)
#             cols = the 3 approaches
#   row  2:   accuracy bar chart spanning all columns
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9],
                        hspace=0.45, wspace=0.32)

t_ms = np.arange(T, dtype=float)

for demo_row, (trace, d_val, k_true) in enumerate(demos):
    smooth_sig = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)

    for ai, (a_label, a_fn, a_col) in enumerate(APPROACHES):
        ax = fig.add_subplot(gs[demo_row, ai])

        scores, combined, winning = a_fn(trace)

        # adaptive threshold
        threshold = np.nanpercentile(combined, 85)
        peaks, _  = find_peaks(combined,
                                height=threshold,
                                distance=PEAK_MIN_DIST)

        # plot per-bin scores
        for k in range(len(D_LABELS)):
            s = scores[k]
            v = ~np.isnan(s)
            if v.sum() == 0:
                continue
            ax.plot(t_ms[v], s[v], color=D_COLORS[k], lw=1.0, alpha=0.6,
                    label=f'D={D_LABELS[k]}')

        # combined max
        vc = ~np.isnan(combined)
        ax.plot(t_ms[vc], combined[vc], color='black', lw=1.8, zorder=5,
                label='Combined max')

        # threshold
        ax.axhline(threshold, color='gray', lw=0.8, ls=':', alpha=0.7)

        # detected peaks + winning D-bin colour
        for p in peaks:
            ax.axvline(p, color=D_COLORS[int(winning[p])],
                       lw=1.2, ls='--', alpha=0.75, zorder=6)
            ax.scatter([p], [combined[p]], s=55,
                       color=D_COLORS[int(winning[p])], zorder=7)

        ax.set_xlim(0, T-1)
        ax.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
        ax.set_ylabel('Score', fontsize=FS_YLABEL)
        ax.set_title(
            f'{a_label}\n'
            f'D={d_val:.3f} µm²/s  (true bin: {D_LABELS[k_true]})  '
            f'| {len(peaks)} peak(s)',
            fontsize=FS_TITLE,
            color=D_COLORS[k_true],
        )
        ax.legend(fontsize=FS_LEGEND - 1, loc='upper right',
                  framealpha=0.8, edgecolor='gray', ncol=2)
        style_ax(ax)

# ── accuracy bar chart ─────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, :])
n_bins  = len(D_LABELS)
n_app   = 3
x       = np.arange(n_bins)
w       = 0.25
offsets = [-w, 0, w]

for ai, (a_label, _, a_col) in enumerate(APPROACHES):
    bars = ax_bar.bar(x + offsets[ai], acc[ai] * 100, width=w,
                      color=a_col, alpha=0.80, label=a_label,
                      edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, acc[ai]):
        if v > 0.05:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.0,
                        f'{v:.0%}', ha='center', va='bottom',
                        fontsize=FS_TICK - 1, color='black')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(D_LABELS, fontsize=FS_TICK)
ax_bar.set_ylabel('Classification accuracy  (%)', fontsize=FS_LABEL)
ax_bar.set_xlabel('True D-bin  (µm²/s)', fontsize=FS_LABEL)
ax_bar.set_title(
    f'Winner D-bin classification accuracy at true event peak  '
    f'(n={N_VALID} traces)',
    fontsize=FS_TITLE + 1
)
ax_bar.set_ylim(0, 110)
ax_bar.legend(fontsize=FS_LEGEND + 1, loc='upper right',
              framealpha=0.85, edgecolor='gray')
ax_bar.axhline(100, color='gray', lw=0.5, ls='--', alpha=0.4)
style_ax(ax_bar)

fig.suptitle('Matched filter scoring approach comparison',
             fontsize=FS_SUPTITLE, color='black', y=1.01)

plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')
