"""
stage3_batch.py
────────────────
Runs Stage 3 detection on N_PER_BIN traces per D-bin.

Figure: 5 rows (D-bins) × N_PER_BIN columns
  Each panel: intensity trace (kHz) with detected segments shaded by
  winning template width. Detected event centres marked with dashed lines.

Summary statistics printed per D-bin:
  - fraction of traces with ≥1 event detected
  - mean ± std of events per trace
  - distribution of winning template widths
  - distribution of pad fractions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE   = BASE / 'cache_i_train.npy'
D_FILE   = BASE / 'cache_d_train.npy'
TPL_FILE = BASE / 'burst_templates.npy'
OUT_FIG  = BASE / 'stage3_batch.png'

# ── config ─────────────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]

W_D           = 4096
SIGMA_SMOOTH  = 10
PEAK_MIN_DIST = 128
PEAK_THRESH_PCTILE = 85
N_PER_BIN     = 5        # traces shown per D-bin in figure
N_STATS       = 200      # traces per D-bin used for statistics
SEED          = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.facecolor':'white','figure.facecolor':'white',
})
FS_SUPTITLE=18; FS_TITLE=10; FS_LABEL=10; FS_YLABEL=9
FS_TICK=8;      FS_LEGEND=7;  SPINE_LW=1.2

def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=FS_TICK, colors='black')

# ── load ───────────────────────────────────────────────────────────────────────
print('Loading …')
i_all     = np.load(I_FILE, mmap_mode='r')
d_all     = np.load(D_FILE, mmap_mode='r')
templates = np.load(TPL_FILE, allow_pickle=True).item()
N, T_TRACE = i_all.shape

T_norm = {}
for k, label in enumerate(D_LABELS):
    if label in templates:
        t = templates[label]['mean'].copy()
        T_norm[k] = t / np.linalg.norm(t)

# ── filter bank ────────────────────────────────────────────────────────────────
def filter_bank(signal):
    N_ = len(signal)
    scores = np.full((len(D_LABELS), N_), np.nan)
    for k in range(len(D_LABELS)):
        if k not in T_norm:
            continue
        W   = W_HALVES[k]
        raw = correlate(signal, T_norm[k], mode='valid')
        scores[k, W: W + len(raw)] = raw
    any_v    = np.any(~np.isnan(scores), axis=0)
    combined = np.nanmax(scores, axis=0)
    winning  = np.zeros(N_, dtype=int)
    winning[any_v] = np.nanargmax(scores[:, any_v], axis=0)
    return combined, winning

def extract_segment(signal, t_peak, W_seg):
    N_  = len(signal)
    t_l = max(0,    t_peak - W_seg)
    t_r = min(N_-1, t_peak + W_seg)
    seg = signal[t_l : t_r + 1]
    L   = len(seg)
    if L < W_D:
        pad_frac = (W_D - L) / W_D
        n_win    = 1
    elif L == W_D:
        pad_frac = 0.0
        n_win    = 1
    else:
        stride   = W_D // 8
        n_win    = max(1, (L - W_D) // stride + 1)
        pad_frac = 0.0
    return t_l, t_r, pad_frac, n_win

def run_stage3(signal):
    combined, winning = filter_bank(signal)
    threshold = np.nanpercentile(combined, PEAK_THRESH_PCTILE)
    peaks, _  = find_peaks(combined, height=threshold, distance=PEAK_MIN_DIST)
    events = []
    for p in peaks:
        k_win = int(winning[p])
        W_seg = W_HALVES[k_win]
        t_l, t_r, pf, nw = extract_segment(signal, int(p), W_seg)
        events.append(dict(t_peak=int(p), t_left=t_l, t_right=t_r,
                           W_seg=W_seg, pad_frac=pf, n_win=nw,
                           k_win=k_win, score=float(combined[p])))
    return events, combined

# ── select traces ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

# for figure: N_PER_BIN random traces per bin
fig_traces  = {}   # k -> list of trace indices
# for stats:  N_STATS  random traces per bin
stat_traces = {}

for k in range(len(D_LABELS)):
    mask = np.where((d_all >= D_EDGES[k]) & (d_all < D_EDGES[k+1]))[0]
    chosen = rng.choice(mask, size=min(N_STATS + N_PER_BIN, len(mask)),
                        replace=False)
    fig_traces[k]  = chosen[:N_PER_BIN]
    stat_traces[k] = chosen[N_PER_BIN: N_PER_BIN + N_STATS]

# ── figure ─────────────────────────────────────────────────────────────────────
n_rows = len(D_LABELS)
n_cols = N_PER_BIN
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.2 * n_cols, 2.8 * n_rows))
fig.suptitle('Stage 3 detection — sample traces across D-bins  '
             f'(shading = detected segment, colour = winning template)',
             fontsize=FS_SUPTITLE, y=1.01)

t_ms = np.arange(T_TRACE, dtype=float)

for row, k in enumerate(range(len(D_LABELS))):
    for col, idx in enumerate(fig_traces[k]):
        ax    = axes[row, col]
        trace = i_all[idx].astype(np.float64)
        d_val = d_all[idx]

        events, combined = run_stage3(trace)

        # intensity trace
        ax.plot(t_ms, trace / 1e3, color='black', lw=0.6, alpha=0.75)

        # detected segments
        for ev in events:
            c = D_COLORS[ev['k_win']]
            ax.axvspan(ev['t_left'], ev['t_right'], color=c, alpha=0.22)
            ax.axvline(ev['t_peak'], color=c, lw=0.9, ls='--', alpha=0.8)

        # row label (left-most column only)
        if col == 0:
            ax.set_ylabel(f'D={D_LABELS[k]}\n(kHz)', fontsize=FS_YLABEL,
                          color=D_COLORS[k])

        ax.set_title(f'D={d_val:.3f}  n={len(events)}ev',
                     fontsize=FS_TITLE, color=D_COLORS[k])
        ax.set_xlim(0, T_TRACE - 1)
        ax.set_xlabel('ms', fontsize=FS_LABEL)
        style_ax(ax)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'Figure → {OUT_FIG}')

# ── statistics over N_STATS traces per bin ─────────────────────────────────────
print(f'\n── Stage 3 statistics ({N_STATS} traces / D-bin) ─────────────────────────')

# template width labels for display
W_LABELS = [f'±{w}ms' for w in W_HALVES]

for k in range(len(D_LABELS)):
    n_events_list = []
    pad_list      = []
    w_seg_counts  = defaultdict(int)
    detected      = 0

    for idx in stat_traces[k]:
        trace  = i_all[idx].astype(np.float64)
        events, _ = run_stage3(trace)
        n = len(events)
        n_events_list.append(n)
        if n > 0:
            detected += 1
        for ev in events:
            pad_list.append(ev['pad_frac'])
            w_seg_counts[ev['W_seg']] += 1

    ne  = np.array(n_events_list)
    pad = np.array(pad_list) if pad_list else np.array([0.])

    print(f'\n  D = {D_LABELS[k]}')
    print(f'    Detection rate : {detected/N_STATS:.0%}  '
          f'({detected}/{N_STATS} traces with ≥1 event)')
    print(f'    Events/trace   : {ne.mean():.2f} ± {ne.std():.2f}  '
          f'(min={ne.min()} max={ne.max()})')
    print(f'    Pad fraction   : {pad.mean():.0%} ± {pad.std():.0%}  '
          f'(median {np.median(pad):.0%})')
    print(f'    Winning template widths:')
    for w in W_HALVES:
        cnt = w_seg_counts.get(w, 0)
        total_ev = sum(w_seg_counts.values())
        bar = '█' * int(30 * cnt / max(total_ev, 1))
        print(f'      ±{w:>4}ms : {cnt:>4}  ({cnt/max(total_ev,1):>5.1%})  {bar}')
