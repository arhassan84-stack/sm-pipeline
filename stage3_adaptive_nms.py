"""
stage3_adaptive_nms.py
──────────────────────
Replaces fixed PEAK_MIN_DIST with adaptive Non-Maximum Suppression (NMS):
after a peak is accepted at t_peak with winning template k_win,
all other candidate peaks within ±W_HALVES[k_win] are suppressed.

This prevents over-detection of multiple spurious peaks within a single
broad slow-diffuser burst while leaving well-separated fast-diffuser
spikes untouched.

Performance evaluated on N_STATS traces per D-bin:
  - Each training trace contains exactly ONE molecule event
  - True event centre = argmax of Gaussian-smoothed trace
  - Metrics: exact detection (n=1, within tolerance), over-detection, miss

Figure:
  Top    – 5×5 grid of example traces (adaptive NMS)
  Bottom – performance comparison bars (fixed vs adaptive NMS)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE   = BASE / 'cache_i_train.npy'
D_FILE   = BASE / 'cache_d_train.npy'
TPL_FILE = BASE / 'burst_templates.npy'
OUT_FIG  = BASE / 'stage3_adaptive_nms.png'

# ── config ─────────────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]

W_D                = 4096
SIGMA_SMOOTH       = 10
PEAK_THRESH_PCTILE = 85
PEAK_MIN_RAW       = 10     # minimal distance for initial raw peak finding
FIXED_MIN_DIST     = 128    # baseline fixed suppression distance
CAP_VALUES         = [128, 256, 512]   # suppression radius caps to sweep

MERGE_CLOSE_EVENTS = True   # merge adjacent events whose gap < min(W_seg) of the pair

# ── adaptive window expansion ──────────────────────────────────────────────────
ADAPTIVE_EXPAND    = True   # expand window outward until signal drops to threshold
EXPAND_THRESH_FRAC = 0.10   # stop level = bg + frac × (peak − bg)
EXPAND_SIGMA       = 30     # smoothing sigma used for expansion decision (ms)
EXPAND_MAX_MULT    = 3      # hard cap: at most ±MULT×W_seg from peak
BG_PCTILE          = 10     # percentile of trace used as background estimate

N_PER_BIN  = 5
N_STATS    = 500
SEED       = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.facecolor':'white','figure.facecolor':'white',
})
FS_SUPTITLE=16; FS_TITLE=9; FS_LABEL=10; FS_YLABEL=9
FS_TICK=8;      FS_LEGEND=8; SPINE_LW=1.2

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

# ── fixed NMS (baseline) ───────────────────────────────────────────────────────
def detect_fixed(combined, winning, threshold):
    peaks, _ = find_peaks(combined, height=threshold, distance=FIXED_MIN_DIST)
    return peaks

# ── adaptive NMS ───────────────────────────────────────────────────────────────
def detect_adaptive(combined, winning, threshold, radius_cap=None):
    """
    Greedy NMS with suppression radius = min(W_HALVES[k_win], radius_cap).
    radius_cap=None means uncapped (original adaptive NMS).
    1. Find all local maxima above threshold (minimal raw distance=10ms)
    2. Sort by score descending
    3. Accept highest peak; suppress all candidates within ±radius
    4. Repeat on remaining candidates
    """
    raw_peaks, _ = find_peaks(combined, height=threshold, distance=PEAK_MIN_RAW)
    if len(raw_peaks) == 0:
        return np.array([], dtype=int)

    scores_at = combined[raw_peaks]
    order      = np.argsort(scores_at)[::-1]
    candidates = raw_peaks[order].tolist()
    active     = [True] * len(candidates)
    accepted   = []

    for i, p in enumerate(candidates):
        if not active[i]:
            continue
        accepted.append(p)
        k_win  = int(winning[p])
        radius = W_HALVES[k_win]
        if radius_cap is not None:
            radius = min(radius, radius_cap)
        for j, q in enumerate(candidates):
            if active[j] and j != i and abs(q - p) <= radius:
                active[j] = False

    return np.array(sorted(accepted), dtype=int)

# ── segment extraction ─────────────────────────────────────────────────────────
def extract(signal, t_peak, W_seg):
    N_  = len(signal)
    t_l = max(0, t_peak - W_seg)
    t_r = min(N_ - 1, t_peak + W_seg)
    L   = t_r - t_l + 1
    pad_frac = max(0.0, (W_D - L) / W_D) if L < W_D else 0.0
    n_win = 1 if L <= W_D else max(1, (L - W_D) // (W_D // 8) + 1)
    return t_l, t_r, pad_frac, n_win

# ── adaptive window expansion ──────────────────────────────────────────────────
def expand_window(signal, t_peak, t_left, t_right, W_seg):
    """
    Walk outward from the template window [t_left, t_right] until the
    Gaussian-smoothed signal drops to bg + EXPAND_THRESH_FRAC*(peak−bg).
    Hard cap: t_peak ± EXPAND_MAX_MULT*W_seg.
    Returns (new_t_left, new_t_right).
    """
    N_     = len(signal)
    smooth = gaussian_filter1d(signal.astype(np.float64), sigma=EXPAND_SIGMA)
    bg     = np.percentile(signal, BG_PCTILE)
    stop   = bg + EXPAND_THRESH_FRAC * max(smooth[t_peak] - bg, 0.0)
    cap_l  = max(0,      t_peak - EXPAND_MAX_MULT * W_seg)
    cap_r  = min(N_ - 1, t_peak + EXPAND_MAX_MULT * W_seg)

    left = t_left
    while left > cap_l and smooth[left - 1] > stop:
        left -= 1

    right = t_right
    while right < cap_r and smooth[right + 1] > stop:
        right += 1

    return left, right


# ── event merging ──────────────────────────────────────────────────────────────
def merge_events(events):
    """
    Merge adjacent demarcated events when the gap between their windows is
    smaller than the half-width of the narrower window.

    Rationale: if two events are 'closer to each other than to the outer edge
    of their own windows', they are likely part of the same molecule traversal.

    Merge rule:
      gap = t_left_B − t_right_A  (< 0 means overlap)
      merge iff gap < min(W_seg_A, W_seg_B)
    Result: union window, peak = higher-scoring centre, k_win = winner's k_win.
    Iterates until no further merges are possible.
    """
    if len(events) < 2:
        return events
    events = sorted(events, key=lambda e: e['t_peak'])
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(events):
            if i + 1 < len(events):
                A, B = events[i], events[i + 1]
                gap  = B['t_left'] - A['t_right']
                if gap < min(A['W_seg'], B['W_seg']):
                    # pick the higher-scoring peak as the representative
                    win = A if A['score'] >= B['score'] else B
                    merged.append(dict(
                        t_peak  = win['t_peak'],
                        t_left  = min(A['t_left'],  B['t_left']),
                        t_right = max(A['t_right'], B['t_right']),
                        W_seg   = win['W_seg'],
                        pad_frac= win['pad_frac'],
                        n_win   = win['n_win'],
                        k_win   = win['k_win'],
                        score   = win['score'],
                    ))
                    i += 2
                    changed = True
                    continue
            merged.append(events[i])
            i += 1
        events = merged
    return events


def run_stage3(signal, mode='adaptive', radius_cap=None):
    """
    mode: 'fixed'    – fixed PEAK_MIN_DIST suppression
          'adaptive' – adaptive NMS (radius = W_HALVES[k_win], optionally capped)
    """
    combined, winning = filter_bank(signal)
    threshold = np.nanpercentile(combined, PEAK_THRESH_PCTILE)
    if mode == 'fixed':
        peaks = detect_fixed(combined, winning, threshold)
    else:
        peaks = detect_adaptive(combined, winning, threshold, radius_cap=radius_cap)
    events = []
    N_ = len(signal)
    for p in peaks:
        k_win = int(winning[p])
        W_seg = W_HALVES[k_win]
        t_l   = max(0,      int(p) - W_seg)
        t_r   = min(N_ - 1, int(p) + W_seg)
        # signal-adaptive expansion
        if ADAPTIVE_EXPAND:
            t_l, t_r = expand_window(signal, int(p), t_l, t_r, W_seg)
        L  = t_r - t_l + 1
        pf = max(0.0, (W_D - L) / W_D) if L < W_D else 0.0
        nw = 1 if L <= W_D else max(1, (L - W_D) // (W_D // 8) + 1)
        events.append(dict(t_peak=int(p), t_left=t_l, t_right=t_r,
                           W_seg=W_seg, pad_frac=pf, n_win=nw,
                           k_win=k_win, score=float(combined[p])))
    if MERGE_CLOSE_EVENTS:
        events = merge_events(events)
    return events, combined

# ── select traces ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
fig_idx  = {}
stat_idx = {}
for k in range(len(D_LABELS)):
    mask   = np.where((d_all >= D_EDGES[k]) & (d_all < D_EDGES[k+1]))[0]
    chosen = rng.choice(mask, size=min(N_STATS + N_PER_BIN, len(mask)),
                        replace=False)
    fig_idx[k]  = chosen[:N_PER_BIN]
    stat_idx[k] = chosen[N_PER_BIN: N_PER_BIN + N_STATS]

# ── performance evaluation ─────────────────────────────────────────────────────
print(f'Evaluating on {N_STATS} traces per D-bin …')

# build method list: fixed + uncapped adaptive + capped variants
methods = [('fixed', 'Fixed 128ms', dict(mode='fixed'))]
methods += [('adaptive', 'Adaptive (uncapped)', dict(mode='adaptive', radius_cap=None))]
methods += [(f'cap{c}', f'Adaptive cap={c}ms', dict(mode='adaptive', radius_cap=c))
            for c in CAP_VALUES]

results = {m[0]: {k: defaultdict(list) for k in range(len(D_LABELS))}
           for m in methods}

for k in range(len(D_LABELS)):
    for idx in stat_idx[k]:
        trace  = i_all[idx].astype(np.float64)
        smooth = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
        t_true = int(np.argmax(smooth))
        tol    = W_HALVES[k]

        for mkey, _, mkwargs in methods:
            evs, _ = run_stage3(trace, **mkwargs)
            n      = len(evs)
            results[mkey][k]['n_events'].append(n)
            if n >= 1:
                errs = [abs(ev['t_peak'] - t_true) for ev in evs]
                best = min(errs)
                results[mkey][k]['localisation_err'].append(best)
                results[mkey][k]['exact'].append(int(n == 1 and best <= tol))
            else:
                results[mkey][k]['localisation_err'].append(np.nan)
                results[mkey][k]['exact'].append(0)

    print(f'  D={D_LABELS[k]} done')

# aggregate
for mkey, _, _ in methods:
    for k in range(len(D_LABELS)):
        ne  = np.array(results[mkey][k]['n_events'])
        ex  = np.array(results[mkey][k]['exact'])
        loc = np.array(results[mkey][k]['localisation_err'])
        results[mkey][k]['pct_exact'] = ex.mean()
        results[mkey][k]['pct_over']  = (ne > 1).mean()
        results[mkey][k]['pct_miss']  = (ne == 0).mean()
        results[mkey][k]['loc_med']   = np.nanmedian(loc)

# ── summary table ──────────────────────────────────────────────────────────────
col_w = 11
print(f'\n{"D-bin":>12}  ' +
      ''.join(f'{lbl:>{col_w}}' for _, lbl, _ in methods))
for metric, label in [('pct_exact','Exact %'), ('pct_over','Over %'), ('loc_med','Loc(ms)')]:
    print(f'\n  {label}')
    for k in range(len(D_LABELS)):
        row = f'  {D_LABELS[k]:>12}  '
        for mkey, _, _ in methods:
            v = results[mkey][k][metric]
            row += f'{v*100 if "pct" in metric else v:>{col_w}.1f}'
        print(row)

# mean exact across D-bins
print(f'\n  Mean exact %')
row = f'  {"MEAN":>12}  '
for mkey, _, _ in methods:
    v = np.mean([results[mkey][k]['pct_exact'] for k in range(len(D_LABELS))])
    row += f'{v*100:>{col_w}.1f}'
print(row)

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1.0],
                        hspace=0.40)

# top: 5×5 grid of traces (adaptive NMS)
gs_top = gridspec.GridSpecFromSubplotSpec(
    len(D_LABELS), N_PER_BIN, subplot_spec=gs[0],
    hspace=0.55, wspace=0.30)

t_ms = np.arange(T_TRACE, dtype=float)

for row, k in enumerate(range(len(D_LABELS))):
    for col, idx in enumerate(fig_idx[k]):
        ax    = fig.add_subplot(gs_top[row, col])
        trace = i_all[idx].astype(np.float64)
        d_val = d_all[idx]

        events, _ = run_stage3(trace, mode='adaptive')

        ax.plot(t_ms, trace / 1e3, color='black', lw=0.6, alpha=0.75)
        for ev in events:
            c = D_COLORS[ev['k_win']]
            ax.axvspan(ev['t_left'], ev['t_right'], color=c, alpha=0.22)
            ax.axvline(ev['t_peak'], color=c, lw=0.9, ls='--', alpha=0.8)

        if col == 0:
            ax.set_ylabel(f'D={D_LABELS[k]}\n(kHz)', fontsize=FS_YLABEL,
                          color=D_COLORS[k])
        ax.set_title(f'D={d_val:.3f}  n={len(events)}ev',
                     fontsize=FS_TITLE, color=D_COLORS[k])
        ax.set_xlim(0, T_TRACE - 1)
        ax.set_xlabel('ms', fontsize=FS_LABEL)
        style_ax(ax)

# bottom: performance comparison bars
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[1], wspace=0.35)

bar_metrics = [
    ('pct_exact', 'Exact detection rate\n(n=1, within tolerance)', 'higher is better'),
    ('pct_over',  'Over-detection rate\n(n>1 events)',             'lower is better'),
    ('loc_med',   'Localisation error (ms)\n(median |t_det − t_true|)', 'lower is better'),
]
# colour palette for all methods
M_COLORS = ['#555555', '#0077BB', '#EE7733', '#009988', '#CC3311']
x = np.arange(len(D_LABELS))
n_m = len(methods)
w   = 0.80 / n_m   # bar width so all fit

for col, (metric, title, hint) in enumerate(bar_metrics):
    ax = fig.add_subplot(gs_bot[col])
    for mi, (mkey, lbl, _) in enumerate(methods):
        offset = (mi - (n_m - 1) / 2) * w
        vals   = [results[mkey][k][metric] for k in range(len(D_LABELS))]
        bars   = ax.bar(x + offset, vals, width=w * 0.9,
                        color=M_COLORS[mi], alpha=0.85, label=lbl,
                        edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > (0.03 if 'pct' in metric else 3):
                fmt = f'{v:.0%}' if 'pct' in metric else f'{v:.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.008 if 'pct' in metric else 1.5),
                        fmt, ha='center', va='bottom',
                        fontsize=max(FS_TICK - 2, 6), color='black', rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(D_LABELS, fontsize=FS_TICK, rotation=15, ha='right')
    ax.set_title(f'{title}\n({hint})', fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND - 1, framealpha=0.85, edgecolor='gray',
              loc='upper right', ncol=1)
    if 'pct' in metric:
        ax.set_ylim(0, 1.25)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    style_ax(ax)

fig.suptitle('Stage 3 — Adaptive NMS  vs  Fixed suppression distance\n'
             f'(top: adaptive NMS examples  |  bottom: performance on {N_STATS} traces/bin)',
             fontsize=FS_SUPTITLE, y=1.01)

plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')
