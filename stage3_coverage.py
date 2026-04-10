"""
stage3_coverage.py
──────────────────
Quantifies how well the detected segment covers the actual burst signal,
comparing template-only windows against signal-adaptive expansion.

Adaptive expansion (option 2):
  After the template sets t_left / t_right, walk outward until the
  Gaussian-smoothed trace drops to:
      stop_level = bg + EXPAND_THRESH_FRAC × (smooth[t_peak] − bg)
  Hard cap at ±EXPAND_MAX_MULT × W_seg from t_peak.

Metrics per detected event:
  1. Photon coverage
       Σ (signal − bg).clip(0) [t_left:t_right]
     / Σ (signal − bg).clip(0) [t_peak ± LOCAL_MULT·W_seg]
  2. Width ratio = actual window width / smoothed-trace FWHM

Figure:
  Top    – worst-coverage (template-only) examples per D-bin
             light green = template window
             dark green  = expansion gain
             red         = FWHM extent still outside expanded window
  Bottom – before / after box plots of photon coverage per D-bin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE   = BASE / 'cache_i_train.npy'
D_FILE   = BASE / 'cache_d_train.npy'
TPL_FILE = BASE / 'burst_templates.npy'
OUT_FIG  = BASE / 'stage3_coverage.png'

# ── config ─────────────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]

W_D                = 4096
SIGMA_SMOOTH       = 10
PEAK_THRESH_PCTILE = 85
PEAK_MIN_RAW       = 10
RADIUS_CAP         = 128
MERGE_CLOSE_EVENTS = True

# ── adaptive expansion ─────────────────────────────────────────────────────────
ADAPTIVE_EXPAND    = True   # toggle: False = template-only, True = + expansion
EXPAND_THRESH_FRAC = 0.10   # expand until smooth < bg + frac*(peak−bg)
EXPAND_SIGMA       = 30     # smoothing sigma used for expansion decision (ms)
EXPAND_MAX_MULT    = 3      # hard cap: expand at most ±MULT×W_seg from peak

# ── coverage analysis ──────────────────────────────────────────────────────────
BG_PCTILE  = 10
LOCAL_MULT = 3      # local region for denominator = ±LOCAL_MULT×W_seg

N_STATS    = 500
N_EXAMPLES = 5      # worst-coverage examples shown per D-bin
SEED       = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
})
FS_SUPTITLE = 20; FS_TITLE = 9; FS_LABEL = 12; FS_YLABEL = 10
FS_TICK = 8;      FS_LEGEND = 8; SPINE_LW = 1.5

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

# ── detection ──────────────────────────────────────────────────────────────────
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

def detect_adaptive(combined, winning, threshold):
    raw_peaks, _ = find_peaks(combined, height=threshold, distance=PEAK_MIN_RAW)
    if len(raw_peaks) == 0:
        return np.array([], dtype=int)
    order      = np.argsort(combined[raw_peaks])[::-1]
    candidates = raw_peaks[order].tolist()
    active     = [True] * len(candidates)
    accepted   = []
    for i, p in enumerate(candidates):
        if not active[i]:
            continue
        accepted.append(p)
        radius = min(W_HALVES[int(winning[p])], RADIUS_CAP)
        for j, q in enumerate(candidates):
            if active[j] and j != i and abs(q - p) <= radius:
                active[j] = False
    return np.array(sorted(accepted), dtype=int)

# ── adaptive window expansion ──────────────────────────────────────────────────
def expand_window(trace, t_peak, t_left, t_right, W_seg):
    """
    Walk outward from the template window until the smoothed signal drops
    to bg + EXPAND_THRESH_FRAC × (peak − bg).
    Hard cap: ±EXPAND_MAX_MULT × W_seg from t_peak.
    Returns (new_t_left, new_t_right).
    """
    N_      = len(trace)
    smooth  = gaussian_filter1d(trace.astype(np.float64), sigma=EXPAND_SIGMA)
    bg      = np.percentile(trace, BG_PCTILE)
    stop    = bg + EXPAND_THRESH_FRAC * max(smooth[t_peak] - bg, 0.0)
    cap_l   = max(0,      t_peak - EXPAND_MAX_MULT * W_seg)
    cap_r   = min(N_ - 1, t_peak + EXPAND_MAX_MULT * W_seg)

    left = t_left
    while left > cap_l and smooth[left - 1] > stop:
        left -= 1

    right = t_right
    while right < cap_r and smooth[right + 1] > stop:
        right += 1

    return left, right

def merge_events(events):
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
                    win = A if A['score'] >= B['score'] else B
                    merged.append(dict(
                        t_peak=win['t_peak'], k_win=win['k_win'],
                        score=win['score'],   W_seg=win['W_seg'],
                        pad_frac=win['pad_frac'], n_win=win['n_win'],
                        t_left_tpl=min(A['t_left_tpl'], B['t_left_tpl']),
                        t_right_tpl=max(A['t_right_tpl'], B['t_right_tpl']),
                        t_left=min(A['t_left'],  B['t_left']),
                        t_right=max(A['t_right'], B['t_right']),
                    ))
                    i += 2; changed = True; continue
            merged.append(events[i])
            i += 1
        events = merged
    return events

def run_stage3(signal):
    combined, winning = filter_bank(signal)
    threshold = np.nanpercentile(combined, PEAK_THRESH_PCTILE)
    peaks     = detect_adaptive(combined, winning, threshold)
    N_  = len(signal)
    events = []
    for p in peaks:
        k_win = int(winning[p])
        W_seg = W_HALVES[k_win]
        tpl_l = max(0, int(p) - W_seg)
        tpl_r = min(N_ - 1, int(p) + W_seg)
        # signal-adaptive expansion
        if ADAPTIVE_EXPAND:
            exp_l, exp_r = expand_window(signal, int(p), tpl_l, tpl_r, W_seg)
        else:
            exp_l, exp_r = tpl_l, tpl_r
        L  = exp_r - exp_l + 1
        pf = max(0.0, (W_D - L) / W_D) if L < W_D else 0.0
        nw = 1 if L <= W_D else max(1, (L - W_D) // (W_D // 8) + 1)
        events.append(dict(
            t_peak=int(p), k_win=k_win, score=float(combined[p]),
            W_seg=W_seg, pad_frac=pf, n_win=nw,
            t_left_tpl=tpl_l, t_right_tpl=tpl_r,   # template-only boundaries
            t_left=exp_l,     t_right=exp_r,         # final (possibly expanded)
        ))
    if MERGE_CLOSE_EVENTS:
        events = merge_events(events)
    return events

# ── coverage metrics ────────────────────────────────────────────────────────────
def event_metrics(trace, ev):
    t_peak = ev['t_peak']
    W_seg  = ev['W_seg']
    N_     = len(trace)
    bg     = np.percentile(trace, BG_PCTILE)
    burst  = np.maximum(trace.astype(np.float64) - bg, 0.0)

    loc_l  = max(0, t_peak - LOCAL_MULT * W_seg)
    loc_r  = min(N_ - 1, t_peak + LOCAL_MULT * W_seg)
    local_ph  = burst[loc_l : loc_r + 1].sum()

    # coverage for template window
    tpl_ph = burst[ev['t_left_tpl'] : ev['t_right_tpl'] + 1].sum()
    cov_tpl = tpl_ph / max(local_ph, 1.0)

    # coverage for expanded window
    exp_ph  = burst[ev['t_left'] : ev['t_right'] + 1].sum()
    cov_exp = exp_ph / max(local_ph, 1.0)

    # FWHM
    smooth   = gaussian_filter1d(trace.astype(np.float64), sigma=SIGMA_SMOOTH)
    peak_val = smooth[t_peak]
    half_h   = bg + 0.5 * max(peak_val - bg, 0.0)
    lfw = t_peak
    while lfw > 0 and smooth[lfw] >= half_h:
        lfw -= 1
    rfw = t_peak
    while rfw < N_ - 1 and smooth[rfw] >= half_h:
        rfw += 1
    fwhm = rfw - lfw

    return cov_tpl, cov_exp, fwhm, lfw, rfw, bg

# ── select traces ───────────────────────────────────────────────────────────────
rng      = np.random.default_rng(SEED)
stat_idx = {}
for k in range(len(D_LABELS)):
    mask   = np.where((d_all >= D_EDGES[k]) & (d_all < D_EDGES[k + 1]))[0]
    stat_idx[k] = rng.choice(mask, size=min(N_STATS, len(mask)), replace=False)

# ── run analysis ────────────────────────────────────────────────────────────────
print(f'Running coverage analysis on {N_STATS} traces per D-bin …')

cov_tpl_data = {k: [] for k in range(len(D_LABELS))}
cov_exp_data = {k: [] for k in range(len(D_LABELS))}
example_pool = {k: [] for k in range(len(D_LABELS))}   # sorted by cov_tpl asc

for k in range(len(D_LABELS)):
    for idx in stat_idx[k]:
        trace  = i_all[idx].astype(np.float64)
        events = run_stage3(trace)
        if not events:
            continue
        smooth = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
        t_true = int(np.argmax(smooth))
        ev     = min(events, key=lambda e: abs(e['t_peak'] - t_true))
        cov_t, cov_e, fwhm, lfw, rfw, bg = event_metrics(trace, ev)
        cov_tpl_data[k].append(cov_t)
        cov_exp_data[k].append(cov_e)
        example_pool[k].append((cov_t, int(idx), ev))

    example_pool[k].sort(key=lambda x: x[0])   # worst template coverage first

    arr_t = np.array(cov_tpl_data[k])
    arr_e = np.array(cov_exp_data[k])
    print(f'  D={D_LABELS[k]}'
          f'  template: median={np.median(arr_t):.1%}  <80%={( arr_t<0.80).sum()}'
          f'  |  expanded: median={np.median(arr_e):.1%}  <80%={(arr_e<0.80).sum()}'
          f'  (n={len(arr_t)})')

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.8, 1.0], hspace=0.45)

# ── top: worst-template-coverage traces per D-bin ──────────────────────────────
gs_top = gridspec.GridSpecFromSubplotSpec(
    len(D_LABELS), N_EXAMPLES, subplot_spec=gs[0], hspace=0.65, wspace=0.30)

t_ms = np.arange(T_TRACE, dtype=float)

for row, k in enumerate(range(len(D_LABELS))):
    for col, (cov_t, idx, ev) in enumerate(example_pool[k][:N_EXAMPLES]):
        ax    = fig.add_subplot(gs_top[row, col])
        trace = i_all[idx].astype(np.float64)
        cov_t2, cov_e, fwhm, lfw, rfw, bg = event_metrics(trace, ev)

        ax.plot(t_ms, trace / 1e3, color='black', lw=0.55, alpha=0.75)
        ax.axhline(bg / 1e3, color='gray', lw=0.6, ls=':', alpha=0.5)

        # template window (light green)
        ax.axvspan(ev['t_left_tpl'], ev['t_right_tpl'],
                   color='#228833', alpha=0.18, label='Template')

        # expansion gain (dark green, shown only where window exceeded template)
        if ev['t_left'] < ev['t_left_tpl']:
            ax.axvspan(ev['t_left'], ev['t_left_tpl'],
                       color='#004400', alpha=0.35, label='Expansion')
        if ev['t_right'] > ev['t_right_tpl']:
            ax.axvspan(ev['t_right_tpl'], ev['t_right'],
                       color='#004400', alpha=0.35)

        # FWHM still uncaptured after expansion (red)
        if lfw < ev['t_left']:
            ax.axvspan(lfw, ev['t_left'], color='#EE6677', alpha=0.35,
                       label='Uncaptured')
        if rfw > ev['t_right']:
            ax.axvspan(ev['t_right'], rfw, color='#EE6677', alpha=0.35)

        ax.axvline(ev['t_peak'], color=D_COLORS[k], lw=0.9, ls='--', alpha=0.85)

        tpl_w = ev['t_right_tpl'] - ev['t_left_tpl']
        exp_w = ev['t_right']     - ev['t_left']
        ax.set_title(
            f'tpl={cov_t2:.0%}→exp={cov_e:.0%}\n'
            f'win {tpl_w}→{exp_w}ms',
            fontsize=FS_TITLE, color=D_COLORS[k])
        if col == 0:
            ax.set_ylabel(f'D={D_LABELS[k]}\n(kHz)', fontsize=FS_YLABEL,
                          color=D_COLORS[k])
        ax.set_xlim(0, T_TRACE - 1)
        ax.set_xlabel('ms', fontsize=FS_LABEL)
        style_ax(ax)

# ── bottom: before / after coverage box plots ──────────────────────────────────
ax_cov = fig.add_subplot(gs[1])

x  = np.arange(len(D_LABELS))
w  = 0.30
bp_kw = dict(
    patch_artist=True,
    medianprops=dict(color='black', lw=1.8),
    whiskerprops=dict(color='black', lw=1.0),
    capprops=dict(color='black', lw=1.0),
    flierprops=dict(marker='.', markersize=2, linestyle='none'),
    widths=w,
)

for k in range(len(D_LABELS)):
    # template-only (lighter, left)
    if cov_tpl_data[k]:
        bp = ax_cov.boxplot(cov_tpl_data[k], positions=[x[k] - w * 0.6],
                            **bp_kw)
        bp['boxes'][0].set(facecolor=D_COLORS[k], alpha=0.30)
        bp['fliers'][0].set(markerfacecolor=D_COLORS[k],
                            markeredgecolor=D_COLORS[k])

    # expanded (solid, right)
    if cov_exp_data[k]:
        bp = ax_cov.boxplot(cov_exp_data[k], positions=[x[k] + w * 0.6],
                            **bp_kw)
        bp['boxes'][0].set(facecolor=D_COLORS[k], alpha=0.70)
        bp['fliers'][0].set(markerfacecolor=D_COLORS[k],
                            markeredgecolor=D_COLORS[k])

ax_cov.axhline(0.90, color='gray', lw=1.0, ls='--', alpha=0.65,
               label='90 % threshold')
ax_cov.axhline(1.00, color='gray', lw=0.6, ls=':', alpha=0.35)

# legend proxies
from matplotlib.patches import Patch
ax_cov.legend(
    handles=[
        Patch(facecolor='gray', alpha=0.30, label='Template window'),
        Patch(facecolor='gray', alpha=0.70, label='Expanded window'),
        plt.Line2D([0], [0], color='gray', lw=1.0, ls='--', label='90 % threshold'),
    ],
    fontsize=FS_LEGEND + 1, loc='lower left', framealpha=0.85, edgecolor='gray')

ax_cov.set_xticks(x)
ax_cov.set_xticklabels(D_LABELS, fontsize=FS_TICK + 1)
ax_cov.set_ylabel('Photon coverage fraction', fontsize=FS_LABEL)
ax_cov.set_title(
    f'Burst photon coverage: template-only (light) vs adaptive expansion (solid)\n'
    f'(stop level = bg + {EXPAND_THRESH_FRAC:.0%}·peak,  '
    f'cap = ±{EXPAND_MAX_MULT}·W_seg,  '
    f'expand σ = {EXPAND_SIGMA} ms)',
    fontsize=FS_LABEL)
ax_cov.set_ylim(0.3, 1.08)
ax_cov.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
style_ax(ax_cov)

fig.suptitle(
    f'Stage 3 — Segment coverage: template window vs signal-adaptive expansion\n'
    f'Top: worst template-coverage traces per D-bin  |  '
    f'light green = template,  dark green = expansion gain,  red = still uncaptured',
    fontsize=FS_SUPTITLE, y=1.01)

plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')
