"""
stage3_detection.py
────────────────────
Stage 3 pipeline: multi-scale matched filter → event detection → segment
extraction → padding/windowing to W_D ready for the D model.

The filter bank's job is purely detection and demarcation:
  1. Slide 5 templates across i100_clean → combined response R(t)
  2. Find peaks in R(t) → event centres {t_peak}
  3. For each peak: winning template k* → segment half-width W* = W_HALVES[k*]
  4. Extract i100_clean[t_peak − W* : t_peak + W* + 1]
  5. Pad / window to W_D bins → D-model input

D-bin classification from the winner template is NOT used downstream —
Stage 4 (D model) handles all diffusion coefficient estimation.

Figure (2 rows, one per demo trace):
  Left  — full intensity trace with detected segments shaded
  Right — combined filter response R(t) with detected peaks marked
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
OUT_FIG  = BASE / 'stage3_detection.png'

# ── config ─────────────────────────────────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024, 512, 256, 128, 64]   # template half-widths (ms)

W_D           = 4096    # D model input length (ms at dt=1ms)
SIGMA_SMOOTH  = 10      # ms — smoothing for peak finding
PEAK_MIN_DIST = 128     # ms — minimum separation between detected events
PEAK_THRESH_PCTILE = 85 # combined response percentile used as detection threshold
SEED          = 42

# ── aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.facecolor':'white','figure.facecolor':'white',
})
FS_SUPTITLE=20; FS_TITLE=13; FS_LABEL=13; FS_YLABEL=12
FS_TICK=11;     FS_LEGEND=10; SPINE_LW=1.5
SEG_ALPHA = 0.20    # shading transparency for demarcated segments

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

# unit-energy templates
T_norm = {}
for k, label in enumerate(D_LABELS):
    if label in templates:
        t = templates[label]['mean'].copy()
        T_norm[k] = t / np.linalg.norm(t)

# ── core filter bank ───────────────────────────────────────────────────────────
def filter_bank(signal):
    """
    Returns
    -------
    combined : (N,) — max matched-filter score across all templates at each bin
    winning  : (N,) int — index of highest-scoring template at each bin
    """
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

# ── segment extraction (§3.4 padding / windowing) ─────────────────────────────
def extract_segment(signal, t_peak, W_seg):
    """
    Extract segment of half-width W_seg around t_peak from signal.
    Apply padding / windowing to match W_D for D-model input.

    Returns
    -------
    d_inputs : list of 1-D arrays, each length W_D  — D-model input windows
    t_left   : int — left boundary in signal coordinates
    t_right  : int — right boundary in signal coordinates
    pad_frac : float — fraction of D-model input that is zero-padded
    """
    N_   = len(signal)
    t_l  = max(0,    t_peak - W_seg)
    t_r  = min(N_-1, t_peak + W_seg)
    seg  = signal[t_l : t_r + 1].copy()
    L    = len(seg)

    if L < W_D:                          # Case 1: pad symmetrically
        pad_total = W_D - L
        pad_l     = pad_total // 2
        pad_r     = pad_total - pad_l
        padded    = np.concatenate([np.zeros(pad_l), seg, np.zeros(pad_r)])
        d_inputs  = [padded]
        pad_frac  = pad_total / W_D

    elif L == W_D:                       # Case 2: exact fit
        d_inputs = [seg]
        pad_frac = 0.0

    else:                                # Case 3: sliding windows
        stride   = W_D // 8
        d_inputs = []
        start    = 0
        while start + W_D <= L:
            d_inputs.append(seg[start : start + W_D])
            start += stride
        pad_frac = 0.0

    return d_inputs, t_l, t_r, pad_frac

# ── Stage 3 pipeline (one trace) ──────────────────────────────────────────────
def run_stage3(signal):
    """
    Full Stage 3: detection → demarcation → extraction.

    Returns list of dicts, one per detected event:
      t_peak, t_left, t_right, W_seg, pad_frac, d_inputs, k_win
    """
    combined, winning = filter_bank(signal)

    # adaptive detection threshold
    threshold = np.nanpercentile(combined, PEAK_THRESH_PCTILE)
    peaks, _  = find_peaks(combined, height=threshold,
                            distance=PEAK_MIN_DIST)

    events = []
    for p in peaks:
        k_win   = int(winning[p])
        W_seg   = W_HALVES[k_win]
        d_inputs, t_l, t_r, pf = extract_segment(signal, int(p), W_seg)
        events.append({
            't_peak'   : int(p),
            't_left'   : t_l,
            't_right'  : t_r,
            'W_seg'    : W_seg,
            'pad_frac' : pf,
            'n_windows': len(d_inputs),
            'd_inputs' : d_inputs,    # ready for D model
            'k_win'    : k_win,
            'score'    : float(combined[p]),
        })
    return events, combined, winning

# ── select demo traces: one slow, one fast ─────────────────────────────────────
rng   = np.random.default_rng(SEED)
demos = []
for k_demo in [0, 4]:
    mask  = np.where((d_all >= D_EDGES[k_demo]) & (d_all < D_EDGES[k_demo+1]))[0]
    cands = rng.choice(mask, size=min(500, len(mask)), replace=False)
    best  = cands[np.argmax([i_all[c].max() for c in cands])]
    demos.append((i_all[best].astype(np.float64), float(d_all[best]), k_demo, int(best)))

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(demos), 2, figsize=(16, 5 * len(demos)))
fig.suptitle('Stage 3 — Event detection and segment demarcation',
             fontsize=FS_SUPTITLE, color='black', y=1.01)

t_ms = np.arange(T_TRACE, dtype=float)

for row, (trace, d_val, k_true, idx) in enumerate(demos):
    ax_tr = axes[row, 0]
    ax_sc = axes[row, 1]

    events, combined, winning = run_stage3(trace)

    print(f'\nTrace #{idx}  D={d_val:.3f} µm²/s  (bin: {D_LABELS[k_true]})')
    print(f'  {len(events)} event(s) detected:')
    for ev in events:
        print(f'    t_peak={ev["t_peak"]:4d} ms  '
              f'segment=[{ev["t_left"]:4d},{ev["t_right"]:4d}] ms  '
              f'W_seg=±{ev["W_seg"]} ms  '
              f'pad={ev["pad_frac"]:.0%}  '
              f'D-windows={ev["n_windows"]}  '
              f'(winning template: D={D_LABELS[ev["k_win"]]})')

    # ── left: intensity trace + shaded segments ───────────────────────────────
    ax_tr.plot(t_ms, trace / 1e3, color='black', lw=0.8, alpha=0.7,
               label='i₁₀₀ (kHz)')

    for ev in events:
        kw    = ev['k_win']
        color = D_COLORS[kw]
        ax_tr.axvspan(ev['t_left'], ev['t_right'],
                      color=color, alpha=SEG_ALPHA)
        ax_tr.axvline(ev['t_peak'], color=color, lw=1.3, ls='--', alpha=0.85)
        ax_tr.text(ev['t_peak'], ax_tr.get_ylim()[1] if ax_tr.get_ylim()[1] > 0 else 1,
                   f"±{ev['W_seg']}ms\n{ev['n_windows']}×W_D",
                   fontsize=FS_TICK - 2, ha='center', va='top',
                   color=color, clip_on=True)

    ax_tr.set_xlim(0, T_TRACE - 1)
    ax_tr.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
    ax_tr.set_ylabel('Intensity  (kHz)', fontsize=FS_YLABEL)
    ax_tr.set_title(
        f'D = {d_val:.3f} µm²/s  (true bin: {D_LABELS[k_true]})  '
        f'— {len(events)} event(s) detected',
        fontsize=FS_TITLE, color=D_COLORS[k_true])

    # legend for segment shading
    from matplotlib.patches import Patch
    handles = [Patch(color=D_COLORS[k], alpha=0.4,
                     label=f'Template ±{W_HALVES[k]}ms')
               for k in range(len(D_LABELS))]
    ax_tr.legend(handles=handles, fontsize=FS_LEGEND,
                 loc='upper right', framealpha=0.85, edgecolor='gray')
    style_ax(ax_tr)

    # ── right: combined filter response ───────────────────────────────────────
    vc = ~np.isnan(combined)
    ax_sc.plot(t_ms[vc], combined[vc], color='black', lw=1.5,
               label='Combined max R(t)')
    ax_sc.fill_between(t_ms[vc], 0, combined[vc], color='black', alpha=0.08)

    threshold = np.nanpercentile(combined, PEAK_THRESH_PCTILE)
    ax_sc.axhline(threshold, color='gray', lw=1.0, ls='--', alpha=0.7,
                  label=f'Threshold  ({PEAK_THRESH_PCTILE}th pctile)')

    for ev in events:
        kw    = ev['k_win']
        color = D_COLORS[kw]
        ax_sc.axvline(ev['t_peak'], color=color, lw=1.3, ls='--', alpha=0.85)
        ax_sc.scatter([ev['t_peak']], [ev['score']], s=70,
                      color=color, zorder=6,
                      label=f"t={ev['t_peak']}ms  ±{ev['W_seg']}ms")
        # shade the demarcated region on the response plot too
        ax_sc.axvspan(ev['t_left'], ev['t_right'], color=color, alpha=0.10)

    ax_sc.set_xlim(0, T_TRACE - 1)
    ax_sc.set_xlabel('Time  (ms)', fontsize=FS_LABEL)
    ax_sc.set_ylabel('Filter response  R(t)', fontsize=FS_YLABEL)
    ax_sc.set_title('Combined matched-filter response', fontsize=FS_TITLE)
    ax_sc.legend(fontsize=FS_LEGEND, loc='upper right',
                 framealpha=0.85, edgecolor='gray')
    style_ax(ax_sc)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, facecolor='white', bbox_inches='tight')
print(f'\nFigure → {OUT_FIG}')
