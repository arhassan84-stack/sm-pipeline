"""
extract_burst_templates.py
──────────────────────────
Multi-scale burst template extraction from noise-free training simulations.

Each D-bin gets a template of a different length, matched to its natural
burst duration (geometric progression, doubling per bin):

  D=2.5–15     ±64 ms  → 128 ms  total
  D=0.7–2.5   ±128 ms  → 256 ms  total
  D=0.2–0.7   ±256 ms  → 512 ms  total
  D=0.05–0.2  ±512 ms  → 1024 ms total
  D=0.01–0.05 ±1024 ms → 2048 ms total

For each trace in the training set:
  1. Find peak via Gaussian-smoothed trace
  2. Skip if peak is too close to edge for the D-bin's half-width
  3. Crop window, normalize to peak = 1
  4. Accumulate into D-bin average

Two-panel figure:
  Panel A – All templates on a shared normalized x-axis (−1 to +1)
             → shows shape differences independent of scale
  Panel B – All templates on the same absolute time axis (ms)
             → shows scale differences

Templates saved as .npy for use in Stage 3 matched filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE  = BASE / 'cache_i_train.npy'
D_FILE  = BASE / 'cache_d_train.npy'
OUT_FIG = BASE / 'burst_templates.png'
OUT_NPY = BASE / 'burst_templates.npy'

# ── D-bin definitions + per-bin template half-widths ──────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA',   '#66CCEE',  '#228833', '#CCBB44', '#EE6677']
W_HALVES = [1024,         512,        256,        128,       64]   # ms per bin

SIGMA_SMOOTH = 10      # Gaussian smoothing σ for peak finding (ms)
MAX_PER_BIN  = 3000    # max curves stored per bin for percentile bands

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':       'black', 'axes.labelcolor': 'black',
    'xtick.color':      'black', 'ytick.color':     'black',
    'axes.facecolor':   'white', 'figure.facecolor':'white',
})
FS_SUPTITLE=20; FS_TITLE=14; FS_LABEL=15; FS_YLABEL=14
FS_TICK=12;     FS_LEGEND=11; SPINE_LW=1.5

def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_color('black')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=FS_TICK, colors='black')

# ── load data ─────────────────────────────────────────────────────────────────
print('Loading training data …')
i_all = np.load(I_FILE, mmap_mode='r')
d_all = np.load(D_FILE, mmap_mode='r')
N, T  = i_all.shape
print(f'  {N} traces × {T} bins  (dt = 1 ms)')

# ── accumulate per D-bin ──────────────────────────────────────────────────────
W_MAX = max(W_HALVES)   # 1024 — largest half-window needed

template_sum   = {k: np.zeros(2 * W_HALVES[k] + 1) for k in range(len(D_LABELS))}
template_count = {k: 0                               for k in range(len(D_LABELS))}
template_curves = {k: []                              for k in range(len(D_LABELS))}

print('Extracting burst shapes …')
for i in range(N):
    trace = i_all[i].astype(np.float64)
    if trace.sum() < 1:
        continue

    smooth = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
    peak   = int(np.argmax(smooth))

    d_val = d_all[i]
    k     = min(np.searchsorted(D_EDGES[1:], d_val), len(D_LABELS) - 1)
    W     = W_HALVES[k]

    # skip if peak too close to edge for this bin's half-width
    if peak < W or peak + W >= T:
        continue

    segment  = trace[peak - W : peak + W + 1].copy()
    peak_val = segment[W]   # center bin
    if peak_val < 1:
        continue
    segment /= peak_val     # normalize: center → 1.0

    template_sum[k]   += segment
    template_count[k] += 1
    if len(template_curves[k]) < MAX_PER_BIN:
        template_curves[k].append(segment.copy())

    if (i + 1) % 10000 == 0:
        print(f'  … {i+1}/{N}')

print('Done.\n')

# ── compute mean templates ─────────────────────────────────────────────────────
templates = {}   # label → {'mean': array, 'p10': array, 'p90': array, 'w_half': int}
for k, label in enumerate(D_LABELS):
    n = template_count[k]
    W = W_HALVES[k]
    if n == 0:
        print(f'WARNING: no traces for D={label}')
        continue
    mean = template_sum[k] / n
    mat  = np.array(template_curves[k])
    templates[label] = {
        'mean':   mean,
        'p10':    np.percentile(mat, 10, axis=0),
        'p90':    np.percentile(mat, 90, axis=0),
        'w_half': W,
        'n':      n,
    }
    print(f'  D={label:>10}  W=±{W:>4} ms  n={n:>5} traces')

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Multi-scale burst templates per D-bin  (normalized: peak = 1)',
             fontsize=FS_SUPTITLE, color='black')

# ── Panel A: normalized x-axis (shape comparison) ────────────────────────────
ax = axes[0]
for k, (label, color) in enumerate(zip(D_LABELS, D_COLORS)):
    if label not in templates:
        continue
    d      = templates[label]
    W      = d['w_half']
    t_norm = np.linspace(-1, 1, 2 * W + 1)   # −1 → +1 regardless of W
    ax.plot(t_norm, d['mean'], color=color, lw=2.0,
            label=f"D={label}  (±{W} ms)")
    ax.fill_between(t_norm, d['p10'], d['p90'], color=color, alpha=0.12)

ax.axvline(0,    color='gray', lw=0.8, ls='--', alpha=0.7)
ax.axhline(0,    color='gray', lw=0.5, ls='-',  alpha=0.3)
ax.axhline(0.10, color='gray', lw=0.7, ls=':',  alpha=0.6)
ax.axhline(0.50, color='gray', lw=0.7, ls='--', alpha=0.6)
ax.text(0.97,  0.12, '10%', color='gray', fontsize=FS_TICK-1,
        ha='right', transform=ax.get_yaxis_transform())
ax.text(0.97,  0.52, '50%', color='gray', fontsize=FS_TICK-1,
        ha='right', transform=ax.get_yaxis_transform())

ax.set_xlim(-1, 1)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel('Normalised time  (−1 = left edge, 0 = peak, +1 = right edge)',
              fontsize=FS_LABEL)
ax.set_ylabel('Normalised intensity', fontsize=FS_YLABEL)
ax.set_title('Shape comparison\n(each template on its own time scale)',
             fontsize=FS_TITLE)
ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.85, edgecolor='gray')
style_ax(ax)

# ── Panel B: absolute time axis (scale comparison) ────────────────────────────
ax = axes[1]
for k, (label, color) in enumerate(zip(D_LABELS, D_COLORS)):
    if label not in templates:
        continue
    d    = templates[label]
    W    = d['w_half']
    t_ms = np.arange(-W, W + 1, dtype=float)
    ax.plot(t_ms, d['mean'], color=color, lw=2.0,
            label=f"D={label}  (±{W} ms)")
    ax.fill_between(t_ms, d['p10'], d['p90'], color=color, alpha=0.12)

ax.axvline(0,    color='gray', lw=0.8, ls='--', alpha=0.7)
ax.axhline(0,    color='gray', lw=0.5, ls='-',  alpha=0.3)
ax.axhline(0.10, color='gray', lw=0.7, ls=':',  alpha=0.6)
ax.axhline(0.50, color='gray', lw=0.7, ls='--', alpha=0.6)
ax.text(W_MAX * 0.97, 0.12, '10%', color='gray', fontsize=FS_TICK-1, ha='right')
ax.text(W_MAX * 0.97, 0.52, '50%', color='gray', fontsize=FS_TICK-1, ha='right')

ax.set_xlim(-W_MAX, W_MAX)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel('Time relative to peak  (ms)', fontsize=FS_LABEL)
ax.set_ylabel('Normalised intensity', fontsize=FS_YLABEL)
ax.set_title('Scale comparison\n(all templates on the same absolute axis)',
             fontsize=FS_TITLE)
ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.85, edgecolor='gray')
style_ax(ax)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, facecolor='white')
print(f'\nFigure → {OUT_FIG}')

# ── save templates ─────────────────────────────────────────────────────────────
np.save(OUT_NPY, templates)
print(f'Templates → {OUT_NPY}')

# ── summary table ──────────────────────────────────────────────────────────────
print('\n── Template summary ──────────────────────────────────────────────')
print(f'{"D-bin":>12}  {"W_half":>7}  {"n":>6}  '
      f'{"val@edge":>9}  {"val@50%W":>9}  {"val@25%W":>9}')
for k, label in enumerate(D_LABELS):
    if label not in templates:
        continue
    d    = templates[label]
    W    = d['w_half']
    mean = d['mean']
    c    = W   # center index
    print(f'{label:>12}  ±{W:>5} ms  {d["n"]:>6}  '
          f'{mean[0]:>9.3f}  '           # value at template edge
          f'{mean[c - W//2]:>9.3f}  '   # value at 50% of half-width
          f'{mean[c - W//4]:>9.3f}')     # value at 25% of half-width
