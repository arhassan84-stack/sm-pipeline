"""
analyze_burst_expansion.py
──────────────────────────
For every simulated noise-free trace in the training set:
  1. Find the peak bin (center of the molecule event)
  2. Compute the cumulative photon sum expanding symmetrically outward:
       S(t) = sum( i[peak-t : peak+t+1] )   for t = 0, 1, 2, …
  3. Normalize: F(t) = S(t) / S(T_max)  ∈ [0, 1]

Then stratify by D and extract:
  - Median cumulative-fraction curve  F̃(t | D-bin)
  - t_90, t_95, t_99 per trace (expansion radius capturing 90/95/99% of photons)
  - Total photon count per trace  (in actual photon counts = sum(i) × dt_ms × 1e-3)

Produces a 3-panel figure:
  Panel A  – Cumulative fraction curves F(t) per D-bin (median ± 10/90 pct)
  Panel B  – Distribution of t_95  (ms) per D-bin
  Panel C  – Distribution of total photons per D-bin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = Path('/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_')
I_FILE = BASE / 'cache_i_train.npy'
D_FILE = BASE / 'cache_d_train.npy'
OUT    = BASE / 'burst_expansion_analysis.png'

# ── constants ───────────────────────────────────────────────────────────────
DT_MS        = 1.0          # ms per bin
DT_S         = DT_MS * 1e-3 # seconds per bin  (for photon count conversion)
SIGMA_SMOOTH = 10           # Gaussian smoothing for peak finding (bins = ms)
MAX_RADIUS   = 512          # maximum expansion radius (ms) to compute curves for
N_SAMPLE     = 50_000       # use all traces (or reduce for speed)
SEED         = 42

# ── aesthetics ──────────────────────────────────────────────────────────────
CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_YLABEL   = 14
FS_TICK     = 12

# ── D-bin definitions (log10 space) ─────────────────────────────────────────
D_EDGES  = [0.01, 0.05, 0.2, 0.7, 2.5, 15.0]   # µm²/s
D_LABELS = ['0.01–0.05', '0.05–0.2', '0.2–0.7', '0.7–2.5', '2.5–15']
D_COLORS = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677']

# ── load data ────────────────────────────────────────────────────────────────
print('Loading training data …')
i_all = np.load(I_FILE, mmap_mode='r')   # (N, 4096)  counts/s
d_all = np.load(D_FILE, mmap_mode='r')   # (N,)       µm²/s

rng = np.random.default_rng(SEED)
idx = rng.choice(len(i_all), size=min(N_SAMPLE, len(i_all)), replace=False)
idx.sort()

# ── per-trace computation ────────────────────────────────────────────────────
print(f'Computing cumulative expansion curves for {len(idx)} traces …')

T = i_all.shape[1]          # 4096
R = min(MAX_RADIUS, T // 2) # expansion radius range

# Outputs
t90  = np.zeros(len(idx), dtype=np.float32)  # expansion radius for 90% photons
t95  = np.zeros(len(idx), dtype=np.float32)
t99  = np.zeros(len(idx), dtype=np.float32)
n_ph = np.zeros(len(idx), dtype=np.float32)  # total photons in trace
d_arr = np.zeros(len(idx), dtype=np.float64)

# Store curves for a random subset for Panel A (memory-friendly)
CURVE_SAMPLE = 5000
curve_idx    = rng.choice(len(idx), size=min(CURVE_SAMPLE, len(idx)), replace=False)
curve_idx.sort()
curves_by_dbin = {k: [] for k in range(len(D_LABELS))}  # list of F(t) arrays

for n, gi in enumerate(idx):
    trace = i_all[gi].astype(np.float64)      # (4096,)  counts/s
    d_val = d_all[gi]

    # total photon count
    total_counts = trace.sum() * DT_S          # photons
    n_ph[n]  = total_counts
    d_arr[n] = d_val

    if total_counts < 1:
        t90[n] = t95[n] = t99[n] = 0
        continue

    # find peak via smoothed trace
    smooth  = gaussian_filter1d(trace, sigma=SIGMA_SMOOTH)
    peak    = int(np.argmax(smooth))

    # cumulative sum expanding symmetrically from peak
    S = np.empty(R + 1, dtype=np.float64)
    S[0] = trace[peak]
    for r in range(1, R + 1):
        left_val  = trace[peak - r] if (peak - r) >= 0 else 0.0
        right_val = trace[peak + r] if (peak + r) < T  else 0.0
        S[r] = S[r-1] + left_val + right_val

    # normalize
    S_max = S[-1]
    if S_max < 1e-12:
        t90[n] = t95[n] = t99[n] = R
        continue
    F = S / S_max

    # find radii where F first exceeds thresholds
    t90[n] = float(np.searchsorted(F, 0.90))
    t95[n] = float(np.searchsorted(F, 0.95))
    t99[n] = float(np.searchsorted(F, 0.99))

    # store curve for Panel A (subset only)
    if n in curve_idx:
        d_bin = np.searchsorted(D_EDGES[1:], d_val)
        d_bin = min(d_bin, len(D_LABELS) - 1)
        curves_by_dbin[d_bin].append(F.copy())

    if (n + 1) % 5000 == 0:
        print(f'  … {n+1}/{len(idx)}')

print('Done.')

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black'})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Burst expansion analysis — training data (noise-free, dt=1ms)',
             fontsize=FS_SUPTITLE, color='black')

t_axis = np.arange(R + 1, dtype=float)   # ms

# ── Panel A: cumulative fraction curves per D-bin ────────────────────────────
ax = axes[0]
for k, (label, color) in enumerate(zip(D_LABELS, D_COLORS)):
    curves = curves_by_dbin[k]
    if len(curves) == 0:
        continue
    mat  = np.array(curves)              # (n_curves, R+1)
    med  = np.median(mat, axis=0)
    p10  = np.percentile(mat, 10, axis=0)
    p90  = np.percentile(mat, 90, axis=0)
    ax.plot(t_axis, med, color=color, lw=1.8, label=f'D={label} µm²/s')
    ax.fill_between(t_axis, p10, p90, color=color, alpha=0.15)

for thr, ls in [(0.90, '--'), (0.95, ':'), (0.99, '-.')]:
    ax.axhline(thr, color='gray', lw=0.8, ls=ls, alpha=0.8)
    ax.text(R * 0.98, thr + 0.005, f'{int(thr*100)}%', color='gray',
            fontsize=FS_TICK - 1, ha='right', va='bottom')

ax.set_xlim(0, R)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Expansion radius t  (ms)', fontsize=FS_LABEL)
ax.set_ylabel('Cumulative photon fraction F(t)', fontsize=FS_YLABEL)
ax.set_title('Cumulative photon capture vs expansion radius\n(median ± 10/90 pct)',
             fontsize=FS_TITLE)
ax.legend(fontsize=FS_TICK - 1, loc='lower right')
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LW)
    spine.set_color('black')
ax.tick_params(labelsize=FS_TICK, colors='black')

# ── Panel B: distribution of t_95 per D-bin ──────────────────────────────────
ax = axes[1]
bins_t = np.linspace(0, MAX_RADIUS, 80)
for k, (label, color) in enumerate(zip(D_LABELS, D_COLORS)):
    mask = (d_arr >= D_EDGES[k]) & (d_arr < D_EDGES[k + 1])
    if mask.sum() == 0:
        continue
    vals = t95[mask]
    ax.hist(vals, bins=bins_t, color=color, alpha=0.55, label=f'D={label}',
            density=True, histtype='stepfilled', edgecolor='none')
    ax.axvline(np.median(vals), color=color, lw=1.5, ls='--')

ax.set_xlabel('t₉₅  (ms) — radius capturing 95% of photons', fontsize=FS_LABEL)
ax.set_ylabel('Density', fontsize=FS_YLABEL)
ax.set_title('Distribution of t₉₅ per D-bin\n(dashed = median)', fontsize=FS_TITLE)
ax.legend(fontsize=FS_TICK - 1)
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LW)
    spine.set_color('black')
ax.tick_params(labelsize=FS_TICK, colors='black')

# ── Panel C: distribution of total photons per D-bin ─────────────────────────
ax = axes[2]
log_edges = np.linspace(np.log10(max(n_ph.min(), 1)), np.log10(n_ph.max() + 1), 80)
for k, (label, color) in enumerate(zip(D_LABELS, D_COLORS)):
    mask = (d_arr >= D_EDGES[k]) & (d_arr < D_EDGES[k + 1])
    if mask.sum() == 0:
        continue
    vals = np.log10(np.maximum(n_ph[mask], 1))
    ax.hist(vals, bins=log_edges, color=color, alpha=0.55, label=f'D={label}',
            density=True, histtype='stepfilled', edgecolor='none')
    ax.axvline(np.median(vals), color=color, lw=1.5, ls='--')

ax.set_xlabel('log₁₀(total photons in trace)', fontsize=FS_LABEL)
ax.set_ylabel('Density', fontsize=FS_YLABEL)
ax.set_title('Total photon count distribution per D-bin\n(dashed = median)',
             fontsize=FS_TITLE)
ax.legend(fontsize=FS_TICK - 1)
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_LW)
    spine.set_color('black')
ax.tick_params(labelsize=FS_TICK, colors='black')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(OUT, dpi=150, facecolor='white')
print(f'Saved → {OUT}')

# ── text summary ──────────────────────────────────────────────────────────────
print('\n── Summary statistics by D-bin ──────────────────────────────')
print(f'{"D range":>14}  {"N":>6}  {"t90 med":>8}  {"t95 med":>8}  '
      f'{"t99 med":>8}  {"N_ph med":>10}  {"N_ph p5":>10}')
for k, (label,) in enumerate(zip(D_LABELS,)):
    mask = (d_arr >= D_EDGES[k]) & (d_arr < D_EDGES[k + 1])
    if mask.sum() == 0:
        continue
    print(f'{label:>14}  {mask.sum():>6}  '
          f'{np.median(t90[mask]):>8.1f}  '
          f'{np.median(t95[mask]):>8.1f}  '
          f'{np.median(t99[mask]):>8.1f}  '
          f'{np.median(n_ph[mask]):>10.0f}  '
          f'{np.percentile(n_ph[mask], 5):>10.0f}')
