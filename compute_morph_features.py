"""
Compute 13 morphological features (groups 1-5) for multidt training/test caches.

Features per dt channel (13 total):
  Group 1 — Weighted moments (signal treated as prob. dist. over time):
    [0] centroid_norm   — centre of mass / T  (0→1)
    [1] rms_width_norm  — RMS spread / T      (0→1)
    [2] skewness        — 3rd standardised moment
    [3] kurtosis        — 4th standardised moment

  Group 2 — Width fractions (fraction of bins above threshold of Imax):
    [4] fwhm_frac       — bins ≥ Imax/2          / T
    [5] fw1e_frac       — bins ≥ Imax/e          / T
    [6] fw1e2_frac      — bins ≥ Imax/e²         / T

  Group 3 — Energy containment (fraction of bins to capture X% of total signal):
    [7]  T50_frac
    [8]  T75_frac
    [9]  T90_frac
    [10] T95_frac

  Group 4 — Inverse Participation Ratio:
    [11] ipr            — T·Σ(I²)/(ΣI)²   (1=uniform, T=all in one bin)

  Group 5 — Signal entropy:
    [12] entropy        — −Σ p·log(p)   (0=one bin, log(T)=uniform)

Saves:
  cache_morphfeat_{train,test}_multidt_dt{010,050,100}.npy  shape (N, 13)

Plots:
  plots_morphfeat/feat_{i:02d}_{name}.png  — scatter vs D, 3 dt panels
"""
import numpy as np, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE       = '/gpfs/gibbs/pi/holley/hassan/new_'
N_SCATTER  = 30_000   # subsample for scatter plots

FEAT_NAMES = [
    'Centroid (frac)', 'RMS width (frac)', 'Skewness', 'Kurtosis',
    'FWHM (frac)', 'FW 1/e (frac)', 'FW 1/e² (frac)',
    'T50 (frac)', 'T75 (frac)', 'T90 (frac)', 'T95 (frac)',
    'IPR', 'Entropy',
]
N_FEATS = len(FEAT_NAMES)  # 13

DT_CONFIGS = [
    ('dt010', 40960, 1000),   # (tag, T_bins, chunk_size)
    ('dt050',  8192, 4000),
    ('dt100',  4096, 8000),
]

# ── Feature computation ────────────────────────────────────────────────────────

def compute_features(I_f32):
    """I_f32: (N, T) float32  →  (N, 13) float32.
    Accumulations done in float64 to avoid precision loss for large T."""
    N, T = I_f32.shape
    eps  = 1e-12
    I    = I_f32.astype(np.float64)

    Isum = I.sum(axis=1, keepdims=True).clip(eps)       # (N, 1)
    p    = I / Isum                                       # probability dist, (N, T)

    # --- Group 1: weighted moments ---
    t      = np.arange(T, dtype=np.float64)
    t0     = (p * t).sum(axis=1)                         # centroid, bins
    dt_    = t[None, :] - t0[:, None]                    # (N, T)
    m2     = (p * dt_**2).sum(axis=1).clip(0)
    sigma  = np.sqrt(m2)
    skew   = (p * dt_**3).sum(axis=1) / (sigma**3 + eps)
    kurt   = (p * dt_**4).sum(axis=1) / (sigma**4 + eps)
    c_norm = t0 / T
    s_norm = sigma / T

    # --- Group 2: width fractions ---
    Imax   = I.max(axis=1, keepdims=True).clip(eps)
    fwhm   = (I >= 0.5    * Imax).sum(axis=1) / T
    fw1e   = (I >= Imax / np.e   ).sum(axis=1) / T
    fw1e2  = (I >= Imax / np.e**2).sum(axis=1) / T

    # --- Group 3: energy containment ---
    I_sort  = np.sort(I, axis=1)[:, ::-1].copy()        # descending
    cumfrac = I_sort.cumsum(axis=1) / Isum               # (N, T)
    T50  = (cumfrac < 0.50).sum(axis=1) / T
    T75  = (cumfrac < 0.75).sum(axis=1) / T
    T90  = (cumfrac < 0.90).sum(axis=1) / T
    T95  = (cumfrac < 0.95).sum(axis=1) / T

    # --- Group 4: IPR ---
    ipr = T * (I**2).sum(axis=1) / (Isum.squeeze()**2 + eps)

    # --- Group 5: entropy ---
    p_c     = np.clip(p, eps, None)
    entropy = -(p_c * np.log(p_c)).sum(axis=1)

    return np.column_stack([
        c_norm, s_norm, skew, kurt,
        fwhm, fw1e, fw1e2,
        T50, T75, T90, T95,
        ipr, entropy,
    ]).astype(np.float32)


# ── Compute and save for train + test ─────────────────────────────────────────

d_train = np.load(f'{BASE}/cache_d_train_multidt.npy')
d_test  = np.load(f'{BASE}/cache_d_test_multidt.npy')

feats_train = {}   # tag → (N_train, 13)
feats_test  = {}   # tag → (N_test,  13)

for split, d_all, store in [('train', d_train, feats_train),
                              ('test',  d_test,  feats_test)]:
    N = len(d_all)
    print(f'\n=== {split}  N={N} ===')
    for tag, T_bins, chunk in DT_CONFIGS:
        fname_in  = f'{BASE}/cache_i_{split}_multidt_{tag}.npy'
        fname_out = f'{BASE}/cache_morphfeat_{split}_multidt_{tag}.npy'
        I_mmap    = np.load(fname_in, mmap_mode='r')
        feats     = np.empty((N, N_FEATS), dtype=np.float32)
        t_start   = time.time()
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            feats[s:e] = compute_features(np.array(I_mmap[s:e]))
            if (s // chunk) % 10 == 0:
                print(f'  {tag}  {e:6d}/{N}  ({time.time()-t_start:.0f}s)')
        np.save(fname_out, feats)
        store[tag] = feats
        print(f'  saved {os.path.basename(fname_out)}  '
              f'shape={feats.shape}  ({time.time()-t_start:.0f}s total)')


# ── Scatter plots: each feature vs D, 3 dt panels ─────────────────────────────

out_dir = f'{BASE}/plots_morphfeat'
os.makedirs(out_dir, exist_ok=True)

# Combine train + test for richer scatter
d_all   = np.concatenate([d_train, d_test])
mask    = d_all <= 10
d_plot  = d_all[mask]

feat_combined = {}
for tag, _, _ in DT_CONFIGS:
    f_tr = feats_train[tag]
    f_te = feats_test[tag]
    combined = np.concatenate([f_tr, f_te], axis=0)[mask]
    feat_combined[tag] = combined

# Subsample
rng   = np.random.default_rng(0)
N_tot = len(d_plot)
idx   = rng.choice(N_tot, size=min(N_SCATTER, N_tot), replace=False)
d_sub = d_plot[idx]
logd  = np.log10(d_sub)

CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_TITLE    = 13
FS_LABEL    = 12
FS_TICK     = 10

DT_LABELS = ['dt = 0.1 ms', 'dt = 0.5 ms', 'dt = 1.0 ms']
DT_TAGS   = ['dt010', 'dt050', 'dt100']

for fi, fname in enumerate(FEAT_NAMES):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('black')
    fig.suptitle(f'Feature: {fname}', color='white', fontsize=14, y=1.01)

    for ax, tag, dt_label in zip(axes, DT_TAGS, DT_LABELS):
        y = feat_combined[tag][idx, fi]
        ax.set_facecolor('black')
        ax.scatter(logd, y, s=2, alpha=0.15, color=CURVE_COLOR, rasterized=True)
        ax.set_title(dt_label, color='white', fontsize=FS_TITLE)
        ax.set_xlabel('log₁₀(D  [µm²/s])', color='white', fontsize=FS_LABEL)
        ax.set_ylabel(fname, color='white', fontsize=FS_LABEL)
        ax.tick_params(colors='white', labelsize=FS_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(SPINE_LW)

    plt.tight_layout()
    safe = fname.replace('/', '_').replace(' ', '_').replace('²', '2').replace('₁', '1')
    fpath = f'{out_dir}/feat_{fi:02d}_{safe}.png'
    fig.savefig(fpath, dpi=130, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f'  plot saved: {os.path.basename(fpath)}')

print(f'\nAll done. Plots in {out_dir}/')
