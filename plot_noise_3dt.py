"""Plot 10 noise-augmented FCS traces from training cache, 3 dt panels per figure.

Loads real multidt cache files (cache_i_train_multidt_dt{010,050,100}.npy) via mmap.
Picks 10 training samples spanning the log-D range, adds correlated Poisson
background noise, and saves one figure per sample to plots_noise_3dt/.

Noise physics (DT_MIN_S-corrected):
  Traces are counts/s.
  n_frac  ~ |N(0, 0.15)| clipped at 0.50
  n_abs   = n_frac * max(i010)               [counts/s]
  bg_counts ~ Poisson(n_abs * DT010_S)       [integer counts per 0.1ms bin]
  bg010   = bg_counts / DT010_S              [counts/s at dt010]
  bg050   = bg010.reshape(8192,  5).mean(1)  [counts/s at dt050, same photon stream]
  bg100   = bg010.reshape(4096, 10).mean(1)  [counts/s at dt100]
"""
import numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────
DT010_S        = 1e-4
N010, N050, N100 = 40960, 8192, 4096
NOISE_SIGMA    = 0.15
NOISE_MAX_FRAC = 0.50
N_EXAMPLES     = 10

# ── Plot aesthetics ────────────────────────────────────────────────────────────
CURVE_COLOR = '#00FF00'
DIM_COLOR   = '#005500'
SPINE_LW    = 1.5
FS_SUPTITLE = 16
FS_LABEL    = 12
FS_TICK     = 10

BASE = '/gpfs/gibbs/pi/holley/hassan/new_'

# ── Load caches via mmap (only rows we select will be paged in) ───────────────
print("Opening cache files (mmap)...")
i010_all = np.load(f'{BASE}/cache_i_train_multidt_dt010.npy', mmap_mode='r')
i050_all = np.load(f'{BASE}/cache_i_train_multidt_dt050.npy', mmap_mode='r')
i100_all = np.load(f'{BASE}/cache_i_train_multidt_dt100.npy', mmap_mode='r')
d_all    = np.load(f'{BASE}/cache_d_train_multidt.npy')
print(f"  i010:{i010_all.shape}  i050:{i050_all.shape}  "
      f"i100:{i100_all.shape}  d:{d_all.shape}")

# ── Pick 10 diverse samples across log-D range ────────────────────────────────
mask   = d_all <= 10
valid  = np.where(mask)[0]
log_d  = np.log10(d_all[valid])
rng_p  = np.random.default_rng(7)
edges  = np.linspace(log_d.min(), log_d.max(), N_EXAMPLES + 1)
chosen = []
for i in range(N_EXAMPLES):
    lo, hi = edges[i], edges[i + 1]
    cands  = np.where((log_d >= lo) & (log_d < hi))[0]
    if len(cands):
        chosen.append(int(valid[rng_p.choice(cands)]))
print(f"Selected {len(chosen)} examples")
print("D values: " + str([round(float(d_all[i]), 4) for i in chosen]))

# ── Noise RNG + output dir ────────────────────────────────────────────────────
rng_n   = np.random.default_rng(42)
out_dir = f'{BASE}/plots_noise_3dt'
os.makedirs(out_dir, exist_ok=True)

t010_ax = np.arange(N010) * DT010_S * 1e3   # ms
t050_ax = np.arange(N050) * 5e-4    * 1e3
t100_ax = np.arange(N100) * 1e-3    * 1e3

# ── Generate figures ──────────────────────────────────────────────────────────
for k, gi in enumerate(chosen):
    D_val = float(d_all[gi])
    print(f"[{k+1}/{len(chosen)}] D={D_val:.4f} um2/s  idx={gi}")

    i010 = np.array(i010_all[gi], dtype=np.float32)   # copy out of mmap
    i050 = np.array(i050_all[gi], dtype=np.float32)
    i100 = np.array(i100_all[gi], dtype=np.float32)

    # Draw noise
    n_frac = float(min(abs(rng_n.normal(0.0, NOISE_SIGMA)), NOISE_MAX_FRAC))
    n_abs  = n_frac * float(i010.max())           # counts/s

    bg_counts = rng_n.poisson(n_abs * DT010_S, size=N010).astype(np.float32)
    bg010 = bg_counts / DT010_S
    bg050 = bg010.reshape(N050,  5).mean(axis=1)
    bg100 = bg010.reshape(N100, 10).mean(axis=1)

    i010_n = i010 + bg010
    i050_n = i050 + bg050
    i100_n = i100 + bg100

    # Figure: 3 stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    fig.patch.set_facecolor('black')
    fig.suptitle(
        f'D = {D_val:.3f} μm²/s   bg = {n_abs/1e3:.2f} kHz   n_frac = {n_frac*100:.1f}%',
        color='white', fontsize=FS_SUPTITLE, y=0.99)

    panels = [
        (axes[0], t010_ax, i010,   i010_n, f'dt = 0.1 ms  ({N010:,} bins)'),
        (axes[1], t050_ax, i050,   i050_n, f'dt = 0.5 ms  ({N050:,} bins)'),
        (axes[2], t100_ax, i100,   i100_n, f'dt = 1.0 ms  ({N100:,} bins)'),
    ]

    for ax, t, clean, noisy, subtitle in panels:
        ax.set_facecolor('black')
        ax.plot(t, clean / 1e3, color=DIM_COLOR,   lw=0.5, alpha=0.9,  label='clean')
        ax.plot(t, noisy / 1e3, color=CURVE_COLOR, lw=0.5, alpha=0.85, label='noisy')
        ax.axhline(n_abs / 1e3, color='red', lw=1.2, ls='--',
                   label=f'bg = {n_abs/1e3:.2f} kHz')
        ax.set_title(subtitle, color='white', fontsize=FS_LABEL, pad=3)
        ax.set_ylabel('I (kHz)', color='white', fontsize=FS_LABEL)
        ax.tick_params(colors='white', labelsize=FS_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(SPINE_LW)
        leg = ax.legend(fontsize=9, framealpha=0.3, loc='upper right')
        for txt in leg.get_texts():
            txt.set_color('white')

    axes[2].set_xlabel('Time (ms)', color='white', fontsize=FS_LABEL)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f'{out_dir}/trace_{k+1:02d}_D{D_val:.3f}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  saved {os.path.basename(fname)}")

print(f"\nDone — {len(chosen)} figures in {out_dir}/")
