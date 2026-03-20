"""Plot 10 individual noise trace examples with DT-corrected Poisson units.

Uses locally available i100_sample.npy (dt100, 4096 bins at 1ms) and d_sample.npy.
Noise physics: traces are counts/s; bg_counts ~ Poisson(n_abs * DT_BIN_S),
then bg = bg_counts / DT_BIN_S restores counts/s.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Constants
DT_BIN_S       = 1e-3   # dt100 bin width = 1 ms in seconds
NOISE_SIGMA    = 0.15
NOISE_MAX_FRAC = 0.50

# Plot aesthetics
CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12

# Load local samples
print("Loading i100_sample.npy and d_sample.npy ...")
i100 = np.load('i100_sample.npy').astype(np.float32)   # (200, 4096) counts/s
d    = np.load('d_sample.npy').astype(np.float32)       # (200,)
print("  i100: %s  range: %.0f .. %.0f counts/s" % (i100.shape, i100.min(), i100.max()))
print("  d:    %s  range: %.4f .. %.4f um2/s" % (d.shape, d.min(), d.max()))

# Pick 10 diverse examples across log-D range
rng  = np.random.default_rng(7)
mask = d <= 10
valid = np.where(mask)[0]
log_d = np.log10(d[valid])
bins  = np.linspace(log_d.min(), log_d.max(), 11)
chosen = []
for i in range(10):
    lo, hi = bins[i], bins[i+1]
    cands  = np.where((log_d >= lo) & (log_d < hi))[0]
    if len(cands) == 0:
        continue
    chosen.append(valid[rng.choice(cands)])
print("Selected D values: %s" % [round(float(d[i]), 4) for i in chosen])

# Draw noise and plot
rng2    = np.random.default_rng(42)
out_dir = 'noise_traces_individual'
os.makedirs(out_dir, exist_ok=True)

N_BINS = i100.shape[1]  # 4096
time_s = np.arange(N_BINS) * DT_BIN_S   # time axis in seconds (0 .. 4.096 s)

for k, gi in enumerate(chosen):
    trace_clean = i100[gi]              # (4096,) counts/s

    n_frac = float(min(abs(rng2.normal(0.0, NOISE_SIGMA)), NOISE_MAX_FRAC))
    n_abs  = n_frac * float(trace_clean.max())   # background level in counts/s

    # Correct: Poisson(n_abs * DT_BIN_S) gives integer counts per 1ms bin
    bg_counts = rng2.poisson(n_abs * DT_BIN_S, size=N_BINS).astype(np.float32)
    bg        = bg_counts / DT_BIN_S           # restore to counts/s

    trace_noisy = trace_clean + bg
    D_val       = float(d[gi])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.plot(time_s, trace_clean  / 1e3, color=CURVE_COLOR, alpha=0.40, lw=0.7, label='clean')
    ax.plot(time_s, trace_noisy  / 1e3, color=CURVE_COLOR, alpha=0.85, lw=0.7, label='noisy')
    ax.axhline(n_abs / 1e3, color='red', lw=1.3, ls='--',
               label='bg = %.2f kHz' % (n_abs / 1e3))

    ax.set_xlabel('Time (s)',  color='white', fontsize=FS_LABEL)
    ax.set_ylabel('I (kHz)',   color='white', fontsize=FS_LABEL)
    ax.set_title(
        'D = %.3f um2/s   bg = %.2f kHz   n_frac = %.1f%%' % (D_val, n_abs/1e3, n_frac*100),
        color='white', fontsize=FS_TITLE)

    ax.tick_params(colors='white', labelsize=FS_TICK)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(SPINE_LW)

    leg = ax.legend(fontsize=10, framealpha=0.3)
    for txt in leg.get_texts():
        txt.set_color('white')

    plt.tight_layout()
    fname = '%s/trace_%02d_D%.3f.png' % (out_dir, k+1, D_val)
    fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print("  [%d/10] %s  D=%.3f um2/s  bg=%.2f kHz  n_frac=%.1f%%" % (
          k+1, os.path.basename(fname), D_val, n_abs/1e3, n_frac*100))

print("Done — all 10 plots saved to %s/" % out_dir)
