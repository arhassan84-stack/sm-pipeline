"""
Plot 10 random simulations from the multi-dt dataset.
Each figure shows 3 subplots: dt=0.1ms, dt=0.5ms, dt=1.0ms traces
for the same underlying molecular trajectory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DT_CONFIGS = [
    ('dt010', 0.1),
    ('dt050', 0.5),
    ('dt100', 1.0),
]

TRACE_FILES = {tag: f'sims_multidt_noise0_i_{tag}.npy' for tag, _ in DT_CONFIGS}
D_FILE      = 'sims_multidt_noise0_d.npy'
N_PLOTS     = 10
SEED        = 7

CURVE_COLOR  = '#00FF00'     # bright green
SPINE_LW     = 1.5           # 1.5× default spine linewidth

# Font sizes (1.5× originals)
FS_SUPTITLE  = 20
FS_TITLE     = 14
FS_LABEL     = 15
FS_YLABEL    = 14
FS_TICK      = 12

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading files...")
traces  = {tag: np.load(TRACE_FILES[tag], mmap_mode='r') for tag, _ in DT_CONFIGS}
d_vals  = np.load(D_FILE)
N_total = len(d_vals)
print(f"  Total traces: {N_total:,}  D range: [{d_vals.min():.4f}, {d_vals.max():.4f}]")

rng     = np.random.default_rng(SEED)
indices = rng.choice(N_total, size=N_PLOTS, replace=False)

os.makedirs('plots_multidt', exist_ok=True)

for fig_num, idx in enumerate(indices):
    D   = d_vals[idx]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(f'Simulation #{idx}  |  D = {D:.4f} µm²/s',
                 fontsize=FS_SUPTITLE, fontweight='bold')

    for ax, (tag, dt_ms) in zip(axes, DT_CONFIGS):
        trace_raw = np.array(traces[tag][idx], dtype=np.float32)
        trace_khz = trace_raw / 1e3                           # counts/s → kHz
        N_bins    = len(trace_khz)
        t_s       = np.arange(N_bins) * dt_ms * 1e-3          # ms → seconds

        ax.plot(t_s, trace_khz, lw=0.5, color=CURVE_COLOR, alpha=0.95)
        ax.set_ylabel('Intensity (kHz)', fontsize=FS_YLABEL)
        ax.set_title(
            f'dt = {dt_ms} ms  |  {N_bins:,} bins  |  '
            f'mean = {trace_khz.mean():.1f} kHz  std = {trace_khz.std():.1f} kHz',
            fontsize=FS_TITLE, loc='left')
        ax.set_xlim(0, t_s[-1])
        ax.tick_params(labelsize=FS_TICK)

        # Thicken figure box (all four spines)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)

    axes[-1].set_xlabel('Time (s)', fontsize=FS_LABEL)
    plt.tight_layout()

    fname = f'plots_multidt/multidt_trace_{fig_num+1:02d}_idx{idx}_D{D:.4f}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")

print(f"Done. {N_PLOTS} figures saved to plots_multidt/")
