# -*- coding: utf-8 -*-
"""
3 separate figures from the original model_progress layout.
White background, everything else unchanged.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# -- Aesthetics ----------------------------------------------------------------
CURVE_COLOR  = '#00BB00'   # green (slightly darker than #00FF00 for readability on white)
SPINE_LW     = 1.5
FS_SUPTITLE  = 20
FS_TITLE     = 14
FS_LABEL     = 15
FS_TICK      = 12
FS_ANNOT     = 10

def dt(s): return datetime.strptime(s, '%Y-%m-%d %H:%M')

# -- dt=1ms models -------------------------------------------------------------
DT1 = [
    (dt('2026-02-25 16:09'), 'mlp_bn',              0, 15.2, None,  None,  0.9109),
    (dt('2026-02-25 16:12'), 'resnet_mlp',           0, 15.6, None,  None,  0.9088),
    (dt('2026-02-25 16:14'), 'cnn_fusion',           1, 14.9, None,  None,  0.9054),
    (dt('2026-02-25 17:31'), 'transformer',          2, 15.9, None,  None,  0.9039),
    (dt('2026-02-25 17:39'), 'weighted_loss',        0, 15.4, None,  None,  0.8887),
    (dt('2026-02-25 17:41'), 'mlp_extfeat',          0, 15.4, None,  None,  0.9138),
    (dt('2026-02-25 17:52'), 'cnn_multiscale',       1, 14.7, None,  None,  0.9174),
    (dt('2026-02-26 11:37'), 'specialist_slow',      0, 23.1, 11.9,  46.2,  0.2162),
    (dt('2026-02-26 12:05'), 'balanced_ensemble',    1, 14.4, 11.7,  20.0,  0.9167),
    (dt('2026-02-26 16:17'), 'ensemble_4model',      1, 14.1, 11.3,  19.7,  0.9155),
    (dt('2026-02-26 17:15'), 'mlp_weighted',         0, 15.1, 12.7,  20.0,  0.9098),
    (dt('2026-02-26 17:17'), 'resnet_weighted',      0, 15.0, 12.5,  20.3,  0.9097),
    (dt('2026-02-26 17:41'), 'cnn_peakcrop',         1, 15.7, 13.0,  21.3,  0.9012),
    (dt('2026-02-26 17:42'), 'ensemble_peakcrop',    1, 14.2, 11.4,  20.0,  0.9151),
    (dt('2026-02-26 17:55'), 'mlp_cosine',           0, 14.6, None,  None,  0.9131),
    (dt('2026-02-26 17:57'), 'resnet_cosine',        0, 14.7, None,  None,  0.9122),
    (dt('2026-02-26 18:00'), 'cnn_90pct',            1, 14.2, None,  None,  0.9068),
    (dt('2026-02-26 18:01'), 'ensemble',             1, 13.9, None,  None,  0.9143),
    (dt('2026-02-26 18:50'), 'fttransformer',        2, 14.7, 12.1,  20.1,  0.9196),
    (dt('2026-02-26 18:51'), 'ensemble_ftt',         1, 13.8, 11.1,  19.3,  0.9177),
    (dt('2026-02-27 14:49'), 'ftt_large',            2, 15.0, 12.3,  20.5,  0.9164),
    (dt('2026-02-27 15:32'), 'ftt_v2',               2, 14.8, 12.2,  20.2,  0.9117),
    (dt('2026-02-27 16:00'), 'cnn_s123',             1, 14.1, 11.0,  20.4,  0.9104),
    (dt('2026-02-27 16:04'), 'cnn_multiscale_90k',   1, 14.3, 11.3,  20.3,  0.9051),
    (dt('2026-03-02 14:06'), 'wavenet',              3, 13.4, 10.8,  18.8,  0.9190),
    (dt('2026-03-02 14:09'), 'cnn_wloss',            1, 14.8, 12.0,  20.7,  0.9076),
    (dt('2026-03-02 15:00'), 'ftt_wloss',            2, 14.6, 12.0,  19.9,  0.9198),
    (dt('2026-03-02 16:41'), 'wavenet_ftt',          3, 14.5, 12.2,  19.2,  0.9034),
    (dt('2026-03-02 17:37'), 'wavenet_s2',           3, 13.3, 10.6,  18.8,  0.9212),
    (dt('2026-03-02 17:56'), 'wavenet_s3',           3, 13.5, 10.8,  19.2,  0.9222),
    (dt('2026-03-02 18:49'), 'wavenet_ftt_v2',       3, 14.5, 12.0,  19.7,  0.9229),
    (dt('2026-03-02 19:14'), 'wavenet_alt',          3, 13.5, 10.9,  18.8,  0.9201),
    (dt('2026-03-03 18:06'), 'wavenet_stride2',      3, 12.8, 10.1,  18.2,  0.9320),
    (dt('2026-03-03 19:01'), 'wavenet_wide',         4, 13.6, 10.7,  19.5,  0.9213),
    (dt('2026-03-03 19:24'), 'wavenet_s4',           3, 13.2, 10.6,  18.7,  0.9175),
    (dt('2026-03-03 19:45'), 'wavenet_s5',           3, 13.6, 10.9,  19.2,  0.9231),
    (dt('2026-03-04 18:15'), 'wavenet_s3_aug',       3, 13.6, 11.6,  17.7,  0.9200),
    (dt('2026-03-04 18:23'), 'wavenet_s4_aug',       3, 12.7, 10.4,  17.3,  0.9224),
    (dt('2026-03-04 18:27'), 'wavenet_s5_aug',       3, 13.5, 11.5,  17.7,  0.9309),
    (dt('2026-03-04 18:58'), 'wavenet_s2_aug',       3, 13.6, 11.4,  18.1,  0.9268),
    (dt('2026-03-04 19:04'), 'wavenet_aug',          3, 13.3, 11.0,  18.0,  0.9254),
    (dt('2026-03-04 19:40'), 'wavenet_alt_aug',      3, 13.5, 11.2,  18.2,  0.9267),
    (dt('2026-03-04 19:41'), 'wavenet_stride2_aug',  3, 13.4, 11.7,  16.8,  0.9266),
    (dt('2026-03-04 20:20'), 'wavenet_wide_aug',     5, 12.3,  9.8,  17.6,  0.9206),
    (dt('2026-03-05 15:32'), 'wn_wide_specialist',   5,288.7,420.3,  17.5,  0.9238),
    (dt('2026-03-05 17:08'), 'wavenet_wide_aug_v2',  5, 13.1, 10.8,  17.7,  0.9235),
    (dt('2026-03-09 14:10'), 'wn_wide_fused3',       5, 14.0, 11.1,  20.0,  0.9266),
    (dt('2026-03-09 14:18'), 'wn_wide_fused5',       5, 12.6, 10.1,  17.8,  0.9200),
    (dt('2026-03-09 14:19'), 'wn_wide_fused10',      5, 13.5, 11.2,  18.2,  0.9267),
    (dt('2026-03-09 15:23'), 'wn_wide_fused20',      5, 13.8, 11.9,  17.7,  0.9300),
    (dt('2026-03-09 15:52'), 'wn_wide_smooth5',      5, 13.5, 11.1,  18.6,  0.9233),
    (dt('2026-03-09 15:58'), 'wn_wide_fused50',      5, 13.0, 10.6,  17.9,  0.9262),
    (dt('2026-03-09 16:08'), 'wn_wide_smooth20',     5, 14.3, 12.3,  18.6,  0.9220),
    (dt('2026-03-09 16:18'), 'wn_wide_smooth3',      5, 13.0, 10.8,  17.7,  0.9244),
    (dt('2026-03-09 16:44'), 'wn_wide_smooth50',     5, 14.6, 12.5,  18.8,  0.9221),
    (dt('2026-03-09 17:51'), 'wn_wide_smooth10',     5, 13.8, 11.6,  18.3,  0.9223),
    (dt('2026-03-11 15:49'), 'wn_wide_npe',          5, 12.5, 10.0,  17.5,  0.9293),
]

# -- dt-sweep models -----------------------------------------------------------
DT_SWEEP = [
    (dt('2026-03-10 17:57'), 0.5,  12.3, 10.6, 15.9, 0.9461),
    (dt('2026-03-12 15:43'), 0.8,  12.7, 10.7, 16.8, 0.9350),
    (dt('2026-03-12 19:56'), 0.25, 13.2, 11.4, 16.9, 0.9404),
    (dt('2026-03-12 21:24'), 0.4,  12.2, 10.6, 15.5, 0.9395),
    (dt('2026-03-13 11:03'), 0.2,  13.0, 11.2, 16.7, 0.9461),
]

# -- Family colours ------------------------------------------------------------
FAM_COL = {
    0: '#888888',
    1: '#aaaaff',
    2: '#ff9900',
    3: '#00cccc',
    4: '#ffdd00',
    5: '#00cc44',
}
FAM_LAB = {
    0: 'MLP / ResNet',
    1: 'CNN / Ensemble',
    2: 'FT-Transformer',
    3: 'WaveNet ch=128',
    4: 'WaveNet wide (no aug)',
    5: 'WaveNet wide + aug',
}

dates    = [r[0] for r in DT1]
families = [r[2] for r in DT1]
mapes    = [r[3] for r in DT1]
mapes_s  = [r[4] for r in DT1]
mapes_f  = [r[5] for r in DT1]

# Running best (exclude outlier >50 from specialist)
best_so_far, best_dates = [], []
running_min = np.inf
for d, m in sorted(zip(dates, mapes), key=lambda x: x[0]):
    if m is None or m > 50:
        continue
    if m < running_min:
        running_min = m
        best_dates.append(d)
        best_so_far.append(m)
best_dates.append(dt('2026-03-13 23:59'))
best_so_far.append(best_so_far[-1])

def running_best_series(dates_in, vals_in):
    pairs = [(d, v) for d, v in sorted(zip(dates_in, vals_in), key=lambda x: x[0])
             if v is not None and v <= 50]
    rb_d, rb_v, cur = [], [], np.inf
    for d, v in pairs:
        if v < cur:
            cur = v
            rb_d.append(d); rb_v.append(v)
    rb_d.append(dt('2026-03-13 23:59')); rb_v.append(rb_v[-1])
    return rb_d, rb_v

rb_s_d, rb_s_v = running_best_series(dates, mapes_s)
rb_f_d, rb_f_v = running_best_series(dates, mapes_f)


def style_ax(ax):
    ax.set_facecolor('white')
    ax.grid(True, color='#dddddd', lw=0.5)
    ax.tick_params(labelsize=FS_TICK, colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa')
        sp.set_linewidth(SPINE_LW)


# =============================================================================
# Figure 1 -- Overall MAPE over time (dt=1ms)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(15, 6))
fig1.patch.set_facecolor('white')

plotted_fams = set()
for d, fam, m in zip(dates, families, mapes):
    if m is None or m > 50:
        continue
    label = FAM_LAB[fam] if fam not in plotted_fams else ''
    ax1.scatter(d, m, color=FAM_COL[fam], s=45, zorder=3,
                label=label, alpha=0.85, edgecolors='none')
    plotted_fams.add(fam)

ax1.step(best_dates, best_so_far, where='post',
         color=CURVE_COLOR, lw=2.5, zorder=5, label='Best so far')

milestones = [
    (dt('2026-02-26 18:01'), 13.9, 'Ensemble\n13.9%',    (-3,  8)),
    (dt('2026-02-26 18:51'), 13.8, 'Ensemble+FTT\n13.8%', (4,   8)),
    (dt('2026-03-02 14:06'), 13.4, 'WaveNet\n13.4%',      (-3,  8)),
    (dt('2026-03-03 18:06'), 12.8, 'WN stride2\n12.8%',   (4,   8)),
    (dt('2026-03-04 18:23'), 12.7, 'WN s4 aug\n12.7%',    (-48, 8)),
    (dt('2026-03-04 20:20'), 12.3, 'WN wide aug\n12.3%',  (4,   8)),
]
for mx, my, lbl, off in milestones:
    ax1.annotate(lbl, xy=(mx, my), xytext=(off[0], off[1]),
                 textcoords='offset points', fontsize=9,
                 color='#222222', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))

ax1.set_ylabel('Test MAPE (%)', fontsize=FS_LABEL)
ax1.set_title('Overall MAPE  |  dt = 1 ms  |  test set ~9 500 samples',
              fontsize=FS_TITLE, loc='left')
ax1.set_ylim(11.5, 17.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.legend(fontsize=9, loc='upper right', frameon=True,
           facecolor='white', edgecolor='#cccccc')
style_ax(ax1)

fig1.tight_layout()
fig1.savefig('plots_progress/fig1_overall_mape.png', dpi=150, bbox_inches='tight',
             facecolor='white')
plt.close(fig1)
print("Saved plots_progress/fig1_overall_mape.png")


# =============================================================================
# Figure 2 -- MAPE by regime (d<1 and d>=1) over time
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('white')

slow_pairs = [(d, v) for d, v, m in zip(dates, mapes_s, mapes)
              if v is not None and v <= 50 and m is not None and m <= 50]
fast_pairs = [(d, v) for d, v, m in zip(dates, mapes_f, mapes)
              if v is not None and v <= 50 and m is not None and m <= 50]

plotted_slow = plotted_fast = False
for d, v in slow_pairs:
    ax2.scatter(d, v, color='#00aaff', s=35, alpha=0.7, zorder=3, edgecolors='none',
                label='d<1 (slow)' if not plotted_slow else '')
    plotted_slow = True
for d, v in fast_pairs:
    ax2.scatter(d, v, color='#ff5555', s=35, alpha=0.7, zorder=3, edgecolors='none',
                label='d>=1 (fast)' if not plotted_fast else '')
    plotted_fast = True

ax2.step(rb_s_d, rb_s_v, where='post', color='#0077cc', lw=2.5, zorder=5,
         label='Best d<1 (slow)')
ax2.step(rb_f_d, rb_f_v, where='post', color='#cc2222', lw=2.5, zorder=5,
         label='Best d>=1 (fast)')

ax2.set_ylabel('Test MAPE (%)', fontsize=FS_LABEL)
ax2.set_title('MAPE by diffusion regime  |  dt = 1 ms',
              fontsize=FS_TITLE, loc='left')
ax2.set_ylim(8.5, 25.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax2.legend(fontsize=9, loc='upper right', frameon=True,
           facecolor='white', edgecolor='#cccccc')
style_ax(ax2)

fig2.tight_layout()
fig2.savefig('plots_progress/fig2_regime_mape.png', dpi=150, bbox_inches='tight',
             facecolor='white')
plt.close(fig2)
print("Saved plots_progress/fig2_regime_mape.png")


# =============================================================================
# Figure 3 -- dt-sweep bar chart
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor('white')

dt_vals  = [0.2,  0.25, 0.4,  0.5,  0.8,  1.0]
dt_mapes = [13.0, 13.2, 12.2, 12.3, 12.7, 12.3]
dt_slow  = [11.2, 11.4, 10.6, 10.6, 10.7,  9.8]
dt_fast  = [16.7, 16.9, 15.5, 15.9, 16.8, 17.6]

x = np.arange(len(dt_vals))
w = 0.28
bars_all  = ax3.bar(x - w, dt_mapes, w, color='#555555', label='Overall',     zorder=3)
bars_slow = ax3.bar(x,     dt_slow,  w, color='#00aaff', label='d<1 (slow)',   zorder=3)
bars_fast = ax3.bar(x + w, dt_fast,  w, color='#ff5555', label='d>=1 (fast)',  zorder=3)

best_idx = int(np.argmin(dt_mapes))
bars_all[best_idx].set_edgecolor(CURVE_COLOR)
bars_all[best_idx].set_linewidth(2.5)

ax3.set_xticks(x)
ax3.set_xticklabels([f'{v} ms' for v in dt_vals], fontsize=FS_TICK)
ax3.set_ylabel('Test MAPE (%)', fontsize=FS_LABEL)
ax3.set_title('WaveNet wide + aug  |  dt sweep  |  test set ~28 k samples',
              fontsize=FS_TITLE, loc='left')
ax3.set_ylim(8.0, 19.5)
ax3.legend(fontsize=9, loc='upper right', frameon=True,
           facecolor='white', edgecolor='#cccccc')
style_ax(ax3)
ax3.grid(True, axis='y', color='#dddddd', lw=0.5)
ax3.grid(False, axis='x')

fig3.tight_layout()
fig3.savefig('plots_progress/fig3_dt_sweep.png', dpi=150, bbox_inches='tight',
             facecolor='white')
plt.close(fig3)
print("Saved plots_progress/fig3_dt_sweep.png")
