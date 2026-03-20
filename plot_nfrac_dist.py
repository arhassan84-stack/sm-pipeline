"""Plot the distribution of n_frac = |N(0, 0.15)| clipped at 0.50."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

NOISE_SIGMA    = 0.15
NOISE_MAX_FRAC = 0.50
N_SAMPLES      = 2_000_000

CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12

rng    = np.random.default_rng(0)
n_frac = np.abs(rng.normal(0.0, NOISE_SIGMA, size=N_SAMPLES))
n_frac = np.clip(n_frac, 0, NOISE_MAX_FRAC)

# Analytical half-normal PDF (before clip)
x_pdf = np.linspace(0, NOISE_MAX_FRAC, 500)
pdf   = stats.halfnorm.pdf(x_pdf, scale=NOISE_SIGMA)

fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.hist(n_frac, bins=120, density=True, color=CURVE_COLOR,
        alpha=0.55, label='Sampled')
ax.plot(x_pdf, pdf, color=CURVE_COLOR, lw=2.0, label='Half-normal PDF')
ax.axvline(NOISE_MAX_FRAC, color='red', lw=1.5, ls='--',
           label=f'Clip at {NOISE_MAX_FRAC}')

mean_val = n_frac.mean()
ax.axvline(mean_val, color='white', lw=1.2, ls=':',
           label=f'Mean = {mean_val:.3f}  ({mean_val*100:.1f}%)')

ax.set_xlabel('n_frac  (background / peak intensity)', color='white', fontsize=FS_LABEL)
ax.set_ylabel('Density', color='white', fontsize=FS_LABEL)
ax.set_title('Distribution of background fraction  n_frac ~ |N(0, 0.15)| clipped at 0.50',
             color='white', fontsize=FS_TITLE)
ax.tick_params(colors='white', labelsize=FS_TICK)
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(SPINE_LW)

leg = ax.legend(fontsize=11, framealpha=0.3)
for txt in leg.get_texts():
    txt.set_color('white')

plt.tight_layout()
fig.savefig('plots_morphfeat/nfrac_distribution.png', dpi=150,
            bbox_inches='tight', facecolor='black')
plt.close(fig)
print(f"Mean n_frac = {mean_val:.4f}  ({mean_val*100:.2f}%)")
print(f"Median      = {np.median(n_frac):.4f}")
print(f"P90         = {np.percentile(n_frac, 90):.4f}")
print(f"Fraction at clip (=0.50) = {(n_frac == NOISE_MAX_FRAC).mean()*100:.2f}%")
print("Saved plots_morphfeat/nfrac_distribution.png")
