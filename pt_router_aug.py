"""
Router evaluation: wavenet_wide_aug (general) vs wavenet_wide_specialist (fast diffusers).

Strategy:
  - Use wavenet_wide_aug as the router to estimate D for each test sample
  - If router predicts D < threshold → trust the general model (wavenet_wide_aug)
  - If router predicts D >= threshold → switch to the specialist

Sweeps threshold from 0.5 to 3.0 (step 0.05) in predicted-D space and reports
MAPE at each threshold. Saves the best threshold result + scatter plot.

Inputs (must exist on cluster):
  pred_logd_wavenet_wide_aug_test.npy        (9502,) — router + general predictions
  pred_logd_wavenet_wide_specialist_test.npy (9502,) — specialist predictions
  cache_d_test_90pct.npy                     (10000,) — ground-truth labels

Outputs:
  results_router_aug.txt
  router_aug.png
"""
import numpy as np, time
from sklearn.metrics import r2_score, mean_absolute_error

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

t0 = time.time()

# ── Load ground truth (apply d<=10 mask → 9502 samples) ──────────────────────
log("Loading test labels...")
d_test_all = np.load('cache_d_test_90pct.npy')
mask       = d_test_all <= 10
d_test     = d_test_all[mask]
log(f"Test samples: {len(d_test)} (filtered d<=10 from {len(d_test_all)} total)")

# ── Load predictions ──────────────────────────────────────────────────────────
log("Loading model predictions...")
log_general    = np.load('pred_logd_wavenet_wide_aug_test.npy')        # (9502,)
log_specialist = np.load('pred_logd_wavenet_wide_specialist_test.npy') # (9502,)

d_general    = np.exp(log_general)    # router = general model's predicted D
d_specialist = np.exp(log_specialist)

assert len(log_general) == len(d_test), \
    f"Shape mismatch: general={len(log_general)} vs test={len(d_test)}"
assert len(log_specialist) == len(d_test), \
    f"Shape mismatch: specialist={len(log_specialist)} vs test={len(d_test)}"

# ── Baseline MAPEs ────────────────────────────────────────────────────────────
mask_slow = d_test < 1.0
mask_fast = d_test >= 1.0

m_gen_all  = mape(d_test, d_general)
m_gen_slow = mape(d_test[mask_slow], d_general[mask_slow])
m_gen_fast = mape(d_test[mask_fast], d_general[mask_fast])

m_spc_all  = mape(d_test, d_specialist)
m_spc_slow = mape(d_test[mask_slow], d_specialist[mask_slow])
m_spc_fast = mape(d_test[mask_fast], d_specialist[mask_fast])

log(f"\nBaseline MAPEs:")
log(f"  wavenet_wide_aug:        MAPE={m_gen_all:.2f}%  d<1={m_gen_slow:.2f}%  d>=1={m_gen_fast:.2f}%")
log(f"  wavenet_wide_specialist: MAPE={m_spc_all:.2f}%  d<1={m_spc_slow:.2f}%  d>=1={m_spc_fast:.2f}%")

# ── Threshold sweep ───────────────────────────────────────────────────────────
# Threshold is in D-space; the router uses wavenet_wide_aug's predicted D.
# For each sample: if d_general < threshold → use general, else → use specialist.
thresholds = np.arange(0.5, 3.05, 0.05)

log(f"\nSweeping {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}:")
log(f"{'Threshold':>10}  {'N_specialist':>12}  {'MAPE_all':>9}  {'d<1':>7}  {'d>=1':>7}")

results = []
for thr in thresholds:
    use_spec = d_general >= thr
    d_pred   = np.where(use_spec, d_specialist, d_general)

    m_all  = mape(d_test, d_pred)
    m_slow = mape(d_test[mask_slow], d_pred[mask_slow])
    m_fast = mape(d_test[mask_fast], d_pred[mask_fast])
    n_spec = use_spec.sum()

    results.append((thr, m_all, m_slow, m_fast, n_spec, d_pred.copy()))
    log(f"  thr={thr:.2f}  N_spec={n_spec:5d}  MAPE={m_all:.2f}%  d<1={m_slow:.2f}%  d>=1={m_fast:.2f}%")

# ── Best threshold (minimise overall MAPE) ────────────────────────────────────
best_idx = int(np.argmin([r[1] for r in results]))
best_thr, best_mape_all, best_mape_slow, best_mape_fast, best_n_spec, best_d_pred = results[best_idx]

log(f"\n{'='*60}")
log(f"Best threshold: {best_thr:.2f}")
log(f"  N routed to specialist: {best_n_spec} / {len(d_test)}")
log(f"  MAPE={best_mape_all:.2f}%  d<1={best_mape_slow:.2f}%  d>=1={best_mape_fast:.2f}%")
log(f"  vs general alone:  MAPE={m_gen_all:.2f}%  d<1={m_gen_slow:.2f}%  d>=1={m_gen_fast:.2f}%")
log(f"{'='*60}")

# ── Save results ──────────────────────────────────────────────────────────────
runtime = time.time() - t0
r2_best = r2_score(d_test, best_d_pred)

header = (
    f"Router: wavenet_wide_aug → wavenet_wide_specialist\n"
    f"Strategy: if router_D < threshold → general, else → specialist\n"
    f"N_test: {len(d_test)}\n\n"
    f"Baseline — wavenet_wide_aug:\n"
    f"  MAPE={m_gen_all:.2f}%  d<1={m_gen_slow:.2f}%  d>=1={m_gen_fast:.2f}%\n"
    f"Baseline — wavenet_wide_specialist:\n"
    f"  MAPE={m_spc_all:.2f}%  d<1={m_spc_slow:.2f}%  d>=1={m_spc_fast:.2f}%\n\n"
    f"Best threshold: {best_thr:.2f}\n"
    f"  N_specialist: {best_n_spec}\n"
    f"  MAPE={best_mape_all:.2f}%  d<1={best_mape_slow:.2f}%  d>=1={best_mape_fast:.2f}%\n"
    f"  R²={r2_best:.4f}\n"
    f"Runtime: {runtime:.1f}s\n\n"
    f"Full threshold sweep:\n"
    f"{'thr':>6}  {'N_spec':>6}  {'MAPE_all':>9}  {'d<1':>7}  {'d>=1':>7}\n"
)
rows = "".join(
    f"{r[0]:6.2f}  {r[4]:6d}  {r[1]:9.2f}  {r[2]:7.2f}  {r[3]:7.2f}\n"
    for r in results
)
with open('results_router_aug.txt', 'w') as f:
    f.write(header + rows)
log("Saved results_router_aug.txt")

# ── Scatter plot at best threshold ────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

use_spec_best = d_general >= best_thr

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(d_test[~use_spec_best], best_d_pred[~use_spec_best],
           alpha=0.15, s=4, color='steelblue', label='general')
ax.scatter(d_test[use_spec_best],  best_d_pred[use_spec_best],
           alpha=0.25, s=4, color='tomato', label='specialist')
lims = [min(d_test.min(), best_d_pred.min()), max(d_test.max(), best_d_pred.max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True D (µm²/s)'); ax.set_ylabel('Predicted D (µm²/s)')
ax.set_title(f'Router (thr={best_thr:.2f})  MAPE={best_mape_all:.1f}%  '
             f'd<1={best_mape_slow:.1f}%  d≥1={best_mape_fast:.1f}%')
ax.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig('router_aug.png', dpi=150)
log("Saved router_aug.png")
print("Done.")
