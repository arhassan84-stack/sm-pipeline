"""
Ensemble: geometric mean (average in log-space) of 8 aug WaveNet models.
Loads saved pred_logd_wavenet_*_aug_test.npy prediction files — no GPU needed.
"""
import numpy as np, time
from sklearn.metrics import r2_score, mean_absolute_error

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p):
    m = t > 0
    return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100

t0 = time.time()

# ── Load ground truth ──────────────────────────────────────────────────────────
# The aug training scripts already filtered d<=10 before saving predictions,
# so pred_logd_*_aug_test.npy has 9502 rows (not 10000).
# Apply the same mask to the labels here.
log("Loading test labels...")
d_test_all = np.load('cache_d_test_90pct.npy')
mask       = d_test_all <= 10
d_test     = d_test_all[mask]
log(f"Test samples: {len(d_test)} (filtered d<=10 from {len(d_test_all)} total)")

# ── Model list (all 8 aug WaveNets) ───────────────────────────────────────────
models = [
    'wavenet_wide_aug',
    'wavenet_s4_aug',
    'wavenet_stride2_aug',
    'wavenet_s5_aug',
    'wavenet_alt_aug',
    'wavenet_s3_aug',
    'wavenet_s2_aug',
]

# ── Load individual predictions ───────────────────────────────────────────────
log_preds = {}
log("Individual model MAPEs:")
for name in models:
    lp = np.load(f'pred_logd_{name}_test.npy')   # already filtered to d<=10 (9502 rows)
    log_preds[name] = lp
    m_all  = mape(d_test, np.exp(lp))
    m_slow = mape(d_test[d_test < 1],  np.exp(lp[d_test < 1]))
    m_fast = mape(d_test[d_test >= 1], np.exp(lp[d_test >= 1]))
    log(f"  {name:30s}  MAPE={m_all:.2f}%  d<1={m_slow:.2f}%  d>=1={m_fast:.2f}%")

# ── Full 8-model ensemble ─────────────────────────────────────────────────────
stack = np.stack(list(log_preds.values()), axis=0)   # (8, N)
log_pred_ens = stack.mean(axis=0)
d_pred_ens   = np.exp(log_pred_ens)

mask_slow = d_test < 1
mask_fast = d_test >= 1

r2_te   = r2_score(d_test, d_pred_ens)
mae_te  = mean_absolute_error(d_test, d_pred_ens)
mape_te = mape(d_test, d_pred_ens)
mape_sl = mape(d_test[mask_slow], d_pred_ens[mask_slow])
mape_fs = mape(d_test[mask_fast], d_pred_ens[mask_fast])

log(f"\n{'='*60}")
log(f"8-model aug ensemble:  MAPE={mape_te:.2f}%  d<1={mape_sl:.2f}%  d>=1={mape_fs:.2f}%")
log(f"                       R²={r2_te:.4f}  MAE={mae_te:.4f}")
log(f"{'='*60}\n")

# ── Sub-ensembles (top-3 and top-5) ───────────────────────────────────────────
top3 = ['wavenet_wide_aug', 'wavenet_s4_aug', 'wavenet_stride2_aug']
top5 = ['wavenet_wide_aug', 'wavenet_s4_aug', 'wavenet_stride2_aug',
        'wavenet_s5_aug',  'wavenet_alt_aug']

for label, subset in [('top-3', top3), ('top-5', top5)]:
    lp_sub  = np.stack([log_preds[n] for n in subset], axis=0).mean(axis=0)
    dp_sub  = np.exp(lp_sub)
    mt_sub  = mape(d_test, dp_sub)
    ms_sub  = mape(d_test[mask_slow], dp_sub[mask_slow])
    mf_sub  = mape(d_test[mask_fast], dp_sub[mask_fast])
    r2_sub  = r2_score(d_test, dp_sub)
    log(f"{label} ensemble ({', '.join(subset[:2])}...):  "
        f"MAPE={mt_sub:.2f}%  d<1={ms_sub:.2f}%  d>=1={mf_sub:.2f}%  R²={r2_sub:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
runtime = time.time() - t0
summary = (
    f"Task: pt_ensemble_aug\n"
    f"Components: {', '.join(models)}\n"
    f"N_test: {len(d_test)}\n\n"
    f"Individual MAPEs (test):\n"
    + "".join(f"  {n:30s}  {mape(d_test, np.exp(log_preds[n])):.2f}%\n" for n in models)
    + f"\n8-model ensemble:\n"
    f"  MAPE  = {mape_te:.2f}%\n"
    f"  d<1   = {mape_sl:.2f}%\n"
    f"  d>=1  = {mape_fs:.2f}%\n"
    f"  R²    = {r2_te:.4f}\n"
    f"  MAE   = {mae_te:.4f}\n"
    f"  Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open('results_ensemble_aug.txt', 'w') as f:
    f.write(summary)
log("Saved results_ensemble_aug.txt")

# ── Scatter plot ──────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(d_test, d_pred_ens, alpha=0.15, s=4, color='steelblue')
lims = [min(d_test.min(), d_pred_ens.min()), max(d_test.max(), d_pred_ens.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True D (µm²/s)'); ax.set_ylabel('Predicted D (µm²/s)')
ax.set_title(f'8-model aug ensemble  R²={r2_te:.4f}  MAPE={mape_te:.1f}%')
plt.tight_layout()
plt.savefig('ensemble_aug.png', dpi=150)
log("Saved ensemble_aug.png")
print("Done.")
