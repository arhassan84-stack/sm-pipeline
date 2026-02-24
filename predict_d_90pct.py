"""
Train HistGradientBoostingRegressor on 90% of n==0 data (d<=10, log target).
Uses best hyperparameters found in predict_d_hparam.py.
"""
import numpy as np
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape(true, pred):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

t_total = time.time()

log("Loading 90% train cache...")
X_train = np.load('cache_X_train_90pct.npy')
d_train = np.load('cache_d_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy')
d_test  = np.load('cache_d_test_90pct.npy')
log(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Filter d <= 10 ────────────────────────────────────────────────────────────
train_mask = d_train <= 10
X_train, d_train = X_train[train_mask], d_train[train_mask]
test_mask  = d_test <= 10
X_test_eval, d_test_eval = X_test[test_mask], d_test[test_mask]
log(f"After d<=10 filter — Train: {len(d_train)}  Test: {len(d_test_eval)}")

log_d_train = np.log(d_train)

# ── Train with best hyperparameters ──────────────────────────────────────────
log("Training HistGradientBoostingRegressor (best params)...")
model = HistGradientBoostingRegressor(
    max_iter=800,
    max_depth=5,
    learning_rate=0.08,
    min_samples_leaf=80,
    max_leaf_nodes=127,
    l2_regularization=0.0,
    random_state=42,
    verbose=1,
)
model.fit(X_train, log_d_train)
log("Training done — evaluating...")

d_pred_train = np.exp(model.predict(X_train))
d_pred_test  = np.exp(model.predict(X_test_eval))

print(f"\n{'='*55}")
print(f"Train  R²={r2_score(d_train, d_pred_train):.4f}  MAE={mean_absolute_error(d_train, d_pred_train):.4f}  MAPE={mape(d_train, d_pred_train):.1f}%")
print(f"Test   R²={r2_score(d_test_eval, d_pred_test):.4f}  MAE={mean_absolute_error(d_test_eval, d_pred_test):.4f}  MAPE={mape(d_test_eval, d_pred_test):.1f}%")
print(f"{'='*55}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(d_test_eval, d_pred_test, alpha=0.2, s=5, color='steelblue')
lims = [min(d_test_eval.min(), d_pred_test.min()), max(d_test_eval.max(), d_pred_test.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'90% Train — Test (d≤10)  R²={r2_score(d_test_eval, d_pred_test):.4f}')

ax = axes[1]
ax.scatter(d_test_eval, d_pred_test - d_test_eval, alpha=0.2, s=5, color='steelblue')
ax.axhline(0, color='r', lw=1, ls='--')
ax.set_xscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Residual (pred - true)')
ax.set_title('Residuals vs True d')

plt.tight_layout()
plt.savefig('results_90pct.png', dpi=150)
log(f"Saved results_90pct.png — Total runtime: {time.time()-t_total:.1f}s")
print("Done.")
