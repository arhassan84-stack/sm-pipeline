"""
Step 2: Hyperparameter tuning of HistGradientBoostingRegressor
using the full feature set (ACF + scattering + PSD).
"""
import numpy as np
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape(true, pred):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

t_total = time.time()

# ── Load cached features ──────────────────────────────────────────────────────
log("Loading cached features (with PSD)...")
X_train   = np.load('cache_X_train_psd.npy')
d_train   = np.load('cache_d_train.npy')
X_test    = np.load('cache_X_test_psd.npy')
d_test    = np.load('cache_d_test.npy')
log(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Filter d <= 10 ────────────────────────────────────────────────────────────
train_mask = d_train <= 10
X_train, d_train = X_train[train_mask], d_train[train_mask]
test_mask  = d_test <= 10
X_test_eval, d_test_eval = X_test[test_mask], d_test[test_mask]
log(f"After d<=10 filter — Train: {len(d_train)}  Test: {len(d_test_eval)}")

log_d_train = np.log(d_train)

# ── Hyperparameter search ─────────────────────────────────────────────────────
param_dist = {
    'max_iter':        [300, 500, 800],
    'max_depth':       [4, 5, 6, 7, 8],
    'learning_rate':   [0.01, 0.03, 0.05, 0.08, 0.1],
    'min_samples_leaf':[10, 20, 40, 80],
    'max_leaf_nodes':  [31, 63, 127, None],
    'l2_regularization':[0.0, 0.1, 1.0],
}

base_model = HistGradientBoostingRegressor(random_state=42)

log("Starting RandomizedSearchCV (50 iterations, 3-fold CV)...")
search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    scoring='r2',
    random_state=42,
    verbose=2,
    n_jobs=-1,
)
search.fit(X_train, log_d_train)

log(f"Best params: {search.best_params_}")
log(f"Best CV R² (log space): {search.best_score_:.4f}")

# ── Evaluate best model ───────────────────────────────────────────────────────
best = search.best_estimator_
d_pred_train = np.exp(best.predict(X_train))
d_pred_test  = np.exp(best.predict(X_test_eval))

print(f"\n{'='*55}")
print(f"Train   R²={r2_score(d_train, d_pred_train):.4f}  MAE={mean_absolute_error(d_train, d_pred_train):.4f}  MAPE={mape(d_train, d_pred_train):.1f}%")
print(f"Test    R²={r2_score(d_test_eval, d_pred_test):.4f}  MAE={mean_absolute_error(d_test_eval, d_pred_test):.4f}  MAPE={mape(d_test_eval, d_pred_test):.1f}%")
print(f"{'='*55}")
print(f"\nBest hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
print("Done.")
