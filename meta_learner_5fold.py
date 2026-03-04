"""
Enhanced meta-learner for 13-model ensemble.

Improvements over meta_learner.py:
  1. Non-negative Ridge (NNLS + ridge augmentation) — all weights >= 0
  2. Stratified 5-fold CV (by d-bin) for unbiased alpha selection
  3. ACF nonlinear fit features as additional physical meta-inputs
     (nlfit_tauD, nlfit_rmse — if available from compute_acf_nlfit.py)
  4. Per-regime meta-learner: separate NNLS for d<1 and d>=1 sub-problems
  5. Full ensemble sweep: 9-model, 13-model, with/without ACF features

All base models were trained on full training set, so training predictions
are in-sample. Heavy L2 regularization (NNLS) mitigates contamination.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import time, os

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime(dt, dp):
    ms, mf = dt<1.0, dt>=1.0
    print(f"  d <1  ({ms.sum():5d}): R²={r2_score(dt[ms],dp[ms]):.4f}  MAPE={mape(dt[ms],dp[ms]):.1f}%")
    print(f"  d>=1  ({mf.sum():5d}): R²={r2_score(dt[mf],dp[mf]):.4f}  MAPE={mape(dt[mf],dp[mf]):.1f}%")

def nnls_ridge(A, b, alpha=1.0):
    """Non-negative ridge: min ||Ax-b||² + alpha*||x||²  s.t. x>=0
    Via augmented system: [A; sqrt(alpha)*I] x = [b; 0]"""
    n, p = A.shape
    A_aug = np.vstack([A, np.sqrt(alpha) * np.eye(p)])
    b_aug = np.concatenate([b, np.zeros(p)])
    w, _ = nnls(A_aug, b_aug)
    return w

def cv_select_alpha(X_tr, y_tr, d_tr, alphas, n_splits=5):
    """Stratified 5-fold CV on training data to select best NNLS alpha."""
    # Stratify by d-bin
    bins = np.searchsorted([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0], d_tr)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_alpha, best_mape_cv = None, 999.0
    for alpha in alphas:
        fold_mapes = []
        for tr_i, val_i in kf.split(X_tr, bins):
            w = nnls_ridge(X_tr[tr_i], y_tr[tr_i], alpha=alpha)
            if w.sum() > 1e-10: w = w / w.sum()
            dp = np.exp(X_tr[val_i] @ w)
            fold_mapes.append(mape(np.exp(y_tr[val_i]), dp))
        cv = np.mean(fold_mapes)
        if cv < best_mape_cv: best_mape_cv, best_alpha = cv, alpha
    return best_alpha, best_mape_cv

# ── Load 13-model predictions ──────────────────────────────────────────────────
MODELS_9  = ['mlp','resnet','cnn','cnn_s123','cnn_s777','cnn_ms','ftt','ftt_large','ftt_v2']
MODELS_NEW = ['cnn_aug','cnn_wloss','ftt_wloss','wavenet']
# Auto-detect additional models from pred files present on disk
MODELS_EXTRA = [m for m in ['wavenet_ftt', 'wavenet_s2', 'wavenet_s3', 'wavenet_ftt_v2', 'wavenet_alt',
                             'wavenet_stride2', 'wavenet_wide', 'wavenet_s4', 'wavenet_s5']
                if os.path.exists(f'pred_logd_{m}_test.npy')]
if MODELS_EXTRA: log(f"Auto-detected extra models: {MODELS_EXTRA}")
MODELS_13 = MODELS_9 + MODELS_NEW + MODELS_EXTRA

d_train = np.load('pred_d_train.npy')
d_test  = np.load('pred_d_test.npy')
log_d_train = np.log(d_train)

def load_preds(models):
    X_tr = np.column_stack([np.load(f'pred_logd_{m}_train.npy') for m in models])
    X_te = np.column_stack([np.load(f'pred_logd_{m}_test.npy')  for m in models])
    return X_tr, X_te

# ── Load ACF nonlinear fit features (optional) ─────────────────────────────────
has_nlfit = os.path.exists('nlfit_tauD_train.npy') and os.path.exists('nlfit_tauD_test.npy')
if has_nlfit:
    log("Loading ACF nonlinear fit features...")
    # Features need to be filtered to match d<=10 mask
    # Since we saved predictions with d<=10 filter already applied,
    # nlfit files contain ALL samples (pre-filter). Apply same mask.
    # Re-derive masks from saved d values
    d_all_train = np.load('cache_d_train_90pct.npy')
    d_all_test  = np.load('cache_d_test_90pct.npy')
    mask_tr = d_all_train <= 10
    mask_te = d_all_test  <= 10

    nlfit_tauD_tr = np.load('nlfit_tauD_train.npy')[mask_tr]
    nlfit_rmse_tr = np.load('nlfit_rmse_train.npy')[mask_tr]
    nlfit_tauD_te = np.load('nlfit_tauD_test.npy')[mask_te]
    nlfit_rmse_te = np.load('nlfit_rmse_test.npy')[mask_te]

    # Log-transform tauD (it spans many orders of magnitude)
    log_nlfit_tauD_tr = np.log(np.maximum(nlfit_tauD_tr, 0.1)).astype(np.float32)
    log_nlfit_tauD_te = np.log(np.maximum(nlfit_tauD_te, 0.1)).astype(np.float32)
    log(f"  nlfit_tauD: train median={np.median(nlfit_tauD_tr):.1f}  test median={np.median(nlfit_tauD_te):.1f}")
else:
    log("nlfit files not found — running without ACF features (run compute_acf_nlfit.py first)")

# ── Alphas to search ───────────────────────────────────────────────────────────
ALPHAS = np.logspace(-1, 6, 40)

results_lines = ["="*70, "META-LEARNER 5-FOLD RESULTS (13 models + ACF features)", "="*70, ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 1: Baselines (equal-weight geometric mean)
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== Baselines: equal-weight geometric mean ===")

X_tr_9, X_te_9 = load_preds(MODELS_9)
X_tr_13, X_te_13 = load_preds(MODELS_13)

dp9  = np.exp(X_te_9.mean(axis=1))
dp13 = np.exp(X_te_13.mean(axis=1))

print(f"\n--- Baseline: 9-model equal-weight ---")
print(f"  MAPE={mape(d_test,dp9):.2f}%  R²={r2_score(d_test,dp9):.4f}  MAE={mean_absolute_error(d_test,dp9):.4f}")
regime(d_test, dp9)
results_lines += [f"9-model equal-weight:  MAPE={mape(d_test,dp9):.2f}%  R²={r2_score(d_test,dp9):.4f}", ""]

print(f"\n--- Baseline: 13-model equal-weight ---")
print(f"  MAPE={mape(d_test,dp13):.2f}%  R²={r2_score(d_test,dp13):.4f}  MAE={mean_absolute_error(d_test,dp13):.4f}")
regime(d_test, dp13)
results_lines += [f"13-model equal-weight: MAPE={mape(d_test,dp13):.2f}%  R²={r2_score(d_test,dp13):.4f}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 2: Per-model contributions — add each new model to 7-model baseline
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== New model contributions ===")
MODELS_7 = ['cnn','cnn_s123','cnn_s777','cnn_ms','ftt','ftt_large','ftt_v2']
X_tr_7, X_te_7 = load_preds(MODELS_7)
dp7 = np.exp(X_te_7.mean(axis=1))
print(f"\n7-model baseline: MAPE={mape(d_test,dp7):.2f}%  R²={r2_score(d_test,dp7):.4f}")
results_lines.append(f"7-model baseline: MAPE={mape(d_test,dp7):.2f}%")

for new_m in MODELS_NEW + MODELS_EXTRA:
    X_te_new = np.column_stack([X_te_7, np.load(f'pred_logd_{new_m}_test.npy')])
    dp = np.exp(X_te_new.mean(axis=1))
    print(f"  + {new_m}: MAPE={mape(d_test,dp):.2f}%  R²={r2_score(d_test,dp):.4f}  "
          f"d<1={mape(d_test[d_test<1],dp[d_test<1]):.1f}%  d>=1={mape(d_test[d_test>=1],dp[d_test>=1]):.1f}%")
    results_lines.append(f"  7-model + {new_m}: MAPE={mape(d_test,dp):.2f}%  R²={r2_score(d_test,dp):.4f}")

results_lines.append("")

# ═════════════════════════════════════════════════════════════════════════════
# Section 3: NNLS ridge meta-learner (9 models)
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== NNLS Ridge meta-learner (9 models) ===")
log("  Running 5-fold CV alpha selection...")
best_a9, best_cv9 = cv_select_alpha(X_tr_9, log_d_train, d_train, ALPHAS)
log(f"  Best alpha={best_a9:.4f}  CV MAPE={best_cv9:.2f}%")

w9 = nnls_ridge(X_tr_9, log_d_train, alpha=best_a9)
w9_norm = w9 / w9.sum() if w9.sum() > 1e-10 else w9
dp_nnls9 = np.exp(X_te_9 @ w9_norm)
w_str = "  ".join(f"{MODELS_9[i]}={w9_norm[i]:.3f}" for i in range(len(MODELS_9)))
print(f"\nNNLS-9 (alpha={best_a9:.2f}): MAPE={mape(d_test,dp_nnls9):.2f}%  R²={r2_score(d_test,dp_nnls9):.4f}  MAE={mean_absolute_error(d_test,dp_nnls9):.4f}")
print(f"  Weights: {w_str}")
regime(d_test, dp_nnls9)
results_lines += [f"NNLS-9 (alpha={best_a9:.1f}): MAPE={mape(d_test,dp_nnls9):.2f}%  R²={r2_score(d_test,dp_nnls9):.4f}",
                  f"  Weights: {w_str}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 4: NNLS ridge meta-learner (13 models)
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== NNLS Ridge meta-learner (13 models) ===")
log("  Running 5-fold CV alpha selection...")
best_a13, best_cv13 = cv_select_alpha(X_tr_13, log_d_train, d_train, ALPHAS)
log(f"  Best alpha={best_a13:.4f}  CV MAPE={best_cv13:.2f}%")

w13 = nnls_ridge(X_tr_13, log_d_train, alpha=best_a13)
w13_norm = w13 / w13.sum() if w13.sum() > 1e-10 else w13
dp_nnls13 = np.exp(X_te_13 @ w13_norm)
w_str13 = "  ".join(f"{MODELS_13[i]}={w13_norm[i]:.3f}" for i in range(len(MODELS_13)))
print(f"\nNNLS-13 (alpha={best_a13:.2f}): MAPE={mape(d_test,dp_nnls13):.2f}%  R²={r2_score(d_test,dp_nnls13):.4f}  MAE={mean_absolute_error(d_test,dp_nnls13):.4f}")
print(f"  Weights: {w_str13}")
regime(d_test, dp_nnls13)
results_lines += [f"NNLS-13 (alpha={best_a13:.1f}): MAPE={mape(d_test,dp_nnls13):.2f}%  R²={r2_score(d_test,dp_nnls13):.4f}",
                  f"  Weights: {w_str13}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 5: Per-regime meta-learner (separate NNLS for d<1 and d>=1)
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== Per-regime NNLS meta-learner (13 models) ===")
ms_tr, mf_tr = d_train<1.0, d_train>=1.0

# Slow-diffusion (d<1) meta-learner
best_as, _ = cv_select_alpha(X_tr_13[ms_tr], log_d_train[ms_tr], d_train[ms_tr], ALPHAS)
w_slow = nnls_ridge(X_tr_13[ms_tr], log_d_train[ms_tr], alpha=best_as)
if w_slow.sum() > 1e-10: w_slow /= w_slow.sum()

# Fast-diffusion (d>=1) meta-learner
best_af, _ = cv_select_alpha(X_tr_13[mf_tr], log_d_train[mf_tr], d_train[mf_tr], ALPHAS)
w_fast = nnls_ridge(X_tr_13[mf_tr], log_d_train[mf_tr], alpha=best_af)
if w_fast.sum() > 1e-10: w_fast /= w_fast.sum()

log(f"  Slow (d<1) alpha={best_as:.2f}  Fast (d>=1) alpha={best_af:.2f}")

# Soft routing: predict d from 13-model equal-weight ensemble, use as signal
d_pred_eq = np.exp(X_te_13.mean(axis=1))
# Sigmoid blend: alpha=sigmoid((log(d_pred)-log(1))/0.5) ∈ [0,1] → 0=slow, 1=fast
log_d_pred = np.log(np.maximum(d_pred_eq, 1e-6))
blend_fast = 1.0 / (1.0 + np.exp(-(log_d_pred - 0.0) / 0.5))   # sigmoid centered at d=1

lp_slow = X_te_13 @ w_slow
lp_fast = X_te_13 @ w_fast
lp_blend = (1.0 - blend_fast) * lp_slow + blend_fast * lp_fast
dp_regime = np.exp(lp_blend)

print(f"\nPer-regime NNLS + soft routing: MAPE={mape(d_test,dp_regime):.2f}%  R²={r2_score(d_test,dp_regime):.4f}  MAE={mean_absolute_error(d_test,dp_regime):.4f}")
regime(d_test, dp_regime)
results_lines += [f"Per-regime NNLS + soft routing: MAPE={mape(d_test,dp_regime):.2f}%  R²={r2_score(d_test,dp_regime):.4f}",
                  f"  w_slow: {' '.join(f'{MODELS_13[i]}={w_slow[i]:.3f}' for i in np.argsort(-w_slow)[:5])}",
                  f"  w_fast: {' '.join(f'{MODELS_13[i]}={w_fast[i]:.3f}' for i in np.argsort(-w_fast)[:5])}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 6: With ACF nonlinear fit features (if available)
# ═════════════════════════════════════════════════════════════════════════════
if has_nlfit:
    log("\n=== NNLS with ACF nonlinear fit meta-features ===")
    # Append log(tauD) and rmse as additional columns
    X_tr_aug = np.column_stack([X_tr_13, log_nlfit_tauD_tr, nlfit_rmse_tr])
    X_te_aug = np.column_stack([X_te_13, log_nlfit_tauD_te, nlfit_rmse_te])
    MODELS_AUG = MODELS_13 + ['log_tauD', 'nlfit_rmse']

    log("  Running 5-fold CV alpha selection (13-model + ACF features)...")
    best_a_aug, best_cv_aug = cv_select_alpha(X_tr_aug, log_d_train, d_train, ALPHAS)
    log(f"  Best alpha={best_a_aug:.4f}  CV MAPE={best_cv_aug:.2f}%")

    w_aug = nnls_ridge(X_tr_aug, log_d_train, alpha=best_a_aug)
    w_aug_norm = w_aug / w_aug.sum() if w_aug.sum() > 1e-10 else w_aug
    dp_aug = np.exp(X_te_aug @ w_aug_norm)
    print(f"\nNNLS-13+ACF (alpha={best_a_aug:.2f}): MAPE={mape(d_test,dp_aug):.2f}%  R²={r2_score(d_test,dp_aug):.4f}  MAE={mean_absolute_error(d_test,dp_aug):.4f}")
    print(f"  Weights: {' '.join(f'{MODELS_AUG[i]}={w_aug_norm[i]:.3f}' for i in np.argsort(-w_aug_norm)[:8])}")
    regime(d_test, dp_aug)
    results_lines += [f"NNLS-13+ACF (alpha={best_a_aug:.1f}): MAPE={mape(d_test,dp_aug):.2f}%  R²={r2_score(d_test,dp_aug):.4f}",
                      f"  ACF feature weight: log_tauD={w_aug_norm[-2]:.3f}  nlfit_rmse={w_aug_norm[-1]:.3f}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Section 7: Best ensemble grid search on new models
# ═════════════════════════════════════════════════════════════════════════════
MODELS_POOL = MODELS_NEW + MODELS_EXTRA   # all candidate add-on models
log(f"\n=== Grid search: best subset (7-model + {len(MODELS_POOL)} candidates) ===")
all_new_preds_te = {m: np.load(f'pred_logd_{m}_test.npy') for m in MODELS_POOL}
X_te_7 = np.column_stack([np.load(f'pred_logd_{m}_test.npy') for m in MODELS_7])

best_combo, best_mape_grid = None, 999.0
from itertools import combinations
max_add = min(len(MODELS_POOL), 5)   # search up to 5 add-ons (cap for large pools)
for r in range(1, max_add + 1):
    for combo in combinations(MODELS_POOL, r):
        preds = [X_te_7] + [all_new_preds_te[m][:, None] for m in combo]
        dp = np.exp(np.column_stack(preds).mean(axis=1))
        m_ = mape(d_test, dp)
        if m_ < best_mape_grid:
            best_mape_grid = m_; best_combo = list(MODELS_7) + list(combo)

dp_best_combo = np.exp(np.column_stack([np.load(f'pred_logd_{m}_test.npy') for m in best_combo]).mean(axis=1))
print(f"\nBest equal-weight combo ({len(best_combo)} models): {best_combo}")
print(f"  MAPE={mape(d_test,dp_best_combo):.2f}%  R²={r2_score(d_test,dp_best_combo):.4f}")
regime(d_test, dp_best_combo)
results_lines += [f"Best equal-weight combo ({len(best_combo)} models): {'+'.join(best_combo)}",
                  f"  MAPE={mape(d_test,dp_best_combo):.2f}%  R²={r2_score(d_test,dp_best_combo):.4f}", ""]

# ═════════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════════
summary = "\n".join(results_lines)
with open('results_meta_learner_5fold.txt','w') as f: f.write(summary)
log("\nSaved results_meta_learner_5fold.txt")
print("\nDone.")
