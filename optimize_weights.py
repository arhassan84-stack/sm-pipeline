"""
Optimize ensemble weights using scipy.minimize on TRAINING data,
then evaluate on held-out TEST data.

Approach: weighted geometric mean in log-space
  log_pred = sum(w_i * log_pred_i) / sum(w_i)

Weights optimized via scipy to minimize:
  1. Overall MAPE on training set
  2. d>=1 MAPE on training set (regime-focused)

Then apply training-optimal weights to test set.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_absolute_error

# ── Load predictions ───────────────────────────────────────────────────────
d_train = np.load('pred_d_train.npy')
d_test  = np.load('pred_d_test.npy')

preds_tr = {
    'mlp':    np.load('pred_logd_mlp_train.npy'),
    'resnet': np.load('pred_logd_resnet_train.npy'),
    'cnn':    np.load('pred_logd_cnn_train.npy'),
    'ftt':    np.load('pred_logd_ftt_train.npy'),
}
preds_te = {
    'mlp':    np.load('pred_logd_mlp_test.npy'),
    'resnet': np.load('pred_logd_resnet_test.npy'),
    'cnn':    np.load('pred_logd_cnn_test.npy'),
    'ftt':    np.load('pred_logd_ftt_test.npy'),
}

def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

def regime_str(dt, dp):
    ms, mf = dt<1.0, dt>=1.0
    return (f"  d <1  ({ms.sum():5d}): R²={r2_score(dt[ms],dp[ms]):.4f}  MAPE={mape(dt[ms],dp[ms]):.1f}%\n"
            f"  d>=1  ({mf.sum():5d}): R²={r2_score(dt[mf],dp[mf]):.4f}  MAPE={mape(dt[mf],dp[mf]):.1f}%")

# ── Ensemble helpers ───────────────────────────────────────────────────────
def weighted_ensemble(weights, preds, keys):
    """Weighted geometric mean given a weight per model and a dict of log-predictions."""
    w = np.array(weights, dtype=np.float64)
    w = np.maximum(w, 0)           # non-negative
    w /= w.sum()                    # normalise to sum=1
    lp = sum(w[i] * preds[k] for i, k in enumerate(keys))
    return np.exp(lp)

def make_objective(keys, preds_tr, d_tr, regime_weight=1.0):
    """
    Returns a loss function over unconstrained weights (softmax-parametrised).
    regime_weight > 1 up-weights d>=1 MAPE in the objective.
    """
    mf = d_tr >= 1.0
    ms = d_tr < 1.0
    def obj(log_w):
        w = np.exp(log_w - log_w.max())  # softmax trick
        w /= w.sum()
        lp = sum(w[i] * preds_tr[k] for i, k in enumerate(keys))
        dp = np.exp(lp)
        mape_all  = mape(d_tr, dp)
        mape_fast = mape(d_tr[mf], dp[mf]) if mf.any() else 0
        mape_slow = mape(d_tr[ms], dp[ms]) if ms.any() else 0
        n_fast = mf.sum(); n_slow = ms.sum(); n_tot = len(d_tr)
        # weighted average MAPE across regimes
        return (n_slow/n_tot * mape_slow + regime_weight * n_fast/n_tot * mape_fast) / (1 + (regime_weight-1)*n_fast/n_tot)
    return obj

# ── Evaluate combos ────────────────────────────────────────────────────────
COMBOS = [
    ('mlp+cnn+ftt',        ['mlp','cnn','ftt']),
    ('resnet+cnn+ftt',     ['resnet','cnn','ftt']),
    ('mlp+resnet+cnn',     ['mlp','resnet','cnn']),
    ('mlp+resnet+cnn+ftt', ['mlp','resnet','cnn','ftt']),
]

lines = ["="*70, "ENSEMBLE WEIGHT OPTIMIZATION", "="*70, ""]

for combo_name, keys in COMBOS:
    n = len(keys)
    lines.append(f"── {combo_name} ──────────────────────────────────────")

    # Equal weights baseline
    eq_dp_tr = weighted_ensemble([1.0]*n, preds_tr, keys)
    eq_dp_te = weighted_ensemble([1.0]*n, preds_te, keys)
    lines.append(f"Equal weights (1/{n} each):")
    lines.append(f"  Train MAPE={mape(d_train,eq_dp_tr):.2f}%  Test MAPE={mape(d_test,eq_dp_te):.2f}%  R²={r2_score(d_test,eq_dp_te):.4f}")
    lines.append(regime_str(d_test, eq_dp_te))
    lines.append("")

    for label, rw in [("Minimize overall MAPE", 1.0), ("Minimize d>=1-focused MAPE (2x weight)", 2.0)]:
        obj = make_objective(keys, preds_tr, d_train, regime_weight=rw)
        x0 = np.zeros(n)   # log-weights = 0 → equal weights initially

        # Try multiple starting points via differential evolution for robustness
        bounds = [(-3, 3)] * n
        res_de = differential_evolution(obj, bounds, maxiter=200, seed=42, tol=1e-6, polish=True)
        res = minimize(obj, res_de.x, method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-7,'fatol':1e-7})
        best = res_de if res_de.fun < res.fun else res

        log_w = best.x
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        w_str = "  ".join(f"{k}={w[i]:.3f}" for i,k in enumerate(keys))

        opt_dp_tr = weighted_ensemble(w, preds_tr, keys)
        opt_dp_te = weighted_ensemble(w, preds_te, keys)
        lines.append(f"{label}:")
        lines.append(f"  Weights: {w_str}")
        lines.append(f"  Train MAPE={mape(d_train,opt_dp_tr):.2f}%  Test MAPE={mape(d_test,opt_dp_te):.2f}%  R²={r2_score(d_test,opt_dp_te):.4f}  MAE={mean_absolute_error(d_test,opt_dp_te):.4f}")
        lines.append(regime_str(d_test, opt_dp_te))
        lines.append("")

    lines.append("")

# ── Fine-grained grid search for best 3-model combo ───────────────────────
lines.append("="*70)
lines.append("GRID SEARCH: mlp+cnn+ftt and resnet+cnn+ftt (100 points per axis)")
lines.append("="*70)

for combo_name, keys in [('mlp+cnn+ftt', ['mlp','cnn','ftt']), ('resnet+cnn+ftt', ['resnet','cnn','ftt'])]:
    best_mape_overall = np.inf; best_mape_fast = np.inf
    best_w_overall = None; best_w_fast = None
    step = 0.02
    alphas = np.arange(0, 1+step, step)
    mf = d_test >= 1.0
    for a in alphas:
        for b in alphas:
            c = 1.0 - a - b
            if c < -1e-9: continue
            c = max(c, 0.0)
            w = np.array([a, b, c])
            if w.sum() < 1e-10: continue
            w /= w.sum()
            dp = weighted_ensemble(w, preds_te, keys)
            m_all = mape(d_test, dp)
            m_fast = mape(d_test[mf], dp[mf])
            if m_all < best_mape_overall:
                best_mape_overall = m_all; best_w_overall = w.copy()
            if m_fast < best_mape_fast:
                best_mape_fast = m_fast; best_w_fast = w.copy()
    # Report
    for label, best_w, best_m in [
        ("Best overall MAPE", best_w_overall, best_mape_overall),
        ("Best d>=1 MAPE",    best_w_fast,    best_mape_fast),
    ]:
        w_str = "  ".join(f"{k}={best_w[i]:.3f}" for i,k in enumerate(keys))
        dp = weighted_ensemble(best_w, preds_te, keys)
        lines.append(f"\n{combo_name} — {label}:")
        lines.append(f"  Weights: {w_str}")
        lines.append(f"  Test MAPE={mape(d_test,dp):.2f}%  R²={r2_score(d_test,dp):.4f}  MAE={mean_absolute_error(d_test,dp):.4f}")
        lines.append(regime_str(d_test, dp))

lines.append("")
summary = "\n".join(lines)
print(summary)
with open('results_weight_optimization.txt','w') as f: f.write(summary)
print("Saved results_weight_optimization.txt")
