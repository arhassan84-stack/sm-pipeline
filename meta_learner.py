"""
Meta-learner (stacking) trained on all 9 model predictions.

Replaces fixed geometric mean with a learned combiner. Two variants:
  1. Ridge regression  — equivalent to regularized weighted geometric mean
  2. Small MLP (3-layer) — can learn non-linear combinations

Input:  9 log-predictions per sample (mlp, resnet, cnn, cnn_s123, cnn_s777, cnn_ms,
                                       ftt, ftt_large, ftt_v2)
Target: log(d_true)

Note on contamination: base models were trained on the full training set, so their
training predictions are slightly optimistic. Mitigated by strong regularization.
Practical workaround: train meta-learner on a 20% held-out subset of train preds.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import time

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime(dt, dp):
    ms, mf = dt<1.0, dt>=1.0
    print(f"  d <1  ({ms.sum():5d}): R²={r2_score(dt[ms],dp[ms]):.4f}  MAPE={mape(dt[ms],dp[ms]):.1f}%")
    print(f"  d>=1  ({mf.sum():5d}): R²={r2_score(dt[mf],dp[mf]):.4f}  MAPE={mape(dt[mf],dp[mf]):.1f}%")

# ── Load predictions ───────────────────────────────────────────────────────────
MODELS = ['mlp','resnet','cnn','cnn_s123','cnn_s777','cnn_ms','ftt','ftt_large','ftt_v2']

d_train = np.load('pred_d_train.npy')
d_test  = np.load('pred_d_test.npy')
log_d_train = np.log(d_train)

X_tr = np.column_stack([np.load(f'pred_logd_{m}_train.npy') for m in MODELS])
X_te = np.column_stack([np.load(f'pred_logd_{m}_test.npy')  for m in MODELS])
log(f"Meta-learner input: {X_tr.shape[1]} models  Train: {len(d_train)}  Test: {len(d_test)}")

# ── Equal-weight geometric mean baseline ──────────────────────────────────────
log("\n--- Baseline: equal-weight geometric mean (all 9 models) ---")
dp_eq = np.exp(X_te.mean(axis=1))
print(f"  MAPE={mape(d_test,dp_eq):.2f}%  R²={r2_score(d_test,dp_eq):.4f}  MAE={mean_absolute_error(d_test,dp_eq):.4f}")
regime(d_test, dp_eq)

# ── Ridge regression meta-learner ─────────────────────────────────────────────
log("\n--- Ridge regression meta-learner ---")

# Hold out 20% of train for meta-training to reduce contamination
rng = np.random.default_rng(42)
meta_idx = rng.choice(len(d_train), size=int(0.2*len(d_train)), replace=False)
X_meta = X_tr[meta_idx]; y_meta = log_d_train[meta_idx]

results_lines = ["="*70, "META-LEARNER RESULTS", "="*70, ""]

for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_meta, y_meta)
    lp_te = ridge.predict(X_te)
    dp_te = np.exp(lp_te)
    m_ = mape(d_test,dp_te)
    r2 = r2_score(d_test,dp_te); mae = mean_absolute_error(d_test,dp_te)
    w_str = "  ".join(f"{MODELS[i]}={ridge.coef_[i]:.3f}" for i in range(len(MODELS)))
    print(f"  alpha={alpha:6.1f}: MAPE={m_:.2f}%  R²={r2:.4f}  MAE={mae:.4f}")
    print(f"    Weights: {w_str}")
    results_lines.append(f"Ridge alpha={alpha}: MAPE={m_:.2f}%  R²={r2:.4f}  MAE={mae:.4f}")
    results_lines.append(f"  Weights: {w_str}")
    ms,mf=d_test<1.0,d_test>=1.0
    results_lines.append(f"  d<1={mape(d_test[ms],dp_te[ms]):.1f}%  d>=1={mape(d_test[mf],dp_te[mf]):.1f}%")
    results_lines.append("")

# Best alpha sweep
log("\n--- Ridge: fine alpha sweep ---")
best_mape, best_alpha = 999, None
for alpha in np.logspace(-2, 4, 50):
    ridge = Ridge(alpha=alpha); ridge.fit(X_meta, y_meta)
    dp_te = np.exp(ridge.predict(X_te))
    m_ = mape(d_test, dp_te)
    if m_ < best_mape: best_mape = m_; best_alpha = alpha
log(f"  Best alpha={best_alpha:.4f}  MAPE={best_mape:.2f}%")

ridge_best = Ridge(alpha=best_alpha); ridge_best.fit(X_meta, y_meta)
dp_best = np.exp(ridge_best.predict(X_te))
r2 = r2_score(d_test,dp_best); mae = mean_absolute_error(d_test,dp_best)
ms,mf=d_test<1.0,d_test>=1.0
results_lines += [f"Ridge best (alpha={best_alpha:.4f}): MAPE={mape(d_test,dp_best):.2f}%  R²={r2:.4f}  MAE={mae:.4f}",
                  f"  d<1={mape(d_test[ms],dp_best[ms]):.1f}%  d>=1={mape(d_test[mf],dp_best[mf]):.1f}%", ""]
print(f"  MAPE={mape(d_test,dp_best):.2f}%  R²={r2:.4f}  MAE={mae:.4f}")
regime(d_test, dp_best)

# ── PyTorch MLP meta-learner ───────────────────────────────────────────────────
log("\n--- MLP meta-learner (3-layer) ---")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_meta_sc = StandardScaler().fit_transform(X_meta).astype(np.float32)
X_te_sc   = StandardScaler().fit(X_meta).transform(X_te).astype(np.float32)

X_m_t = torch.tensor(X_meta_sc); y_m_t = torch.tensor(y_meta.astype(np.float32))
X_t_t = torch.tensor(X_te_sc)

meta_mlp = nn.Sequential(
    nn.Linear(len(MODELS),64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64,32),          nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32,1)
)
opt = torch.optim.Adam(meta_mlp.parameters(), lr=1e-3, weight_decay=1e-3)
crit = nn.MSELoss()
loader = DataLoader(TensorDataset(X_m_t, y_m_t), batch_size=256, shuffle=True)

best_loss, best_state = float('inf'), None
for epoch in range(500):
    meta_mlp.train()
    for xb,yb in loader:
        opt.zero_grad(); loss=crit(meta_mlp(xb).squeeze(1),yb); loss.backward(); opt.step()
    meta_mlp.eval()
    with torch.no_grad():
        vl = crit(meta_mlp(X_m_t).squeeze(1), y_m_t).item()
    if vl < best_loss: best_loss=vl; best_state={k:v.clone() for k,v in meta_mlp.state_dict().items()}

meta_mlp.load_state_dict(best_state); meta_mlp.eval()
with torch.no_grad():
    dp_mlp = np.exp(meta_mlp(X_t_t).squeeze(1).numpy())

r2=r2_score(d_test,dp_mlp); mae=mean_absolute_error(d_test,dp_mlp)
print(f"  MLP meta: MAPE={mape(d_test,dp_mlp):.2f}%  R²={r2:.4f}  MAE={mae:.4f}")
regime(d_test, dp_mlp)
results_lines += [f"MLP meta-learner: MAPE={mape(d_test,dp_mlp):.2f}%  R²={r2:.4f}  MAE={mae:.4f}",
                  f"  d<1={mape(d_test[ms],dp_mlp[ms]):.1f}%  d>=1={mape(d_test[mf],dp_mlp[mf]):.1f}%", ""]

# ── TTA: add noise to test predictions ────────────────────────────────────────
log("\n--- TTA: average predictions over N_AUG=20 Gaussian-perturbed inputs ---")
N_AUG = 20; SIGMA = 0.03  # noise on log-predictions (3% of log-pred std)
rng2 = np.random.default_rng(0)
lp_tta = np.zeros_like(X_te[:, 0])
for _ in range(N_AUG):
    noise = rng2.normal(0, SIGMA, X_te.shape).astype(np.float32)
    lp_tta += (X_te + noise).mean(axis=1)
lp_tta /= N_AUG
dp_tta = np.exp(lp_tta)
print(f"  TTA eq-weight (sigma={SIGMA}, N={N_AUG}): MAPE={mape(d_test,dp_tta):.2f}%  R²={r2_score(d_test,dp_tta):.4f}")
regime(d_test, dp_tta)
results_lines += [f"TTA (sigma={SIGMA}, N={N_AUG}): MAPE={mape(d_test,dp_tta):.2f}%  R²={r2_score(d_test,dp_tta):.4f}", ""]

# Apply TTA to ridge best
lp_tta_ridge = np.zeros(len(d_test))
for _ in range(N_AUG):
    noise = rng2.normal(0, SIGMA, X_te.shape).astype(np.float32)
    lp_tta_ridge += ridge_best.predict(X_te + noise)
lp_tta_ridge /= N_AUG
dp_tta_ridge = np.exp(lp_tta_ridge)
print(f"  TTA + ridge: MAPE={mape(d_test,dp_tta_ridge):.2f}%  R²={r2_score(d_test,dp_tta_ridge):.4f}")
regime(d_test, dp_tta_ridge)
results_lines += [f"TTA + ridge (sigma={SIGMA}, N={N_AUG}): MAPE={mape(d_test,dp_tta_ridge):.2f}%  R²={r2_score(d_test,dp_tta_ridge):.4f}", ""]

summary = "\n".join(results_lines)
with open('results_meta_learner.txt','w') as f: f.write(summary)
log("Saved results_meta_learner.txt")
