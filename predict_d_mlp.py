"""
Step 3: MLP (Multi-Layer Perceptron) on the full feature set
(ACF + scattering + PSD, 207 features).
"""
import numpy as np
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
log_d_test  = np.log(d_test_eval)

# ── Scale features (critical for MLP) ────────────────────────────────────────
log("Scaling features...")
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test_eval)

# ── Train MLP ─────────────────────────────────────────────────────────────────
log("Training MLP (3 hidden layers: 512-256-128)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True,
)
mlp.fit(X_train_sc, log_d_train)
log(f"Training done — {mlp.n_iter_} iterations")

# ── Evaluate ──────────────────────────────────────────────────────────────────
d_pred_train = np.exp(mlp.predict(X_train_sc))
d_pred_test  = np.exp(mlp.predict(X_test_sc))

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
ax.set_title(f'MLP Test (d≤10)  R²={r2_score(d_test_eval, d_pred_test):.4f}')

ax = axes[1]
ax.plot(mlp.loss_curve_, label='Train loss')
if mlp.best_loss_ is not None:
    ax.axhline(mlp.best_loss_, color='r', ls='--', lw=1, label='Best val loss')
ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
ax.set_title('MLP Training Curve')
ax.legend()

plt.tight_layout()
plt.savefig('mlp_results.png', dpi=150)
log("Saved mlp_results.png")
print(f"Total runtime: {time.time()-t_total:.1f}s")
print("Done.")
