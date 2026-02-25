"""
MLP 2048-1024-512-256-128 on 90% data — wider + deeper experiment.
"""
import numpy as np
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL       = 'mlp_90pct_2048b'
RESULT_FILE = f'results_{LABEL}.txt'

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

train_mask = d_train <= 10
X_train, d_train = X_train[train_mask], d_train[train_mask]
test_mask  = d_test <= 10
X_test_eval, d_test_eval = X_test[test_mask], d_test[test_mask]
log(f"Train: {len(d_train)}  Test: {len(d_test_eval)}")

log_d_train = np.log(d_train)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test_eval)

log("Training MLP 2048-1024-512-256-128, max_iter=2000...")
mlp = MLPRegressor(
    hidden_layer_sizes=(2048, 1024, 512, 256, 128),
    activation='relu', solver='adam',
    learning_rate_init=5e-4,
    max_iter=2000,
    tol=1e-5,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
    random_state=42, verbose=True,
)
mlp.fit(X_tr_sc, log_d_train)
log(f"Done — {mlp.n_iter_} iterations")

d_pred_train = np.exp(mlp.predict(X_tr_sc))
d_pred_test  = np.exp(mlp.predict(X_te_sc))
runtime = time.time() - t_total

r2_tr   = r2_score(d_train, d_pred_train)
mae_tr  = mean_absolute_error(d_train, d_pred_train)
mape_tr = mape(d_train, d_pred_train)
r2_te   = r2_score(d_test_eval, d_pred_test)
mae_te  = mean_absolute_error(d_test_eval, d_pred_test)
mape_te = mape(d_test_eval, d_pred_test)

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 2048-1024-512-256-128  max_iter=2000  lr=5e-4  tol=1e-5  n_iter_no_change=50\n"
    f"Train size: {len(d_train)}  Test size: {len(d_test_eval)}\n"
    f"Iterations run: {mlp.n_iter_}\n\n"
    f"Train  R²={r2_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape_tr:.1f}%\n"
    f"Test   R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(RESULT_FILE, 'w') as fh:
    fh.write(summary)
log(f"Saved {RESULT_FILE}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(d_test_eval, d_pred_test, alpha=0.2, s=5, color='steelblue')
lims = [min(d_test_eval.min(), d_pred_test.min()), max(d_test_eval.max(), d_pred_test.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'{LABEL}  R²={r2_te:.4f}')
ax = axes[1]
ax.plot(mlp.loss_curve_, color='steelblue', label='Train loss')
ax.set_xlabel('Iteration'); ax.set_ylabel('Loss'); ax.set_title('Training Curve'); ax.legend()
plt.tight_layout()
plt.savefig(f'{LABEL}.png', dpi=150)
log(f"Saved {LABEL}.png")
print("Done.")
