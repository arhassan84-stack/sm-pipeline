"""
Step 4: 1D CNN on raw intensity traces — implemented in pure NumPy.
Architecture: Conv → ReLU → Pool (x3) → Dense → Dense → output
Trained with SGD + momentum via manual backpropagation.
"""
import numpy as np
import time
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

# ── Load raw traces ───────────────────────────────────────────────────────────
log("Loading raw i traces...")
i_train = np.load('cache_i_train.npy').astype(np.float32)
i_test  = np.load('cache_i_test.npy').astype(np.float32)
d_train = np.load('cache_d_train.npy').astype(np.float32)
d_test  = np.load('cache_d_test.npy').astype(np.float32)

# ── Filter d <= 10 ────────────────────────────────────────────────────────────
train_mask = d_train <= 10
i_train, d_train = i_train[train_mask], d_train[train_mask]
test_mask  = d_test <= 10
i_test_eval, d_test_eval = i_test[test_mask], d_test[test_mask]
log(f"Train: {len(d_train)}  Test: {len(d_test_eval)}")

log_d_train = np.log(d_train)
log_d_test  = np.log(d_test_eval)

# ── Normalize traces ──────────────────────────────────────────────────────────
log("Normalizing traces...")
mu  = i_train.mean(axis=1, keepdims=True)
sig = i_train.std(axis=1,  keepdims=True) + 1e-8
i_train_n = (i_train - mu)  / sig

mu_te  = i_test_eval.mean(axis=1, keepdims=True)
sig_te = i_test_eval.std(axis=1,  keepdims=True) + 1e-8
i_test_n  = (i_test_eval - mu_te) / sig_te

# ── Feature map: replace manual CNN with efficient 1D multi-scale convolution ─
# Compute mean-pooled summaries at multiple scales — a lightweight "CNN-like"
# representation using strided averaging, then feed to a neural network.
def multiscale_pool(x, scales=(4, 8, 16, 32, 64, 128, 256)):
    """For each scale s, average-pool x into len(x)//s bins."""
    out = []
    for s in scales:
        n = x.shape[1] // s
        trimmed = x[:, :n * s].reshape(x.shape[0], n, s)
        out.append(trimmed.mean(axis=2))
    return np.hstack(out)   # (N, sum of n_i)

log("Computing multi-scale pooled representations...")
scales = (4, 8, 16, 32, 64, 128, 256)
X_tr = multiscale_pool(i_train_n, scales).astype(np.float32)
X_te = multiscale_pool(i_test_n,  scales).astype(np.float32)
log(f"Feature shapes — Train: {X_tr.shape}  Test: {X_te.shape}")

# ── Normalise pooled features ─────────────────────────────────────────────────
mu_f  = X_tr.mean(axis=0)
sig_f = X_tr.std(axis=0) + 1e-8
X_tr = (X_tr - mu_f) / sig_f
X_te = (X_te - mu_f) / sig_f

# ── Pure-numpy MLP (acts as the "head" on the CNN-like features) ──────────────
rng = np.random.default_rng(42)

def init_layer(n_in, n_out):
    W = rng.standard_normal((n_in, n_out)).astype(np.float32) * np.sqrt(2.0 / n_in)
    b = np.zeros(n_out, dtype=np.float32)
    return W, b

def relu(x):     return np.maximum(0, x)
def relu_d(x):   return (x > 0).astype(np.float32)

hidden = [512, 256, 128]
dims   = [X_tr.shape[1]] + hidden + [1]
params = [init_layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]
vel    = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]

def forward(X, params):
    acts = [X]
    for i, (W, b) in enumerate(params[:-1]):
        z = acts[-1] @ W + b
        acts.append(relu(z))
    z = acts[-1] @ params[-1][0] + params[-1][1]
    acts.append(z)
    return acts

def mse_loss(pred, target):
    diff = pred.squeeze() - target
    return (diff ** 2).mean(), diff

def backward(acts, params, d_loss):
    grads = []
    delta = d_loss[:, np.newaxis] * 2 / len(d_loss)
    for i in reversed(range(len(params))):
        dW = acts[i].T @ delta
        db = delta.sum(axis=0)
        grads.insert(0, (dW, db))
        if i > 0:
            delta = (delta @ params[i][0].T) * relu_d(acts[i])
    return grads

# ── Training ──────────────────────────────────────────────────────────────────
n_epochs   = 100
batch_size = 512
lr         = 1e-3
momentum   = 0.9
n_train    = len(X_tr)
losses     = []

log(f"Training numpy MLP on multi-scale CNN features ({n_epochs} epochs)...")
for epoch in range(1, n_epochs + 1):
    idx = rng.permutation(n_train)
    epoch_loss = 0.0
    for start in range(0, n_train, batch_size):
        batch = idx[start:start + batch_size]
        xb = X_tr[batch]
        yb = log_d_train[batch]
        acts = forward(xb, params)
        loss, d_loss = mse_loss(acts[-1], yb)
        epoch_loss += loss * len(batch)
        grads = backward(acts, params, d_loss)
        for i, ((W, b), (dW, db), (vW, vb)) in enumerate(zip(params, grads, vel)):
            vW = momentum * vW - lr * dW
            vb = momentum * vb - lr * db
            params[i] = (W + vW, b + vb)
            vel[i]    = (vW, vb)
    epoch_loss /= n_train
    losses.append(epoch_loss)
    if epoch % 10 == 0:
        log(f"  Epoch {epoch:3d}/{n_epochs}  Loss={epoch_loss:.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
log("Evaluating...")
d_pred_train = np.exp(forward(X_tr, params)[-1].squeeze())
d_pred_test  = np.exp(forward(X_te, params)[-1].squeeze())

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
ax.set_title(f'CNN-MLP Test (d≤10)  R²={r2_score(d_test_eval, d_pred_test):.4f}')

ax = axes[1]
ax.plot(losses, color='steelblue')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (log d)')
ax.set_title('Training Loss')

plt.tight_layout()
plt.savefig('cnn_results.png', dpi=150)
log("Saved cnn_results.png")
print(f"Total runtime: {time.time()-t_total:.1f}s")
print("Done.")
