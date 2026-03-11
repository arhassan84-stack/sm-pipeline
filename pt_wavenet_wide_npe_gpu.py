"""
Neural Posterior Estimation (NPE) with Mixture Density Network (MDN) head.

Replaces the point-estimate regression head of WaveNet1D wide + aug with
a K=8 Gaussian MDN, yielding the full posterior p(D | trace, features).

Architecture:
  WaveNet1D backbone (channels=256):
    - stride-4 input projection: Conv1d(1→256, k=16, stride=4) → 1024 steps
    - 8 DilatedResBlocks(256, d) for d in [1,2,4,8,16,32,64,128]
    - AdaptiveAvgPool1d(1) → 256-dim trace embedding
    - Feature branch: feat_dim → 256 → 128
    - Context net: [256+128=384] → 256 → 64 (shared pre-MDN embedding)
  MDN head (K=8 Gaussian components):
    - pi_head:   Linear(64 → K) → softmax  → mixing weights π_k
    - mu_head:   Linear(64 → K)            → component means μ_k (log-D space)
    - lsig_head: Linear(64 → K), clamp[-6,2] → σ_k = exp(lsig)

Loss: Negative log-likelihood of Gaussian mixture
  log p(y|x) = log Σ_k π_k · N(y; μ_k, σ_k)

Warm-start: loads model_wavenet_wide_aug.pt
  - Backbone (input_proj, blocks, gap, feat_branch) → transferred directly
  - Context net ← head[0:6] (Linear 384→256→64) from original model
  - MDN heads (pi/mu/lsig) → randomly initialised

Point estimate:  E[log D] = Σ_k π_k · μ_k
Posterior std:   √(Σ_k π_k · (σ_k² + (μ_k − E[log D])²))

Label: pt_wavenet_wide_npe
"""

import numpy as np, time, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_wide_npe'
K            = 8          # Gaussian mixture components
CHANNELS     = 256
BATCH_SIZE   = 128
EVAL_BATCH   = 256
MAX_EPOCHS   = 120
PATIENCE     = 20
LR           = 1e-4       # 5× lower than original (fine-tuning regime)
WEIGHT_DECAY = 1e-4
T_MAX        = 100
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── Model ─────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


class WaveNetNPE(nn.Module):
    """WaveNet1D backbone with Mixture Density Network head."""

    def __init__(self, feat_dim, channels=256, dilations=None, K=8):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.K = K

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap     = nn.AdaptiveAvgPool1d(1)

        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )

        # Shared context embedding (mirrors head[0:6] of WaveNet1D)
        self.context_net = nn.Sequential(
            nn.Linear(channels + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64),             nn.ReLU(),
        )

        # MDN heads — randomly initialised
        self.pi_head   = nn.Linear(64, K)   # mixing logits
        self.mu_head   = nn.Linear(64, K)   # component means
        self.lsig_head = nn.Linear(64, K)   # log standard deviations

    def _embed(self, trace, feats):
        """Return 64-dim context vector for each sample."""
        x = self.input_proj(trace.unsqueeze(1))
        x = self.blocks(x)
        x = self.gap(x).squeeze(2)
        f = self.feat_branch(feats)
        return self.context_net(torch.cat([x, f], dim=1))  # (N, 64)

    def forward(self, trace, feats):
        """Return (pi, mu, sigma) each shape (N, K)."""
        h     = self._embed(trace, feats)
        pi    = F.softmax(self.pi_head(h), dim=1)
        mu    = self.mu_head(h)
        sigma = torch.exp(self.lsig_head(h).clamp(-6.0, 2.0))
        return pi, mu, sigma

    def predict(self, trace, feats):
        """Posterior mean and std in log-D space. Returns (mean, std)."""
        pi, mu, sigma = self.forward(trace, feats)
        mean = (pi * mu).sum(1)                                           # (N,)
        var  = (pi * (sigma**2 + (mu - mean.unsqueeze(1))**2)).sum(1)    # (N,)
        return mean, var.sqrt()


def mdn_nll(pi, mu, sigma, y):
    """Negative log-likelihood of Gaussian mixture.
    pi, mu, sigma: (N, K);  y: (N,)
    Uses log-sum-exp for numerical stability.
    """
    y         = y.unsqueeze(1)                                           # (N, 1)
    log_norm  = (-0.5 * ((y - mu) / sigma)**2
                 - sigma.log()
                 - 0.5 * np.log(2.0 * np.pi))                           # (N, K)
    log_comp  = torch.log(pi + 1e-8) + log_norm                         # (N, K)
    return -torch.logsumexp(log_comp, dim=1).mean()


def batch_nll(model, i_tens, x_tens, y_tens, bs=256):
    """Compute mean NLL on CPU tensors without gradient."""
    model.eval()
    total, n = 0.0, len(y_tens)
    with torch.no_grad():
        for j in range(0, n, bs):
            ij = i_tens[j:j+bs].to(device)
            xj = x_tens[j:j+bs].to(device)
            yj = y_tens[j:j+bs].to(device)
            pi, mu, sig = model(ij, xj)
            total += mdn_nll(pi, mu, sig, yj).item() * len(yj)
    return total / n


def batch_predict(model, i_tens, x_tens, bs=256):
    """Return (mean_logd, std_logd) numpy arrays."""
    model.eval()
    means, stds = [], []
    with torch.no_grad():
        for j in range(0, len(i_tens), bs):
            ij = i_tens[j:j+bs].to(device)
            xj = x_tens[j:j+bs].to(device)
            m, s = model.predict(ij, xj)
            means.append(m.cpu()); stds.append(s.cpu())
    return torch.cat(means).numpy(), torch.cat(stds).numpy()


# ── Data loading ──────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log("Loading aug cache + 90% test set...")

i_train = np.load('cache_i_train_aug.npy').astype(np.float32)
X_train = np.load('cache_X_train_aug.npy').astype(np.float32)
d_train = np.load('cache_d_train_aug.npy')
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test  = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test  = np.load('cache_d_test_90pct.npy')

mask_tr = d_train <= 10
i_train, X_train, d_train = i_train[mask_tr], X_train[mask_tr], d_train[mask_tr]
mask_te = d_test <= 10
i_test,  X_test,  d_test  = i_test[mask_te],  X_test[mask_te],  d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

# z-score normalise traces per-sample
i_train_n = (i_train - i_train.mean(1, keepdims=True)) / (i_train.std(1, keepdims=True) + 1e-8)
i_test_n  = (i_test  - i_test.mean(1,  keepdims=True)) / (i_test.std(1,  keepdims=True) + 1e-8)

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng     = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_tr  = torch.tensor(i_train_n[tr_idx]);  X_tr  = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d[tr_idx])
i_val = torch.tensor(i_train_n[val_idx]); X_val = torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d[val_idx])

loader = DataLoader(TensorDataset(i_tr, X_tr, y_tr),
                    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

feat_dim = X_tr.shape[1]

# ── Build model + warm-start from wavenet_wide_aug ────────────────────────────

model = WaveNetNPE(feat_dim=feat_dim, channels=CHANNELS,
                   dilations=DILATIONS, K=K).to(device)

log("Warm-starting backbone from model_wavenet_wide_aug.pt...")
ckpt = torch.load('model_wavenet_wide_aug.pt', map_location=device)
own  = model.state_dict()

transferred, skipped = 0, 0
for ck, cv in ckpt.items():
    # Map original head.{0-5}.* → context_net.{0-5}.*
    nk = (ck.replace('head.0.', 'context_net.0.')
             .replace('head.1.', 'context_net.1.')
             .replace('head.2.', 'context_net.2.')
             .replace('head.3.', 'context_net.3.')
             .replace('head.4.', 'context_net.4.')
             .replace('head.5.', 'context_net.5.'))
    # Skip head.6.* (the old Linear 64→1) — MDN heads stay random
    if 'head.6' in ck:
        skipped += 1
        continue
    if nk in own and own[nk].shape == cv.shape:
        own[nk] = cv
        transferred += 1
    else:
        skipped += 1

model.load_state_dict(own)
log(f"  Transferred {transferred} parameter tensors  |  skipped {skipped}")

# ── Optimiser ─────────────────────────────────────────────────────────────────

opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=1e-6)

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  K={K} MDN components")
log(f"  LR={LR}  T_max={T_MAX}  PATIENCE={PATIENCE}  MAX_EPOCHS={MAX_EPOCHS}")

# ── Training loop ─────────────────────────────────────────────────────────────

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0

    for ib, xb, yb in loader:
        ib, xb, yb = ib.to(device), xb.to(device), yb.to(device)
        opt.zero_grad()
        pi, mu, sigma = model(ib, xb)
        loss = mdn_nll(pi, mu, sigma, yb)
        loss.backward()
        opt.step()
        ep += loss.item() * len(ib)

    ep /= len(i_tr)

    vl = batch_nll(model, i_val, X_val, y_val, bs=EVAL_BATCH)
    sched.step()

    if vl < best_val - 1e-6:
        best_val, best_state, patience_count = (
            vl, {k: v.clone() for k, v in model.state_dict().items()}, 0)
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train_nll={ep:.5f}  val_nll={vl:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done training — best val NLL={best_val:.5f}")

# ── Save model + scaler ───────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)
log(f"Saved model_{LABEL}.pt + scaler")

# ── Evaluate on test set ──────────────────────────────────────────────────────

i_te_t = torch.tensor(i_test_n)
X_te_t = torch.tensor(X_te_sc)
logd_pred_te, logd_std_te = batch_predict(model, i_te_t, X_te_t, bs=EVAL_BATCH)

# Full training set predictions (clean, for meta-learner compatibility)
i_tr_all_t = torch.tensor(i_train_n)
X_tr_all_t = torch.tensor(X_tr_sc)
logd_pred_tr, logd_std_tr = batch_predict(model, i_tr_all_t, X_tr_all_t, bs=EVAL_BATCH)

np.save(f'pred_logd_{LABEL}_test.npy',      logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy',     logd_pred_tr)
np.save(f'pred_logd_std_{LABEL}_test.npy',  logd_std_te)
np.save(f'pred_logd_std_{LABEL}_train.npy', logd_std_tr)
log("Saved predictions + uncertainty arrays")

d_pred_te = np.exp(logd_pred_te)
d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total

ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"Architecture: WaveNetNPE (WaveNet1D backbone + MDN head)\n"
    f"  channels={CHANNELS}  dilations={DILATIONS}  K={K} Gaussian components\n"
    f"  Loss: Negative Log-Likelihood (Gaussian mixture)\n"
    f"  Warm-started from model_wavenet_wide_aug.pt\n"
    f"  LR={LR} (fine-tune)  T_max={T_MAX}  PATIENCE={PATIENCE}\n"
    f"Params: {n_params:,}\n"
    f"Train: {len(d_train)}  Test: {len(d_test)}\n"
    f"Train R²={r2_score(d_train, d_pred_tr):.4f}  "
    f"MAE={mean_absolute_error(d_train, d_pred_tr):.4f}  "
    f"MAPE={mape(d_train, d_pred_tr):.1f}%\n"
    f"Test  R²={r2_score(d_test, d_pred_te):.4f}  "
    f"MAE={mean_absolute_error(d_test, d_pred_te):.4f}  "
    f"MAPE={mape(d_test, d_pred_te):.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms], d_pred_te[ms]):.4f}  "
    f"MAPE={mape(d_test[ms], d_pred_te[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf], d_pred_te[mf]):.4f}  "
    f"MAPE={mape(d_test[mf], d_pred_te[mf]):.1f}%\n"
    f"Uncertainty (test set, posterior std in log-D space):\n"
    f"  median σ={np.median(logd_std_te):.4f}  "
    f"mean σ={np.mean(logd_std_te):.4f}  "
    f"p95 σ={np.percentile(logd_std_te, 95):.4f}\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
