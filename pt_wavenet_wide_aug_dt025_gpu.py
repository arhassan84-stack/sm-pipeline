"""
WaveNet wide + augmentation — trained on dt025 simulation data (dt=0.25ms, 16384 bins/trace).

Architecture identical to pt_wavenet_wide_aug_gpu.py (WaveNet1D channels=256).
AdaptiveAvgPool1d(1) makes the trace branch length-agnostic: 16384-pt input
is handled without any architectural changes.

Key differences vs the dt=1ms aug model:
  - Input traces: 16384 bins (dt=0.25ms, tMax=4096ms) instead of 4096
  - Augmentation shift: ±512 bins (proportional to trace length: 128/4096 * 16384)
  - Training data: cache_i_train_dt025.npy / cache_X_train_dt025.npy / cache_d_train_dt025.npy
  - Test data:     cache_i_test_dt025.npy  / cache_X_test_dt025.npy  / cache_d_test_dt025.npy
  - Augmentation applied online per batch (no pre-built aug cache needed)
  - Feature dim: 302 (same pipeline, parameterised by N=16384 in build_cache_dt.py)

Label: pt_wavenet_wide_aug_dt025
"""

import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_wide_aug_dt025'
CHANNELS     = 256
BATCH_SIZE   = 128
EVAL_BATCH   = 128        # smaller eval batch: 16384-pt traces are 4× longer than dt=1ms
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

# Augmentation parameters (scaled proportionally to 16384-bin traces)
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03
AUG_SHIFT    = 512     # ±512 bins  (same proportion as ±128/4096 on dt=1ms)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


def augment_traces(traces, rng):
    """
    Online trace augmentation (identical types to wavenet_wide_aug pre-built cache):
      1. Amplitude scaling   ~ Uniform[0.85, 1.15]
      2. Additive noise      ~ N(0, 0.03)
      3. Cyclic shift        ~ Uniform[-512, +512] bins
    Applied to raw (unnormalised) traces; z-score normalisation follows.
    """
    N = traces.shape[0]
    scales = rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI, size=(N, 1)).astype(np.float32)
    noise  = rng.normal(0.0, AUG_NOISE_SD, size=traces.shape).astype(np.float32)
    shifts = rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, size=N)
    out    = traces * scales + noise
    out    = np.array([np.roll(row, s) for row, s in zip(out, shifts)], dtype=np.float32)
    return out


def norm_traces(traces):
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return (traces - mu) / sig


def batch_predict(model_fn, *cpu_tensors, bs=128):
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)


# ── Model ─────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn   = nn.BatchNorm1d(channels)
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


class WaveNet1D(nn.Module):
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap    = nn.AdaptiveAvgPool1d(1)   # handles any input length
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        x = self.input_proj(trace.unsqueeze(1))
        x = self.blocks(x)
        x = self.gap(x).squeeze(2)
        f = self.feat_branch(feats)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


# ── Data loading ──────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log("Loading dt025 cache (16384 bins/trace)...")

i_train_raw = np.load('cache_i_train_dt025.npy').astype(np.float32)
X_train     = np.load('cache_X_train_dt025.npy').astype(np.float32)
d_train     = np.load('cache_d_train_dt025.npy')

i_test_raw  = np.load('cache_i_test_dt025.npy').astype(np.float32)
X_test      = np.load('cache_X_test_dt025.npy').astype(np.float32)
d_test      = np.load('cache_d_test_dt025.npy')

mask_tr = d_train <= 10
i_train_raw, X_train, d_train = i_train_raw[mask_tr], X_train[mask_tr], d_train[mask_tr]
mask_te = d_test <= 10
i_test_raw,  X_test,  d_test  = i_test_raw[mask_te],  X_test[mask_te],  d_test[mask_te]

N_BINS = i_train_raw.shape[1]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Bins/trace: {N_BINS}")

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

# Validation: clean, normalised once
i_val_n = norm_traces(i_train_raw[val_idx])
X_val   = torch.tensor(X_tr_sc[val_idx])
y_val   = torch.tensor(log_d[val_idx])

# Index DataLoader — augmentation applied per batch
tr_idx_t   = torch.arange(len(tr_idx))
idx_loader = DataLoader(TensorDataset(tr_idx_t),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

X_tr_t = torch.tensor(X_tr_sc[tr_idx])
y_tr_t = torch.tensor(log_d[tr_idx])

rng_aug = np.random.default_rng(0)

# ── Model + optimiser ─────────────────────────────────────────────────────────

model = WaveNet1D(feat_dim=X_tr_t.shape[1], channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params    = sum(p.numel() for p in model.parameters())
rf_original = sum(2 * d for d in DILATIONS) * 4

log(f"Training {LABEL}  params={n_params:,}")
log(f"  channels={CHANNELS}  dilations={DILATIONS}  RF={rf_original} original lags")
log(f"  Input length: {N_BINS} bins  |  Augmentation shift: ±{AUG_SHIFT} bins")

# ── Training loop ─────────────────────────────────────────────────────────────

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep = 0.0

    for (batch_local_idx,) in idx_loader:
        global_idx = tr_idx[batch_local_idx.numpy()]

        # Online augmentation: scale + noise + shift, then z-score
        ib_raw = i_train_raw[global_idx]
        ib_aug = augment_traces(ib_raw, rng_aug)
        ib_n   = norm_traces(ib_aug)

        ib = torch.tensor(ib_n, dtype=torch.float32).to(device)
        xb = X_tr_t[batch_local_idx].to(device)
        yb = y_tr_t[batch_local_idx].to(device)

        opt.zero_grad()
        loss = crit(model(ib, xb), yb)
        loss.backward()
        opt.step()
        ep += loss.item() * len(ib)

    ep /= len(tr_idx)

    # Validation on clean traces
    model.eval()
    i_val_t = torch.tensor(i_val_n)
    with torch.no_grad():
        vl = crit(
            batch_predict(lambda i, x: model(i, x), i_val_t, X_val, bs=EVAL_BATCH),
            y_val,
        ).item()

    sched.step(epoch - 1)

    if vl < best_val - 1e-6:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.5f}")

# ── Save model + scaler ───────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)
log(f"Saved model_{LABEL}.pt + scaler")

# ── Evaluate on test set ──────────────────────────────────────────────────────

model.eval()
i_te_n = norm_traces(i_test_raw)
i_te_t = torch.tensor(i_te_n)
X_te_t = torch.tensor(X_te_sc)

with torch.no_grad():
    logd_pred_te = batch_predict(
        lambda i, x: model(i, x), i_te_t, X_te_t, bs=EVAL_BATCH
    ).numpy()

# Training predictions (clean, for meta-learner compatibility)
i_tr_all_n = norm_traces(i_train_raw)
i_tr_all_t = torch.tensor(i_tr_all_n)
X_tr_all_t = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(
        lambda i, x: model(i, x), i_tr_all_t, X_tr_all_t, bs=EVAL_BATCH
    ).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)
log(f"Saved predictions")

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"dt=0.25ms  N_BINS={N_BINS}  tMax=4096ms\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}\n"
    f"  Receptive field: {rf_original} original lags\n"
    f"  AdaptiveAvgPool1d handles {N_BINS}-pt input\n"
    f"Params: {n_params:,}\n"
    f"Augmentation: scale U[{AUG_SCALE_LO},{AUG_SCALE_HI}] + noise sd={AUG_NOISE_SD} + shift ±{AUG_SHIFT} (online)\n"
    f"Train: {len(d_train)}  Test: {len(d_test)}\n"
    f"Train R²={r2_score(d_train,d_pred_tr):.4f}  "
    f"MAE={mean_absolute_error(d_train,d_pred_tr):.4f}  "
    f"MAPE={mape(d_train,d_pred_tr):.1f}%\n"
    f"Test  R²={r2_score(d_test,d_pred_te):.4f}  "
    f"MAE={mean_absolute_error(d_test,d_pred_te):.4f}  "
    f"MAPE={mape(d_test,d_pred_te):.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred_te[ms]):.4f}  "
    f"MAPE={mape(d_test[ms],d_pred_te[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred_te[mf]):.4f}  "
    f"MAPE={mape(d_test[mf],d_pred_te[mf]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
