"""
WaveNet wide + augmentation — trained on dt010 simulation data (dt=0.1ms, 40960 bins/trace).

Architecture identical to pt_wavenet_wide_aug_gpu.py (WaveNet1D channels=256).
AdaptiveAvgPool1d(1) makes the trace branch length-agnostic: 40960-pt input
is handled without any architectural changes.

Key differences vs the dt=1ms aug model:
  - Input traces: 40960 bins (dt=0.1ms, tMax=4096ms) instead of 4096
  - Augmentation shift: ±1280 bins (proportional: 128/4096 × 40960)
  - Training data: cache_i_train_dt010.npy / cache_X_train_dt010.npy / cache_d_train_dt010.npy
  - Test data:     cache_i_test_dt010.npy  / cache_X_test_dt010.npy  / cache_d_test_dt010.npy
  - Augmentation runs inside DataLoader workers (num_workers=4), fully overlapping
    with GPU forward/backward pass — zero GPU idle time waiting for data prep
  - EVAL_BATCH=64 (40960-pt traces are 10× longer than dt=1ms)

Label: pt_wavenet_wide_aug_dt010
"""

import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_wide_aug_dt010'
CHANNELS     = 256
BATCH_SIZE   = 32
EVAL_BATCH   = 32
DL_WORKERS   = 4          # DataLoader workers — augmentation runs here in parallel
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

# Augmentation parameters (scaled proportionally to 40960-bin traces)
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03
AUG_SHIFT    = 1280    # ±1280 bins  (same proportion as ±128/4096 on dt=1ms)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── Dataset with worker-parallel augmentation ─────────────────────────────────

class AugFCSDataset(Dataset):
    """
    Returns augmented, z-score-normalised traces from worker processes.
    Each DataLoader worker gets an independent RNG seeded from worker_info.seed,
    so augmentation runs in all 4 workers concurrently while the GPU trains.
    """
    def __init__(self, i_raw, X_tensor, y_tensor):
        # i_raw: numpy float32 (N, N_BINS) — raw unnormalised traces in shared memory
        # X_tensor, y_tensor: CPU tensors (pre-computed once in main process)
        self.i_raw = i_raw
        self.X     = X_tensor
        self.y     = y_tensor
        self._rng  = None     # initialised lazily per worker

    def __len__(self):
        return len(self.y)

    @property
    def rng(self):
        if self._rng is None:
            wi   = torch.utils.data.get_worker_info()
            seed = int(wi.seed % (2**31)) if wi is not None else 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, idx):
        trace = self.i_raw[idx]                    # 1D view — not copied until mutated

        # Augment (creates new arrays — original shared memory unchanged)
        scale = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        noise = self.rng.normal(0.0, AUG_NOISE_SD, size=trace.shape).astype(np.float32)
        shift = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))

        aug = trace * scale + noise
        aug = np.roll(aug, shift)

        # Per-trace z-score normalisation
        aug = (aug - aug.mean()) / (aug.std() + 1e-8)

        return torch.from_numpy(aug.astype(np.float32)), self.X[idx], self.y[idx]


def norm_traces(traces):
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return (traces - mu) / sig


def batch_predict(model_fn, *cpu_tensors, bs=64):
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
        self.gap    = nn.AdaptiveAvgPool1d(1)
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
log("Loading dt010 cache (40960 bins/trace)...")

i_train_raw = np.load('cache_i_train_dt010.npy').astype(np.float32)
X_train     = np.load('cache_X_train_dt010.npy').astype(np.float32)
d_train     = np.load('cache_d_train_dt010.npy')

i_test_raw  = np.load('cache_i_test_dt010.npy').astype(np.float32)
X_test      = np.load('cache_X_test_dt010.npy').astype(np.float32)
d_test      = np.load('cache_d_test_dt010.npy')

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

# Validation: clean, normalised once — not augmented
i_val_n = norm_traces(i_train_raw[val_idx])
X_val   = torch.tensor(X_tr_sc[val_idx])
y_val   = torch.tensor(log_d[val_idx])

# Training DataLoader — augmentation runs in 4 worker processes in parallel
train_ds = AugFCSDataset(
    i_raw    = i_train_raw[tr_idx],          # shared across workers (fork, copy-on-write)
    X_tensor = torch.tensor(X_tr_sc[tr_idx]),
    y_tensor = torch.tensor(log_d[tr_idx]),
)
loader = DataLoader(
    train_ds,
    batch_size        = BATCH_SIZE,
    shuffle           = True,
    num_workers       = DL_WORKERS,
    pin_memory        = True,
    persistent_workers= True,              # keep workers alive between epochs
    prefetch_factor   = 2,                 # each worker pre-fetches 2 batches ahead
)
log(f"DataLoader: {DL_WORKERS} workers, prefetch=2, persistent — augmentation fully parallelised")

# ── Model + optimiser ─────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1]
model = WaveNet1D(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
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

    for ib, xb, yb in loader:
        ib, xb, yb = ib.to(device, non_blocking=True), \
                     xb.to(device, non_blocking=True), \
                     yb.to(device, non_blocking=True)
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

# Full training set predictions (clean, for meta-learner compatibility)
i_tr_all_n = norm_traces(i_train_raw)
i_tr_all_t = torch.tensor(i_tr_all_n)
X_tr_all_t = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(
        lambda i, x: model(i, x), i_tr_all_t, X_tr_all_t, bs=EVAL_BATCH
    ).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)
log("Saved predictions")

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"dt=0.1ms  N_BINS={N_BINS}  tMax=4096ms\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}\n"
    f"  Receptive field: {rf_original} original lags\n"
    f"  AdaptiveAvgPool1d handles {N_BINS}-pt input\n"
    f"Params: {n_params:,}\n"
    f"Augmentation: scale U[{AUG_SCALE_LO},{AUG_SCALE_HI}] + noise sd={AUG_NOISE_SD} "
    f"+ shift ±{AUG_SHIFT} (online, {DL_WORKERS} DataLoader workers)\n"
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
