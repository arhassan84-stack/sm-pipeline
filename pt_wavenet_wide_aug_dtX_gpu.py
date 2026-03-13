"""
WaveNet wide + augmentation — parameterised for any dt value.

Usage:
  python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag dt020 --dt 0.2
  python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag dt040 --dt 0.4
  python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag dt060 --dt 0.6
  python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag dt080 --dt 0.8

Architecture identical to pt_wavenet_wide_aug_gpu.py (WaveNet1D channels=256).
AdaptiveAvgPool1d(1) makes the trace branch length-agnostic.

Parameters derived automatically from --dt:
  N_BINS     = round(4096 / dt_ms)
  AUG_SHIFT  = round(N_BINS / 32)         (same proportion as ±128 on dt=1ms)
  EVAL_BATCH = 256 if N_BINS<=8192 else 128 if N_BINS<=16384 else 64

Augmentation runs in DataLoader workers (num_workers=4), fully overlapping
with GPU forward/backward pass.
"""

import argparse, numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dt_tag', required=True, help='e.g. dt020, dt040, dt060, dt080')
parser.add_argument('--dt',     required=True, type=float, help='dt in ms, e.g. 0.2')
args = parser.parse_args()

DT_TAG  = args.dt_tag
DT_MS   = args.dt
T_MAX   = 4096                              # ms (fixed)
N_BINS  = round(T_MAX / DT_MS)

# Derived augmentation / batch parameters
AUG_SHIFT  = round(N_BINS / 32)            # proportional to ±128/4096 on dt=1ms
EVAL_BATCH = 256 if N_BINS <= 8192 else (128 if N_BINS <= 16384 else 64)

LABEL        = f'pt_wavenet_wide_aug_{DT_TAG}'
CHANNELS     = 256
BATCH_SIZE   = 128
DL_WORKERS   = 4
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── Dataset with worker-parallel augmentation ─────────────────────────────────

class AugFCSDataset(Dataset):
    def __init__(self, i_raw, X_tensor, y_tensor):
        self.i_raw = i_raw
        self.X     = X_tensor
        self.y     = y_tensor
        self._rng  = None

    def __len__(self):
        return len(self.y)

    @property
    def rng(self):
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            seed = int(wi.seed % (2**31)) if wi is not None else 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, idx):
        trace = self.i_raw[idx]
        scale = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        noise = self.rng.normal(0.0, AUG_NOISE_SD, size=trace.shape).astype(np.float32)
        shift = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))
        aug = trace * scale + noise
        aug = np.roll(aug, shift)
        aug = (aug - aug.mean()) / (aug.std() + 1e-8)
        return torch.from_numpy(aug.astype(np.float32)), self.X[idx], self.y[idx]


def norm_traces(traces):
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return (traces - mu) / sig


def batch_predict(model_fn, *cpu_tensors, bs=256):
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
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
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
log(f"dt={DT_MS}ms  N_BINS={N_BINS}  AUG_SHIFT=±{AUG_SHIFT}  EVAL_BATCH={EVAL_BATCH}")
log(f"Loading cache_{'{i,X,d}'}_{'{train,test}'}_{DT_TAG}.npy ...")

i_train_raw = np.load(f'cache_i_train_{DT_TAG}.npy').astype(np.float32)
X_train     = np.load(f'cache_X_train_{DT_TAG}.npy').astype(np.float32)
d_train     = np.load(f'cache_d_train_{DT_TAG}.npy')
i_test_raw  = np.load(f'cache_i_test_{DT_TAG}.npy').astype(np.float32)
X_test      = np.load(f'cache_X_test_{DT_TAG}.npy').astype(np.float32)
d_test      = np.load(f'cache_d_test_{DT_TAG}.npy')

mask_tr = d_train <= 10
i_train_raw, X_train, d_train = i_train_raw[mask_tr], X_train[mask_tr], d_train[mask_tr]
mask_te = d_test <= 10
i_test_raw,  X_test,  d_test  = i_test_raw[mask_te],  X_test[mask_te],  d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Bins/trace: {i_train_raw.shape[1]}")

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_val_n = norm_traces(i_train_raw[val_idx])
X_val   = torch.tensor(X_tr_sc[val_idx])
y_val   = torch.tensor(log_d[val_idx])

train_ds = AugFCSDataset(
    i_raw    = i_train_raw[tr_idx],
    X_tensor = torch.tensor(X_tr_sc[tr_idx]),
    y_tensor = torch.tensor(log_d[tr_idx]),
)
loader = DataLoader(
    train_ds,
    batch_size         = BATCH_SIZE,
    shuffle            = True,
    num_workers        = DL_WORKERS,
    pin_memory         = True,
    persistent_workers = True,
    prefetch_factor    = 2,
)

# ── Model + optimiser ─────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1]
model = WaveNet1D(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params    = sum(p.numel() for p in model.parameters())
rf_original = sum(2 * d for d in DILATIONS) * 4
log(f"Training {LABEL}  params={n_params:,}  DL_WORKERS={DL_WORKERS}")
log(f"  channels={CHANNELS}  dilations={DILATIONS}  RF={rf_original}")

# ── Training loop ─────────────────────────────────────────────────────────────

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for ib, xb, yb in loader:
        ib = ib.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad()
        loss = crit(model(ib, xb), yb)
        loss.backward()
        opt.step()
        ep += loss.item() * len(ib)
    ep /= len(tr_idx)

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

# ── Save ──────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)

def batch_predict_raw(raw_np, X_np_sc, bs=256):
    """Normalize + predict in chunks — avoids materializing full normalized array."""
    n = len(raw_np); outs = []
    X_t = torch.tensor(X_np_sc)
    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = raw_np[i:i+bs]
            mu  = chunk.mean(axis=1, keepdims=True)
            sig = chunk.std(axis=1,  keepdims=True) + 1e-8
            i_n = torch.tensor((chunk - mu) / sig).to(device)
            outs.append(model(i_n, X_t[i:i+bs].to(device)).cpu())
    return torch.cat(outs)

model.eval()
logd_pred_te = batch_predict_raw(i_test_raw,  X_te_sc, bs=EVAL_BATCH).numpy()
logd_pred_tr = batch_predict_raw(i_train_raw, X_tr_sc, bs=EVAL_BATCH).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

d_pred_te = np.exp(logd_pred_te)
d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total

ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"dt={DT_MS}ms  N_BINS={N_BINS}  tMax={T_MAX}ms\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}\n"
    f"  RF={rf_original}  AdaptiveAvgPool1d handles {N_BINS}-pt input\n"
    f"Params: {n_params:,}\n"
    f"Augmentation: scale U[{AUG_SCALE_LO},{AUG_SCALE_HI}] + noise sd={AUG_NOISE_SD} "
    f"+ shift ±{AUG_SHIFT} ({DL_WORKERS} DataLoader workers)\n"
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
