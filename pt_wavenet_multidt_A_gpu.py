"""
Multi-dt WaveNet — Option A: Multi-channel (3 channels × 4096 bins).

dt=0.1ms (40960 bins) and dt=0.5ms (8192 bins) are downsampled to 4096 bins
via AvgPool, then stacked with dt=1.0ms as a (3, 4096) multi-channel input.
input_proj changed from Conv1d(1,...) to Conv1d(3,...).

Features: 302-dim from each dt concatenated → 906-dim feat vector.

Label: wavenet_multidt_A
"""

import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'wavenet_multidt_A'
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
AUG_SHIFT    = 128      # bins in dt100 space; scaled ×10 / ×2 for dt010/dt050
EVAL_BATCH   = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


class AugFCSDatasetA(Dataset):
    """Augments all 3 traces consistently; downsamples to 4096 and stacks as (3,4096)."""
    def __init__(self, i010, i050, i100, X_tensor, y_tensor):
        self.i010 = i010; self.i050 = i050; self.i100 = i100
        self.X = X_tensor; self.y = y_tensor
        self._rng = None

    def __len__(self): return len(self.y)

    @property
    def rng(self):
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            seed = int(wi.seed % (2**31)) if wi is not None else 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, idx):
        t010 = self.i010[idx]; t050 = self.i050[idx]; t100 = self.i100[idx]
        scale = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        s100  = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))

        def aug(t, shift):
            noise = self.rng.normal(0.0, AUG_NOISE_SD, size=t.shape).astype(np.float32)
            a = t * scale + noise
            a = np.roll(a, shift)
            return (a - a.mean()) / (a.std() + 1e-8)

        a010 = aug(t010, s100 * 10)   # same fractional shift
        a050 = aug(t050, s100 * 2)
        a100 = aug(t100, s100)

        # Downsample to 4096: dt010 ÷10, dt050 ÷2
        a010_ds = a010.reshape(4096, 10).mean(axis=1).astype(np.float32)
        a050_ds = a050.reshape(4096,  2).mean(axis=1).astype(np.float32)
        stacked = np.stack([a010_ds, a050_ds, a100.astype(np.float32)], axis=0)  # (3,4096)
        return torch.from_numpy(stacked), self.X[idx], self.y[idx]


def norm_and_stack(i010, i050, i100):
    """Normalize each trace independently, downsample, stack → (N,3,4096)."""
    def nz(arr):
        mu  = arr.mean(axis=1, keepdims=True)
        sig = arr.std(axis=1,  keepdims=True) + 1e-8
        return (arr - mu) / sig
    a010 = nz(i010).reshape(len(i010), 4096, 10).mean(axis=2)
    a050 = nz(i050).reshape(len(i050), 4096,  2).mean(axis=2)
    a100 = nz(i100)
    return np.stack([a010, a050, a100], axis=1).astype(np.float32)  # (N,3,4096)


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
    def forward(self, x): return self.act(self.bn(self.conv(x)) + x)


class WaveNetMultiDT_A(nn.Module):
    """3-channel input (dt010_ds, dt050_ds, dt100) each at 4096 bins."""
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None: dilations = [1,2,4,8,16,32,64,128]
        self.input_proj = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=16, stride=4, padding=6),
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
    def forward(self, traces_3ch, feats):  # traces_3ch: (B,3,4096)
        x = self.input_proj(traces_3ch)
        x = self.blocks(x)
        x = self.gap(x).squeeze(2)
        f = self.feat_branch(feats)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


# ── Data loading ──────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log(f"Loading multi-dt caches (tag=multidt_dt{{010,050,100}})...")

i010_tr = np.load('cache_i_train_multidt_dt010.npy').astype(np.float32)
i050_tr = np.load('cache_i_train_multidt_dt050.npy').astype(np.float32)
i100_tr = np.load('cache_i_train_multidt_dt100.npy').astype(np.float32)
X010_tr = np.load('cache_X_train_multidt_dt010.npy').astype(np.float32)
X050_tr = np.load('cache_X_train_multidt_dt050.npy').astype(np.float32)
X100_tr = np.load('cache_X_train_multidt_dt100.npy').astype(np.float32)
d_train = np.load('cache_d_train_multidt.npy')

i010_te = np.load('cache_i_test_multidt_dt010.npy').astype(np.float32)
i050_te = np.load('cache_i_test_multidt_dt050.npy').astype(np.float32)
i100_te = np.load('cache_i_test_multidt_dt100.npy').astype(np.float32)
X010_te = np.load('cache_X_test_multidt_dt010.npy').astype(np.float32)
X050_te = np.load('cache_X_test_multidt_dt050.npy').astype(np.float32)
X100_te = np.load('cache_X_test_multidt_dt100.npy').astype(np.float32)
d_test  = np.load('cache_d_test_multidt.npy')

# Concatenate features from all 3 dt values (906-dim)
X_train = np.concatenate([X010_tr, X050_tr, X100_tr], axis=1)
X_test  = np.concatenate([X010_te, X050_te, X100_te], axis=1)

mask_tr = d_train <= 10
i010_tr, i050_tr, i100_tr = i010_tr[mask_tr], i050_tr[mask_tr], i100_tr[mask_tr]
X_train, d_train = X_train[mask_tr], d_train[mask_tr]

mask_te = d_test <= 10
i010_te, i050_te, i100_te = i010_te[mask_te], i050_te[mask_te], i100_te[mask_te]
X_test, d_test = X_test[mask_te], d_test[mask_te]

log(f"Train: {len(d_train)}  Test: {len(d_test)}")

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

# Val: normalize and stack
i_val_stacked = norm_and_stack(i010_tr[val_idx], i050_tr[val_idx], i100_tr[val_idx])
X_val = torch.tensor(X_tr_sc[val_idx]); y_val = torch.tensor(log_d[val_idx])

train_ds = AugFCSDatasetA(
    i010=i010_tr[tr_idx], i050=i050_tr[tr_idx], i100=i100_tr[tr_idx],
    X_tensor=torch.tensor(X_tr_sc[tr_idx]),
    y_tensor=torch.tensor(log_d[tr_idx]),
)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=DL_WORKERS, pin_memory=True,
                    persistent_workers=True, prefetch_factor=2)

# ── Model + optimiser ─────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1]
model = WaveNetMultiDT_A(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")

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
        loss.backward(); opt.step()
        ep += loss.item() * len(ib)
    ep /= len(tr_idx)

    model.eval()
    i_val_t = torch.tensor(i_val_stacked)
    with torch.no_grad():
        vl = crit(
            batch_predict(lambda i, x: model(i, x), i_val_t, X_val, bs=EVAL_BATCH),
            y_val).item()

    sched.step(epoch - 1)
    if vl < best_val - 1e-6:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.5f}")

# ── Save + evaluate ───────────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)

model.eval()
i_te_stacked  = norm_and_stack(i010_te, i050_te, i100_te)
i_tr_stacked  = norm_and_stack(i010_tr, i050_tr, i100_tr)
with torch.no_grad():
    logd_pred_te = batch_predict(lambda i, x: model(i, x),
                                 torch.tensor(i_te_stacked), torch.tensor(X_te_sc),
                                 bs=EVAL_BATCH).numpy()
    logd_pred_tr = batch_predict(lambda i, x: model(i, x),
                                 torch.tensor(i_tr_stacked), torch.tensor(X_tr_sc),
                                 bs=EVAL_BATCH).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

d_pred_te = np.exp(logd_pred_te); d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total
ms, mf = d_test < 1.0, d_test >= 1.0

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 3-channel (dt010↓4096, dt050↓4096, dt100) + 906-dim features\n"
    f"  WaveNetMultiDT_A channels={CHANNELS} dilations={DILATIONS}\n"
    f"Params: {n_params:,}\n"
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
with open(f'results_{LABEL}.txt', 'w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
