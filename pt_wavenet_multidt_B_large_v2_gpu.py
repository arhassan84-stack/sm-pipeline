"""
Multi-dt WaveNet B_large_v2 — architectural improvements over B_large:

  1. Attention pooling (replaces GAP): per-position learned weights so the
     model focuses on the decay region rather than averaging uniformly.
  2. Channels 256 → 512: more capacity for the 771k-sample large dataset.
  3. Deeper dilations [1,2,4,8,16,32,64,128,256,512]: larger receptive field
     (~2× vs B_large), better context for resolving fast-decaying ACFs.
  4. AMP (autocast + GradScaler): keeps epoch time comparable to B_large.
  5. CUDA streams: dt010 and dt050 branches computed in parallel.

Same data: cache_*_multidt_large_*.npy  (771k train / 85k test, d<=10)
Label: wavenet_multidt_B_large_v2
"""

import numpy as np, time, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'wavenet_multidt_B_large_v2'
CHANNELS     = 512
BATCH_SIZE   = 128
DL_WORKERS   = 4
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03
AUG_SHIFT    = 128
EVAL_BATCH   = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU — aborting. Use --gres=gpu:... in SLURM.")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── Dataset (unchanged from B_large) ──────────────────────────────────────────

class AugFCSDatasetMultiDT(Dataset):
    def __init__(self, i010_mmap, i010_idx, i050, i100, X_tensor, y_tensor):
        self.i010_mmap = i010_mmap; self.i010_idx = i010_idx
        self.i050 = i050; self.i100 = i100
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
        t010 = np.array(self.i010_mmap[self.i010_idx[idx]], dtype=np.float32)
        t050 = self.i050[idx]; t100 = self.i100[idx]
        scale = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        s100  = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))

        def aug(t, shift):
            noise = self.rng.normal(0.0, AUG_NOISE_SD, size=t.shape).astype(np.float32)
            a = t * scale + noise
            a = np.roll(a, shift)
            return (a - a.mean()) / (a.std() + 1e-8)

        a010 = aug(t010, s100 * 10)
        a050 = aug(t050, s100 * 2)
        a100 = aug(t100, s100)
        return (torch.from_numpy(a010.astype(np.float32)),
                torch.from_numpy(a050.astype(np.float32)),
                torch.from_numpy(a100.astype(np.float32)),
                self.X[idx], self.y[idx])


def norm_traces(traces):
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return ((traces - mu) / sig).astype(np.float32)


# ── Model ──────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x): return self.act(self.bn(self.conv(x)) + x)


class AttentionPool1d(nn.Module):
    """Learned per-position scalar attention weights; replaces AdaptiveAvgPool1d.

    Computes a softmax over the time axis so the model can focus on the
    correlation-decay region rather than averaging over all positions uniformly.
    """
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)   # (B, 1, L)

    def forward(self, x):                                     # x: (B, C, L)
        w = torch.softmax(self.score(x), dim=2)              # (B, 1, L)
        return (x * w).sum(dim=2)                            # (B, C)


class WaveNetBackbone(nn.Module):
    """WaveNet backbone with attention pooling instead of GAP."""
    def __init__(self, channels=512, dilations=None):
        super().__init__()
        if dilations is None: dilations = DILATIONS
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks    = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.attn_pool = AttentionPool1d(channels)

    def forward(self, trace):          # trace: (B, N)
        x = self.input_proj(trace.unsqueeze(1))
        x = self.blocks(x)
        return self.attn_pool(x)       # (B, channels)


class WaveNetMultiDT_B_v2(nn.Module):
    """3 independent WaveNet backbones with attention pooling + larger capacity."""
    def __init__(self, feat_dim, channels=512, dilations=None):
        super().__init__()
        self.branch010 = WaveNetBackbone(channels, dilations)
        self.branch050 = WaveNetBackbone(channels, dilations)
        self.branch100 = WaveNetBackbone(channels, dilations)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # head input: channels*3 + 128 = 1664
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 128, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        # CUDA streams for parallel branch computation
        self.s010 = torch.cuda.Stream()
        self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        cur = torch.cuda.current_stream()
        self.s010.wait_stream(cur)
        self.s050.wait_stream(cur)
        with torch.cuda.stream(self.s010): e010 = self.branch010(t010)
        with torch.cuda.stream(self.s050): e050 = self.branch050(t050)
        e100 = self.branch100(t100)
        f    = self.feat_branch(feats)
        cur.wait_stream(self.s010)
        cur.wait_stream(self.s050)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


# ── Data loading (identical to B_large) ───────────────────────────────────────

t_total = time.time()
log(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
log("Loading dt010 train cache into RAM (~117 GB)...")
i010_tr_mmap = np.load('cache_i_train_multidt_large_dt010.npy')
i050_tr = np.load('cache_i_train_multidt_large_dt050.npy').astype(np.float32)
i100_tr = np.load('cache_i_train_multidt_large_dt100.npy').astype(np.float32)
X010_tr = np.load('cache_X_train_multidt_large_dt010.npy').astype(np.float32)
X050_tr = np.load('cache_X_train_multidt_large_dt050.npy').astype(np.float32)
X100_tr = np.load('cache_X_train_multidt_large_dt100.npy').astype(np.float32)
d_train = np.load('cache_d_train_multidt_large.npy')

i010_te_mmap = np.load('cache_i_test_multidt_large_dt010.npy', mmap_mode='r')
i050_te = np.load('cache_i_test_multidt_large_dt050.npy').astype(np.float32)
i100_te = np.load('cache_i_test_multidt_large_dt100.npy').astype(np.float32)
X010_te = np.load('cache_X_test_multidt_large_dt010.npy').astype(np.float32)
X050_te = np.load('cache_X_test_multidt_large_dt050.npy').astype(np.float32)
X100_te = np.load('cache_X_test_multidt_large_dt100.npy').astype(np.float32)
d_test  = np.load('cache_d_test_multidt_large.npy')

X_train = np.concatenate([X010_tr, X050_tr, X100_tr], axis=1)
X_test  = np.concatenate([X010_te, X050_te, X100_te], axis=1)

mask_tr = d_train <= 10
tr_global = np.where(mask_tr)[0]
i050_tr, i100_tr = i050_tr[mask_tr], i100_tr[mask_tr]
X_train, d_train = X_train[mask_tr], d_train[mask_tr]
mask_te = d_test <= 10
te_global = np.where(mask_te)[0]
i050_te, i100_te = i050_te[mask_te], i100_te[mask_te]
X_test, d_test = X_test[mask_te], d_test[mask_te]

log(f"Train: {len(d_train)}  Test: {len(d_test)}")

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

i010_val_n = norm_traces(np.array(i010_tr_mmap[tr_global[val_idx]], dtype=np.float32))
i050_val_n = norm_traces(i050_tr[val_idx])
i100_val_n = norm_traces(i100_tr[val_idx])
X_val  = torch.tensor(X_tr_sc[val_idx])
y_val  = torch.tensor(log_d[val_idx])

train_ds = AugFCSDatasetMultiDT(
    i010_mmap=i010_tr_mmap, i010_idx=tr_global[tr_idx],
    i050=i050_tr[tr_idx], i100=i100_tr[tr_idx],
    X_tensor=torch.tensor(X_tr_sc[tr_idx]),
    y_tensor=torch.tensor(log_d[tr_idx]),
)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=DL_WORKERS, pin_memory=True,
                    persistent_workers=False, prefetch_factor=2)

# ── Model + optimiser ─────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1]
model  = WaveNetMultiDT_B_v2(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit   = nn.MSELoss()
scaler = GradScaler()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")
log(f"  channels={CHANNELS}  dilations(depth={len(DILATIONS)})={DILATIONS}")
log(f"  AttentionPool1d  AMP  CUDA streams")

# ── Training loop ─────────────────────────────────────────────────────────────

i010_val_t = torch.tensor(i010_val_n)
i050_val_t = torch.tensor(i050_val_n)
i100_val_t = torch.tensor(i100_val_n)

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for t010b, t050b, t100b, xb, yb in loader:
        t010b = t010b.to(device, non_blocking=True)
        t050b = t050b.to(device, non_blocking=True)
        t100b = t100b.to(device, non_blocking=True)
        xb    = xb.to(device, non_blocking=True)
        yb    = yb.to(device, non_blocking=True)
        opt.zero_grad()
        with autocast():
            loss = crit(model(t010b, t050b, t100b, xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()
        ep += loss.item() * len(yb)
    ep /= len(tr_idx)

    model.eval()
    preds_val = []
    with torch.no_grad(), autocast():
        for i in range(0, len(val_idx), EVAL_BATCH):
            preds_val.append(model(
                i010_val_t[i:i + EVAL_BATCH].to(device),
                i050_val_t[i:i + EVAL_BATCH].to(device),
                i100_val_t[i:i + EVAL_BATCH].to(device),
                X_val[i:i + EVAL_BATCH].to(device)).cpu())
    vl = crit(torch.cat(preds_val), y_val).item()

    sched.step(epoch - 1)
    if vl < best_val - 1e-6:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

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


def batch_predict_B(i010_mmap, i010_idx, i050, i100, X_np, bs=128):
    n      = len(i010_idx); outs = []
    X_t    = torch.tensor(X_np)
    i050_t = torch.tensor(norm_traces(i050))
    i100_t = torch.tensor(norm_traces(i100))
    with torch.no_grad(), autocast():
        for s in range(0, n, bs):
            chunk = np.array(i010_mmap[i010_idx[s:s + bs]], dtype=np.float32)
            mu  = chunk.mean(axis=1, keepdims=True)
            sig = chunk.std(axis=1,  keepdims=True) + 1e-8
            t010 = torch.tensor((chunk - mu) / sig).to(device)
            outs.append(model(t010,
                              i050_t[s:s + bs].to(device),
                              i100_t[s:s + bs].to(device),
                              X_t[s:s + bs].to(device)).cpu())
    return torch.cat(outs)


model.eval()
log("Predicting test set...")
logd_pred_te = batch_predict_B(i010_te_mmap, te_global, i050_te, i100_te, X_te_sc).numpy()
log("Predicting train set...")
logd_pred_tr = batch_predict_B(i010_tr_mmap, tr_global, i050_tr, i100_tr, X_tr_sc).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

d_pred_te = np.exp(logd_pred_te); d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total
ms, mf    = d_test < 1.0, d_test >= 1.0

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 3 WaveNet branches + AttentionPool1d  channels={CHANNELS}"
    f"  dilations(depth={len(DILATIONS)})={DILATIONS}\n"
    f"  feat_dim={feat_dim}  head: {CHANNELS*3+128}→768→256→1  AMP  CUDA streams\n"
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
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
