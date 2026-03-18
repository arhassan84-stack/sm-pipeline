"""
Multi-dt WaveNet — Option F: 3 branches with cross-branch attention.

Same 3 WaveNet backbones as multidt_B, but adds a CrossBranchAttention module
between the branch embeddings and the head. The attention allows each branch
embedding to attend to the other two, capturing joint cross-dt representations
(e.g. the ratio of ACF amplitudes across timescales encodes D directly).

Architecture:
  dt=0.10ms (40960) → AvgPool(16) → WaveNetBackbone(256) → e010 (256-dim)  [stream s010]
  dt=0.50ms  (8192) →   native   → WaveNetBackbone(256) → e050 (256-dim)  [stream s050]
  dt=1.00ms  (4096) →   native   → WaveNetBackbone(256) → e100 (256-dim)  [main stream]
                                                              ↓
  [e010, e050, e100] → CrossBranchAttention(d=256, heads=4) → 768-dim
  906-dim features   → feat_branch(906→256→128)             → 128-dim
                                                              ↓
  concat [768 + 128 = 896] → Linear(896,512) → ReLU → Linear(512,128) → ReLU → Linear(128,1)

Label: wavenet_multidt_F
"""

import numpy as np, time, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'wavenet_multidt_F2'
CHANNELS     = 256
BATCH_SIZE   = 128
DL_WORKERS   = 6
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03
AUG_SHIFT    = 128
EVAL_BATCH   = 128
POOL010      = 16   # AvgPool factor for dt010 branch: 40960 → 2560
ATTN_HEADS   = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


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


def batch_predict(model_fn, *cpu_tensors, bs=128):
    n = len(cpu_tensors[0]); outs = []
    with torch.no_grad(), autocast():
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


class WaveNetBackbone(nn.Module):
    def __init__(self, channels=256, dilations=None, avgpool=1):
        super().__init__()
        if dilations is None: dilations = [1,2,4,8,16,32,64,128]
        proj_layers = []
        if avgpool > 1:
            proj_layers.append(nn.AvgPool1d(kernel_size=avgpool, stride=avgpool))
        proj_layers += [
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        ]
        self.input_proj = nn.Sequential(*proj_layers)
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap    = nn.AdaptiveAvgPool1d(1)

    def forward(self, trace):
        x = self.input_proj(trace.unsqueeze(1))
        x = self.blocks(x)
        return self.gap(x).squeeze(2)   # (B, channels)


class CrossBranchAttention(nn.Module):
    """Self-attention over the 3 branch embeddings.
    Treats [e010, e050, e100] as a length-3 sequence, allowing each branch
    to attend to the others before being passed to the head.
    """
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                              batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e010, e050, e100):
        x = torch.stack([e010, e050, e100], dim=1)   # (B, 3, d_model)
        # Cast to FP32 to avoid AMP FP16 overflow in softmax
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            attn_out, _ = self.attn(x.float(), x.float(), x.float())
        attn_out = attn_out.to(x.dtype)
        x = self.norm(x + self.dropout(attn_out))    # residual + LN
        return x.flatten(1)                          # (B, 3*d_model)


class WaveNetMultiDT_F(nn.Module):
    """3 WaveNet branches + cross-branch attention before head."""
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        self.branch010 = WaveNetBackbone(channels, dilations, avgpool=POOL010)
        self.branch050 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.cross_attn = CrossBranchAttention(d_model=channels, nhead=ATTN_HEADS)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # 3*256 + 128 = 896
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.s010 = torch.cuda.Stream()
        self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        cur = torch.cuda.current_stream()
        self.s010.wait_stream(cur)
        self.s050.wait_stream(cur)

        with torch.cuda.stream(self.s010):
            e010 = self.branch010(t010)
        with torch.cuda.stream(self.s050):
            e050 = self.branch050(t050)
        e100 = self.branch100(t100)
        f    = self.feat_branch(feats)

        cur.wait_stream(self.s010)
        cur.wait_stream(self.s050)

        fused = self.cross_attn(e010, e050, e100)   # (B, 768)
        return self.head(torch.cat([fused, f], dim=1)).squeeze(1)


# ── Data loading ──────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log(f"Speedups: AvgPool{POOL010} on dt010 | CUDA streams | AMP | BATCH={BATCH_SIZE}")
log("Loading multi-dt caches...")

i010_tr_mmap = np.load('cache_i_train_multidt_dt010.npy').astype(np.float32)
i050_tr = np.load('cache_i_train_multidt_dt050.npy').astype(np.float32)
i100_tr = np.load('cache_i_train_multidt_dt100.npy').astype(np.float32)
X010_tr = np.load('cache_X_train_multidt_dt010.npy').astype(np.float32)
X050_tr = np.load('cache_X_train_multidt_dt050.npy').astype(np.float32)
X100_tr = np.load('cache_X_train_multidt_dt100.npy').astype(np.float32)
d_train = np.load('cache_d_train_multidt.npy')

i010_te_mmap = np.load('cache_i_test_multidt_dt010.npy').astype(np.float32)
i050_te = np.load('cache_i_test_multidt_dt050.npy').astype(np.float32)
i100_te = np.load('cache_i_test_multidt_dt100.npy').astype(np.float32)
X010_te = np.load('cache_X_test_multidt_dt010.npy').astype(np.float32)
X050_te = np.load('cache_X_test_multidt_dt050.npy').astype(np.float32)
X100_te = np.load('cache_X_test_multidt_dt100.npy').astype(np.float32)
d_test  = np.load('cache_d_test_multidt.npy')

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
                    persistent_workers=True, prefetch_factor=2)

# ── Model + optimiser ─────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1]
model  = WaveNetMultiDT_F(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit   = nn.MSELoss()
scaler = GradScaler()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  (3 branches + cross-attention + CUDA streams + AMP)")
log(f"  dt010: AvgPool{POOL010}→{40960//POOL010} bins | dt050: 8192 | dt100: 4096")
log(f"  CrossBranchAttention: d={CHANNELS} heads={ATTN_HEADS}  head_in={CHANNELS*3+128}")

# ── Training loop ─────────────────────────────────────────────────────────────

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
        scaler.step(opt)
        scaler.update()
        ep += loss.item() * len(yb)
    ep /= len(tr_idx)

    model.eval()
    with torch.no_grad():
        vl = crit(
            batch_predict(lambda a, b, c, x: model(a, b, c, x),
                          torch.tensor(i010_val_n), torch.tensor(i050_val_n),
                          torch.tensor(i100_val_n), X_val, bs=EVAL_BATCH),
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

def batch_predict_F(i010_mmap, i010_idx, i050, i100, X_np, bs=128):
    n = len(i010_idx); outs = []
    X_t    = torch.tensor(X_np)
    i050_t = torch.tensor(norm_traces(i050))
    i100_t = torch.tensor(norm_traces(i100))
    with torch.no_grad(), autocast():
        for s in range(0, n, bs):
            chunk = np.array(i010_mmap[i010_idx[s:s+bs]], dtype=np.float32)
            mu  = chunk.mean(axis=1, keepdims=True)
            sig = chunk.std(axis=1,  keepdims=True) + 1e-8
            t010 = torch.tensor((chunk - mu) / sig).to(device)
            outs.append(model(t010,
                              i050_t[s:s+bs].to(device),
                              i100_t[s:s+bs].to(device),
                              X_t[s:s+bs].to(device)).cpu())
    return torch.cat(outs)

model.eval()
log("Predicting test set...")
logd_pred_te = batch_predict_F(i010_te_mmap, te_global, i050_te, i100_te, X_te_sc).numpy()
log("Predicting train set...")
logd_pred_tr = batch_predict_F(i010_tr_mmap, tr_global, i050_tr, i100_tr, X_tr_sc).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

d_pred_te = np.exp(logd_pred_te); d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total
ms, mf = d_test < 1.0, d_test >= 1.0

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 3 WaveNet branches + CrossBranchAttention "
    f"(dt010 AvgPool{POOL010}→{40960//POOL010}bins, dt050 native, dt100) + {feat_dim}-dim features\n"
    f"  WaveNetMultiDT_F channels={CHANNELS} dilations={DILATIONS}\n"
    f"  CrossBranchAttention: d_model={CHANNELS} nhead={ATTN_HEADS}\n"
    f"  Speedups: AvgPool{POOL010} on dt010 | CUDA streams | AMP\n"
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
