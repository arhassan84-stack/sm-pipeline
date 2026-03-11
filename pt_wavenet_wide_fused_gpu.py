"""
WaveNet1D(channels=256) trained on fused representations.

Each sample is an expanded view of a single physical trace:
  - raw trace       (4096 pts)  normalized independently
  - smoothed trace  (4096 pts)  normalized independently
  → concatenated along time → (8192,) input to WaveNet

  - raw features    (302-dim)
  - smooth features (302-dim)
  → concatenated → (604,) feature input

Training: 390k samples (same count as baseline, not doubled).
Test:     same preparation — raw test trace + smooth{W} test trace,
          raw test features + smooth{W} test features.
          Directly comparable to wavenet_wide_aug baseline (MAPE=12.34%).

Saves:
  model_wavenet_wide_fused{W}.pt
  scaler_wavenet_wide_fused{W}_mean/scale.npy  (604-dim scaler)
  pred_logd_wavenet_wide_fused{W}_test.npy
  results_pt_wavenet_wide_fused{W}.txt

Usage:
  python pt_wavenet_wide_fused_gpu.py --window 10
"""
import argparse
import numpy as np, time, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--window', type=int, required=True,
                    help='Smoothing window used in fused cache (e.g. 10)')
args = parser.parse_args()
W = args.window

LABEL        = f'pt_wavenet_wide_fused{W}'
CHANNELS     = 256
BATCH_SIZE   = 128
EVAL_BATCH   = 256
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

def batch_predict(model_fn, *cpu_tensors, bs=256):
    n=len(cpu_tensors[0]); outs=[]
    for i in range(0, n, bs):
        batch=[t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)


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
    """Identical architecture to wavenet_wide_aug except feat_dim=604."""
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        # Trace branch: handles any input length (AdaptiveAvgPool compresses to 256-d)
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Feature branch: 604-dim input (302 raw + 302 smooth)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Fusion head: channels(256) + feat(128) = 384
        self.head = nn.Sequential(
            nn.Linear(channels + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        x = self.input_proj(trace.unsqueeze(1))   # (B, channels, L/4)
        x = self.blocks(x)                         # (B, channels, L/4)
        x = self.gap(x).squeeze(2)                 # (B, channels)
        f = self.feat_branch(feats)                # (B, 128)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


t_total = time.time()
log(f"Device: {device}  Fused window: W={W}")

# ── Load training data ────────────────────────────────────────────────────────
log("Loading raw training cache...")
i_raw_tr = np.load('cache_i_train_aug.npy').astype(np.float32)      # (N, 4096)
d_train  = np.load('cache_d_train_aug.npy')                          # (N,)

log(f"Loading smooth W={W} training cache...")
i_sm_tr  = np.load(f'cache_i_train_smooth{W}.npy').astype(np.float32)  # (N, 4096)

log(f"Loading fused feature cache (train)...")
X_fused_tr = np.load(f'cache_X_train_fused{W}.npy').astype(np.float32)  # (N, 604)
log(f"  Train: {i_raw_tr.shape} raw + {i_sm_tr.shape} smooth  feats: {X_fused_tr.shape}")

# ── Load test data (same preparation as training) ─────────────────────────────
log("Loading raw test cache...")
i_raw_te = np.load('cache_i_test_90pct.npy').astype(np.float32)     # (M, 4096)
d_test   = np.load('cache_d_test_90pct.npy')                         # (M,)

log(f"Loading smooth W={W} test cache...")
i_sm_te  = np.load(f'cache_i_test_smooth{W}.npy').astype(np.float32)   # (M, 4096)

log(f"Loading fused feature cache (test)...")
X_fused_te = np.load(f'cache_X_test_fused{W}.npy').astype(np.float32)  # (M, 604)
log(f"  Test: {i_raw_te.shape} raw + {i_sm_te.shape} smooth  feats: {X_fused_te.shape}")

# ── Apply d<=10 mask ──────────────────────────────────────────────────────────
mask_tr = d_train <= 10
i_raw_tr, i_sm_tr, X_fused_tr, d_train = (i_raw_tr[mask_tr], i_sm_tr[mask_tr],
                                            X_fused_tr[mask_tr], d_train[mask_tr])
mask_te = d_test <= 10
i_raw_te, i_sm_te, X_fused_te, d_test  = (i_raw_te[mask_te], i_sm_te[mask_te],
                                            X_fused_te[mask_te], d_test[mask_te])
log(f"After d<=10 mask — Train: {len(d_train)}  Test: {len(d_test)}")

# ── Normalize traces (per-trace z-score, each set independently) ──────────────
log("Normalising traces...")
i_raw_tr_n = (i_raw_tr - i_raw_tr.mean(1, keepdims=True)) / (i_raw_tr.std(1, keepdims=True) + 1e-8)
del i_raw_tr
i_sm_tr_n  = (i_sm_tr  - i_sm_tr.mean(1,  keepdims=True)) / (i_sm_tr.std(1,  keepdims=True) + 1e-8)
del i_sm_tr

# Concatenate along time axis: (N, 4096 raw + 4096 smooth) = (N, 8192)
i_train_n = np.hstack([i_raw_tr_n, i_sm_tr_n])
del i_raw_tr_n, i_sm_tr_n
log(f"  Fused trace shape: {i_train_n.shape}")

i_raw_te_n = (i_raw_te - i_raw_te.mean(1, keepdims=True)) / (i_raw_te.std(1, keepdims=True) + 1e-8)
del i_raw_te
i_sm_te_n  = (i_sm_te  - i_sm_te.mean(1,  keepdims=True)) / (i_sm_te.std(1,  keepdims=True) + 1e-8)
del i_sm_te

i_test_n = np.hstack([i_raw_te_n, i_sm_te_n])
del i_raw_te_n, i_sm_te_n
log(f"  Test fused trace shape: {i_test_n.shape}")

# ── Scale features (604-dim) ──────────────────────────────────────────────────
feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_fused_tr).astype(np.float32)
X_te_sc = feat_scaler.transform(X_fused_te).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

# ── Train / val split ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_tr  = torch.tensor(i_train_n[tr_idx]);  X_tr  = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d[tr_idx])
i_val = torch.tensor(i_train_n[val_idx]); X_val = torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d[val_idx])
i_te  = torch.tensor(i_test_n);           X_te  = torch.tensor(X_te_sc)

loader = DataLoader(TensorDataset(i_tr, X_tr, y_tr), batch_size=BATCH_SIZE,
                    shuffle=True, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
model = WaveNet1D(feat_dim=X_tr.shape[1], channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()
scaler_amp = GradScaler()  # for mixed-precision training

n_params = sum(p.numel() for p in model.parameters())
rf_original = sum(2 * d for d in DILATIONS) * 4
log(f"Training {LABEL}  params={n_params:,}")
log(f"  channels={CHANNELS}  dilations={DILATIONS}")
log(f"  trace input: 8192 (4096 raw + 4096 smooth{W})")
log(f"  feature input: {X_tr.shape[1]} (302 raw + 302 smooth{W})")
log(f"  train={len(tr_idx)}  val={len(val_idx)}")

# ── Training loop ─────────────────────────────────────────────────────────────
best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for ib, xb, yb in loader:
        ib, xb, yb = ib.to(device), xb.to(device), yb.to(device)
        opt.zero_grad()
        with autocast():
            loss = crit(model(ib, xb), yb)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(opt)
        scaler_amp.update()
        ep += loss.item() * len(ib)
    ep /= len(i_tr)
    model.eval()
    with torch.no_grad(), autocast():
        vl = crit(batch_predict(lambda i, x: model(i, x), i_val, X_val, bs=EVAL_BATCH), y_val).item()
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
log(f"Done — best val={best_val:.5f}")

# ── Save model + scaler ───────────────────────────────────────────────────────
torch.save(model.state_dict(), f'model_wavenet_wide_fused{W}.pt')
np.save(f'scaler_wavenet_wide_fused{W}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_wavenet_wide_fused{W}_scale.npy', feat_scaler.scale_)
log(f"Saved model_wavenet_wide_fused{W}.pt + scaler (604-dim)")

# ── Predictions ───────────────────────────────────────────────────────────────
model.eval()
i_tr_all = torch.tensor(i_train_n); X_tr_all = torch.tensor(X_tr_sc)
with torch.no_grad(), autocast():
    logd_pred_tr = batch_predict(lambda i, x: model(i, x), i_tr_all, X_tr_all, bs=EVAL_BATCH).float().numpy()
    logd_pred_te = batch_predict(lambda i, x: model(i, x), i_te,     X_te,     bs=EVAL_BATCH).float().numpy()

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

np.save(f'pred_logd_wavenet_wide_fused{W}_test.npy', logd_pred_te)
log(f"Saved pred_logd_wavenet_wide_fused{W}_test.npy")

# ── Summary ───────────────────────────────────────────────────────────────────
r2_tr  = r2_score(d_train, d_pred_tr)
r2_te  = r2_score(d_test,  d_pred_te)
ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"Fused window: W={W}\n"
    f"  Training input: raw trace (4096) + smooth{W} trace (4096) = 8192-pt input\n"
    f"  Feature input:  raw feats (302) + smooth{W} feats (302)  = 604-dim\n"
    f"  Training samples: {len(d_train)} (same as baseline, not doubled)\n"
    f"  Test set: raw test + smooth{W} test (same preparation as training)\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}\n"
    f"Params: {n_params:,}\n"
    f"Train R²={r2_tr:.4f}  MAE={mean_absolute_error(d_train, d_pred_tr):.4f}  "
    f"MAPE={mape(d_train, d_pred_tr):.1f}%\n"
    f"Test  R²={r2_te:.4f}  MAE={mean_absolute_error(d_test,  d_pred_te):.4f}  "
    f"MAPE={mape(d_test,  d_pred_te):.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms], d_pred_te[ms]):.4f}  "
    f"MAPE={mape(d_test[ms], d_pred_te[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf], d_pred_te[mf]):.4f}  "
    f"MAPE={mape(d_test[mf], d_pred_te[mf]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_pt_wavenet_wide_fused{W}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_pt_wavenet_wide_fused{W}.txt"); print("Done.")
