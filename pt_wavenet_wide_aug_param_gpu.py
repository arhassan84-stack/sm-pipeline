"""
Parameterized WaveNet wide aug — dt=1ms, for ensemble and large-arch experiments.

Usage:
  # Ensemble seed 1:
  python pt_wavenet_wide_aug_param_gpu.py --seed 1
  # Ensemble seed 2:
  python pt_wavenet_wide_aug_param_gpu.py --seed 2
  # Large architecture (channels=512):
  python pt_wavenet_wide_aug_param_gpu.py --channels 512

LABEL suffixes:
  seed>0  → {base}_s{seed}        e.g. wavenet_wide_aug_s1
  c!=256  → {base}_c{channels}    e.g. wavenet_wide_aug_c512
  both    → {base}_s{seed}_c{channels}
"""

import argparse, numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--seed',     type=int, default=0,   help='Random seed (0 = default)')
parser.add_argument('--channels', type=int, default=256, help='WaveNet channel width')
args = parser.parse_args()

BASE_LABEL   = 'wavenet_wide_aug'
suffix       = (f'_s{args.seed}' if args.seed > 0 else '') + \
               (f'_c{args.channels}' if args.channels != 256 else '')
LABEL        = BASE_LABEL + suffix
CHANNELS     = args.channels
BATCH_SIZE   = 128 if args.channels <= 256 else 64
EVAL_BATCH   = 256 if args.channels <= 256 else 128
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

if args.seed > 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100

def batch_predict(model_fn, *cpu_tensors, bs=256):
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)


class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn   = nn.BatchNorm1d(channels)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)) + x)


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


t_total = time.time()
log(f"Device: {device}")
log(f"LABEL={LABEL}  channels={CHANNELS}  seed={args.seed}  batch={BATCH_SIZE}")
log("Loading aug training cache + 90pct test cache...")

i_train = np.load('cache_i_train_aug.npy').astype(np.float32)
X_train = np.load('cache_X_train_aug.npy').astype(np.float32)
d_train = np.load('cache_d_train_aug.npy')
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test  = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test  = np.load('cache_d_test_90pct.npy')

mask_tr = d_train <= 10; i_train, X_train, d_train = i_train[mask_tr], X_train[mask_tr], d_train[mask_tr]
mask_te = d_test  <= 10; i_test,  X_test,  d_test  = i_test[mask_te],  X_test[mask_te],  d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

i_train_n = (i_train - i_train.mean(1, keepdims=True)) / (i_train.std(1, keepdims=True) + 1e-8)
i_test_n  = (i_test  - i_test.mean(1,  keepdims=True)) / (i_test.std(1,  keepdims=True) + 1e-8)

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

# Fixed val split (seed=42) so all ensemble members train on same data
rng     = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_tr  = torch.tensor(i_train_n[tr_idx]);  X_tr  = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d[tr_idx])
i_val = torch.tensor(i_train_n[val_idx]); X_val = torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d[val_idx])
i_te  = torch.tensor(i_test_n);           X_te  = torch.tensor(X_te_sc)

loader = DataLoader(TensorDataset(i_tr, X_tr, y_tr),
                    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = WaveNet1D(feat_dim=X_tr.shape[1], channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params  = sum(p.numel() for p in model.parameters())
rf_orig   = sum(2 * d for d in DILATIONS) * 4
log(f"Training {LABEL}  params={n_params:,}  channels={CHANNELS}  RF={rf_orig}")

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for ib, xb, yb in loader:
        ib, xb, yb = ib.to(device), xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(ib, xb), yb)
        loss.backward(); opt.step()
        ep += loss.item() * len(ib)
    ep /= len(i_tr)
    model.eval()
    with torch.no_grad():
        vl = crit(batch_predict(lambda i, x: model(i, x), i_val, X_val, bs=EVAL_BATCH), y_val).item()
    sched.step(epoch - 1)
    lr_now = opt.param_groups[0]['lr']
    if vl < best_val - 1e-6:
        best_val, patience_count = vl, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1
    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={lr_now:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)
log(f"Saved model_{LABEL}.pt + scaler")

model.eval()
i_tr_all = torch.tensor(i_train_n); X_tr_all = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(lambda i, x: model(i, x), i_tr_all, X_tr_all, bs=EVAL_BATCH).numpy()
    logd_pred_te = batch_predict(lambda i, x: model(i, x), i_te,     X_te,     bs=EVAL_BATCH).numpy()

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"channels={CHANNELS}  seed={args.seed}\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}  RF={rf_orig}\n"
    f"Params: {n_params:,}\n"
    f"Train R²={r2_score(d_train,d_pred_tr):.4f}  MAE={mean_absolute_error(d_train,d_pred_tr):.4f}  MAPE={mape(d_train,d_pred_tr):.2f}%\n"
    f"Test  R²={r2_score(d_test,d_pred_te):.4f}  MAE={mean_absolute_error(d_test,d_pred_te):.4f}  MAPE={mape(d_test,d_pred_te):.2f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred_te[ms]):.4f}  MAPE={mape(d_test[ms],d_pred_te[ms]):.2f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred_te[mf]):.4f}  MAPE={mape(d_test[mf],d_pred_te[mf]):.2f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
