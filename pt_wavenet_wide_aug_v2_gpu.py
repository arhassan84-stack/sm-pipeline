"""
Fine-tune wavenet_wide_aug from its saved checkpoint with a lower learning rate.

Loads model_wavenet_wide_aug.pt and continues training on the same augmented
data with LR=5e-5 (10x lower than original 5e-4) and a smooth CosineAnnealingLR
(no restarts) to refine the solution without disrupting learned representations.

Saves:
  model_wavenet_wide_aug_v2.pt
  scaler_wavenet_wide_aug_v2_mean/scale.npy
  pred_logd_wavenet_wide_aug_v2_test.npy
  results_pt_wavenet_wide_aug_v2.txt
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_wide_aug_v2'
CHANNELS     = 256
BATCH_SIZE   = 128
EVAL_BATCH   = 256
MAX_EPOCHS   = 120
PATIENCE     = 20
LR           = 5e-5        # 10x lower than original 5e-4
WEIGHT_DECAY = 1e-4
T_MAX        = 100         # CosineAnnealingLR (no restarts)
ETA_MIN      = 1e-6
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]
CHECKPOINT   = 'model_wavenet_wide_aug.pt'

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
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Fusion: channels(256) + feat(128) = 384
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

# ── Load augmented training cache (same data as original aug version) ──────────
log("Loading augmented training cache...")
i_train = np.load('cache_i_train_aug.npy').astype(np.float32)
X_train = np.load('cache_X_train_aug.npy').astype(np.float32)
d_train = np.load('cache_d_train_aug.npy')

# Apply d<=10 mask (same as aug version)
mask_tr = d_train <= 10
i_train, X_train, d_train = i_train[mask_tr], X_train[mask_tr], d_train[mask_tr]
log(f"Train (d<=10): {len(d_train)}")

log("Loading test set...")
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test  = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test  = np.load('cache_d_test_90pct.npy')
mask_te = d_test <= 10
i_test, X_test, d_test = i_test[mask_te], X_test[mask_te], d_test[mask_te]
log(f"Test: {len(d_test)}")

# ── Normalise traces ──────────────────────────────────────────────────────────
i_train_n = (i_train - i_train.mean(1, keepdims=True)) / (i_train.std(1, keepdims=True) + 1e-8)
i_test_n  = (i_test  - i_test.mean(1,  keepdims=True)) / (i_test.std(1,  keepdims=True) + 1e-8)

# ── Scale features (refit on same aug data → identical to original scaler) ────
feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
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

# ── Build model and load checkpoint ──────────────────────────────────────────
model = WaveNet1D(feat_dim=X_tr.shape[1], channels=CHANNELS, dilations=DILATIONS).to(device)

log(f"Loading checkpoint: {CHECKPOINT}")
state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)
log("Checkpoint loaded successfully")

n_params = sum(p.numel() for p in model.parameters())
log(f"Fine-tuning {LABEL}  params={n_params:,}")
log(f"  LR={LR} (10x lower)  T_max={T_MAX}  eta_min={ETA_MIN}  PATIENCE={PATIENCE}")

# ── Optimizer + smooth cosine scheduler (no restarts) ────────────────────────
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=ETA_MIN)
crit  = nn.MSELoss()

# ── Training loop ─────────────────────────────────────────────────────────────
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
    sched.step()
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
torch.save(model.state_dict(), 'model_wavenet_wide_aug_v2.pt')
np.save('scaler_wavenet_wide_aug_v2_mean.npy',  feat_scaler.mean_)
np.save('scaler_wavenet_wide_aug_v2_scale.npy', feat_scaler.scale_)
log("Saved model_wavenet_wide_aug_v2.pt + scaler")

# ── Predictions ───────────────────────────────────────────────────────────────
model.eval()
i_tr_all = torch.tensor(i_train_n); X_tr_all = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(lambda i, x: model(i, x), i_tr_all, X_tr_all, bs=EVAL_BATCH).numpy()
    logd_pred_te = batch_predict(lambda i, x: model(i, x), i_te,     X_te,     bs=EVAL_BATCH).numpy()

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

np.save('pred_logd_wavenet_wide_aug_v2_test.npy', logd_pred_te)
log("Saved pred_logd_wavenet_wide_aug_v2_test.npy")

# ── Summary ───────────────────────────────────────────────────────────────────
r2_tr  = r2_score(d_train, d_pred_tr)
r2_te  = r2_score(d_test,  d_pred_te)
ms, mf = d_test < 1.0, d_test >= 1.0
summary = (
    f"Task: {LABEL}\n"
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}\n"
    f"Fine-tuned from: {CHECKPOINT}\n"
    f"Params: {n_params:,}\n"
    f"LR={LR}  T_max={T_MAX}  eta_min={ETA_MIN}  PATIENCE={PATIENCE}\n"
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
with open('results_pt_wavenet_wide_aug_v2.txt', 'w') as f:
    f.write(summary)
log("Saved results_pt_wavenet_wide_aug_v2.txt"); print("Done.")
