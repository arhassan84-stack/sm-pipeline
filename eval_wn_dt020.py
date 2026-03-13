"""
Evaluation-only script for pt_wavenet_wide_aug_dt020.
The model trained successfully (epoch 41, early stopping, best val=0.02689)
but crashed at evaluation (OOM in norm_traces). This script loads the saved
checkpoint and computes predictions with chunked normalization.
"""

import numpy as np, time, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

DT_TAG  = 'dt020'
DT_MS   = 0.2
N_BINS  = 20480
LABEL   = f'pt_wavenet_wide_aug_{DT_TAG}'
CHANNELS    = 256
DILATIONS   = [1, 2, 4, 8, 16, 32, 64, 128]
EVAL_BATCH  = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100

t_total = time.time()
log(f"Device: {device}")
log(f"Loading caches for {DT_TAG}...")


class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)) + x)


class WaveNet1D(nn.Module):
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None: dilations = [1,2,4,8,16,32,64,128]
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


# ── Load data ─────────────────────────────────────────────────────────────────
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
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

scaler_mean  = np.load(f'scaler_{LABEL}_mean.npy')
scaler_scale = np.load(f'scaler_{LABEL}_scale.npy')
X_tr_sc = ((X_train - scaler_mean) / scaler_scale).astype(np.float32)
X_te_sc = ((X_test  - scaler_mean) / scaler_scale).astype(np.float32)

# ── Load model ────────────────────────────────────────────────────────────────
feat_dim = X_tr_sc.shape[1]
model = WaveNet1D(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
model.load_state_dict(torch.load(f'model_{LABEL}.pt', map_location=device))
model.eval()
log(f"Loaded model_{LABEL}.pt  feat_dim={feat_dim}")

# ── Chunked predict (avoids OOM) ──────────────────────────────────────────────
def batch_predict_raw(raw_np, X_np_sc, bs=128):
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

log("Predicting on test set...")
logd_pred_te = batch_predict_raw(i_test_raw,  X_te_sc, bs=EVAL_BATCH).numpy()
log("Predicting on train set...")
logd_pred_tr = batch_predict_raw(i_train_raw, X_tr_sc, bs=EVAL_BATCH).numpy()

np.save(f'pred_logd_{LABEL}_test.npy',  logd_pred_te)
np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)

d_pred_te = np.exp(logd_pred_te)
d_pred_tr = np.exp(logd_pred_tr)
runtime   = time.time() - t_total
ms, mf = d_test < 1.0, d_test >= 1.0

summary = (
    f"Task: {LABEL}\n"
    f"dt={DT_MS}ms  N_BINS={N_BINS}  (eval-only: model loaded from checkpoint)\n"
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
