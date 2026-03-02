"""
Hybrid WaveNet+FTT v2 — per-branch learning rates.

Motivation: In v1, a single LR=5e-4 was used for both the WaveNet trace branch
and the FT-Transformer feature branch.  However, the original FTT model trains
best at LR=1e-3 (from Round 6 experiments).  Using a unified low LR under-trains
the transformer, causing the FTT branch to under-contribute and pushing the model
to rely too heavily on the WaveNet branch.

Fix: separate AdamW param groups
  WaveNet branch (input_proj + wave_blocks + head): LR=5e-4  (proven optimal for WaveNet)
  FTT branch (feat_weight/bias/cls_token + transformer + ftt_norm): LR=1e-3  (proven for FTT)

Architecture: identical to pt_wavenet_ftt (668k params)
  WaveNet trace branch: stride-4 proj + 8 DilatedResBlocks + GAP → 128-dim
  FTT feature branch: feature tokenization + 4-layer transformer + CLS → 64-dim
  Fusion head: concat [128+64=192] → 128 → 64 → 1

Saves pred_logd_wavenet_ftt_v2_train/test.npy for meta-learner.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_ftt_v2'
CHANNELS     = 128
E            = 64
N_HEADS      = 8
N_LAYERS     = 4
DIM_FF       = 256
DROPOUT      = 0.1
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_SIZE   = 256
EVAL_BATCH   = 512
MAX_EPOCHS   = 200
PATIENCE     = 30
LR_WAVE      = 5e-4    # WaveNet branch LR
LR_FTT       = 1e-3    # FTT branch LR (higher — transformer needs larger LR)
WEIGHT_DECAY = 1e-4
T0           = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

def batch_predict(model_fn, *cpu_tensors, bs=512):
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


class WaveNetFTT(nn.Module):
    """
    Hybrid: WaveNet dilated CNN (trace) + FT-Transformer (features) → fusion.
    Identical architecture to v1; trained with per-branch learning rates.
    """
    def __init__(self, feat_dim, channels=128, E=64, n_heads=8, n_layers=4,
                 dim_ff=256, dropout=0.1, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        # ── WaveNet trace branch ────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.wave_blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── FT-Transformer feature branch ───────────────────────────────────
        self.feat_weight = nn.Parameter(torch.randn(feat_dim, E) * 0.01)
        self.feat_bias   = nn.Parameter(torch.zeros(feat_dim, E))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, E))
        enc = nn.TransformerEncoderLayer(
            d_model=E, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ftt_norm    = nn.LayerNorm(E)

        # ── Fusion head ─────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(channels + E, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        tx = self.input_proj(trace.unsqueeze(1))
        tx = self.wave_blocks(tx)
        tx = self.gap(tx).squeeze(2)
        tokens = feats.unsqueeze(2) * self.feat_weight.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        cls    = self.cls_token.expand(feats.size(0), -1, -1)
        out    = self.transformer(torch.cat([cls, tokens], dim=1))
        fx     = self.ftt_norm(out[:, 0, :])
        return self.head(torch.cat([tx, fx], dim=1)).squeeze(1)


t_total = time.time()
log(f"Device: {device}")
log("Loading 90% raw traces + features...")
i_train = np.load('cache_i_train_90pct.npy').astype(np.float32)
X_train = np.load('cache_X_train_90pct.npy').astype(np.float32)
d_train = np.load('cache_d_train_90pct.npy')
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test  = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test  = np.load('cache_d_test_90pct.npy')

mask_tr = d_train<=10; i_train,X_train,d_train = i_train[mask_tr],X_train[mask_tr],d_train[mask_tr]
mask_te = d_test<=10;  i_test, X_test, d_test  = i_test[mask_te], X_test[mask_te], d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

i_train_n = (i_train - i_train.mean(1,keepdims=True)) / (i_train.std(1,keepdims=True)+1e-8)
i_test_n  = (i_test  - i_test.mean(1, keepdims=True)) / (i_test.std(1, keepdims=True)+1e-8)

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1*len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_tr  = torch.tensor(i_train_n[tr_idx]);  X_tr = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d[tr_idx])
i_val = torch.tensor(i_train_n[val_idx]); X_val= torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d[val_idx])
i_te  = torch.tensor(i_test_n);           X_te = torch.tensor(X_te_sc)

loader = DataLoader(TensorDataset(i_tr, X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = WaveNetFTT(
    feat_dim=X_tr.shape[1], channels=CHANNELS, E=E,
    n_heads=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF,
    dropout=DROPOUT, dilations=DILATIONS,
).to(device)

# ── Per-branch optimizer: WaveNet at LR_WAVE, FTT at LR_FTT ──────────────────
wave_params = (list(model.input_proj.parameters()) +
               list(model.wave_blocks.parameters()) +
               list(model.head.parameters()))
ftt_params  = ([model.feat_weight, model.feat_bias, model.cls_token] +
               list(model.transformer.parameters()) +
               list(model.ftt_norm.parameters()))

opt = torch.optim.AdamW([
    {'params': wave_params, 'lr': LR_WAVE},
    {'params': ftt_params,  'lr': LR_FTT},
], weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
rf_orig  = sum(2*d for d in DILATIONS) * 4
log(f"Training {LABEL}  params={n_params:,}")
log(f"  WaveNet channels={CHANNELS}  RF={rf_orig}lags  LR={LR_WAVE}")
log(f"  FTT  E={E}  heads={N_HEADS}  layers={N_LAYERS}  LR={LR_FTT}  (2x WaveNet LR)")

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS+1):
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
        vl = crit(batch_predict(lambda i,x: model(i,x), i_val, X_val, bs=EVAL_BATCH), y_val).item()
    sched.step(epoch-1)
    if vl < best_val-1e-6:
        best_val, best_state, patience_count = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
    else:
        patience_count += 1
    if epoch%10==0:
        log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  "
            f"lr_wave={opt.param_groups[0]['lr']:.2e}  lr_ftt={opt.param_groups[1]['lr']:.2e}  "
            f"patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

torch.save(model.state_dict(), 'model_wavenet_ftt_v2.pt')
np.save('scaler_wavenet_ftt_v2_mean.npy', feat_scaler.mean_)
np.save('scaler_wavenet_ftt_v2_scale.npy', feat_scaler.scale_)
log("Saved model_wavenet_ftt_v2.pt + scaler")

model.eval()
i_tr_all = torch.tensor(i_train_n); X_tr_all = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(lambda i,x: model(i,x), i_tr_all, X_tr_all, bs=EVAL_BATCH).numpy()
    logd_pred_te = batch_predict(lambda i,x: model(i,x), i_te,     X_te,     bs=EVAL_BATCH).numpy()

d_pred_tr = np.exp(logd_pred_tr)
d_pred_te = np.exp(logd_pred_te)
runtime   = time.time() - t_total

np.save('pred_logd_wavenet_ftt_v2_train.npy', logd_pred_tr)
np.save('pred_logd_wavenet_ftt_v2_test.npy',  logd_pred_te)
log("Saved pred_logd_wavenet_ftt_v2_train/test.npy")

r2_tr  = r2_score(d_train, d_pred_tr)
r2_te  = r2_score(d_test,  d_pred_te)
ms, mf = d_test<1.0, d_test>=1.0
summary = (
    f"Task: {LABEL}\n"
    f"Architecture: WaveNet(channels={CHANNELS}, dilations={DILATIONS}, RF={rf_orig}lags)"
    f" + FTT(E={E}, heads={N_HEADS}, layers={N_LAYERS}) → fusion\n"
    f"  Per-branch LR: WaveNet={LR_WAVE}  FTT={LR_FTT}\n"
    f"Params: {n_params:,}\n"
    f"Train R²={r2_tr:.4f}  MAE={mean_absolute_error(d_train,d_pred_tr):.4f}  MAPE={mape(d_train,d_pred_tr):.1f}%\n"
    f"Test  R²={r2_te:.4f}  MAE={mean_absolute_error(d_test,d_pred_te):.4f}  MAPE={mape(d_test,d_pred_te):.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred_te[ms]):.4f}  MAPE={mape(d_test[ms],d_pred_te[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred_te[mf]):.4f}  MAPE={mape(d_test[mf],d_pred_te[mf]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt"); print("Done.")
