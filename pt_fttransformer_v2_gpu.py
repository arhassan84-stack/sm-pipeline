"""
FT-Transformer V2 (E=64, 6 layers, 4 heads, dropout=0.2).

Architecture rationale for diversity vs original (E=64, 4L, 8H, dropout=0.1):
  - Depth: 4 → 6 layers (more feature interaction composition)
  - Heads: 8 → 4 (each head attends to a broader pattern; fewer but richer attention maps)
  - Dropout: 0.1 → 0.2 (stronger regularization → different bias/variance tradeoff)
  - Same embedding dim and width → comparable parameter count (~375k vs 236k)

The goal is a model that makes DIFFERENT errors from original FTT and large FTT,
improving ensemble diversity. Fewer heads with higher dropout tends to learn
coarser but more robust feature groupings.

Saves model_fttransformer_v2.pt for ensemble use.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL       = 'pt_fttransformer_v2'
E           = 64
N_HEADS     = 4
N_LAYERS    = 6
DIM_FF      = 256
DROPOUT     = 0.2
BATCH_SIZE  = 1024
MAX_EPOCHS  = 500
PATIENCE    = 40
LR          = 1e-3
WD          = 1e-4
T0          = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100


class FTTransformer(nn.Module):
    def __init__(self, n_features, E=64, n_heads=4, n_layers=6,
                 dim_ff=256, dropout=0.2):
        super().__init__()
        self.feat_weight = nn.Parameter(torch.randn(n_features, E) * 0.01)
        self.feat_bias   = nn.Parameter(torch.zeros(n_features, E))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, E))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=E, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, 1))

    def forward(self, x):
        tokens = x.unsqueeze(2) * self.feat_weight.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        out = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.head(out[:, 0, :]).squeeze(1)


t_total = time.time()
log(f"Device: {device}")
log("Loading 90% train cache...")
X_train = np.load('cache_X_train_90pct.npy'); d_train = np.load('cache_d_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy');  d_test  = np.load('cache_d_test_90pct.npy')
mask_tr = d_train<=10; X_train,d_train = X_train[mask_tr],d_train[mask_tr]
mask_te = d_test<=10;  X_test, d_test  = X_test[mask_te], d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Features: {X_train.shape[1]}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(X_tr_sc), size=int(0.1*len(X_tr_sc)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(X_tr_sc)), val_idx)

X_tr  = torch.tensor(X_tr_sc[tr_idx]).to(device)
y_tr  = torch.tensor(log_d[tr_idx]).to(device)
X_val = torch.tensor(X_tr_sc[val_idx]).to(device)
y_val = torch.tensor(log_d[val_idx]).to(device)
X_te  = torch.tensor(X_te_sc).to(device)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

model = FTTransformer(
    n_features=X_train.shape[1], E=E, n_heads=N_HEADS,
    n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  E={E}  heads={N_HEADS}  layers={N_LAYERS}  dim_ff={DIM_FF}")

best_val, best_state, patience_ct = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); ep = 0.0
    for xb, yb in loader:
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        ep += loss.item() * len(xb)
    ep /= len(X_tr)
    model.eval()
    with torch.no_grad(): vl = crit(model(X_val), y_val).item()
    sched.step(epoch-1)
    if vl < best_val-1e-6: best_val, best_state, patience_ct = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
    else: patience_ct += 1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  patience={patience_ct}")
    if patience_ct >= PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

torch.save(model.state_dict(), 'model_fttransformer_v2.pt')
np.save('scaler_fttransformer_v2_mean.npy', scaler.mean_)
np.save('scaler_fttransformer_v2_scale.npy', scaler.scale_)
log("Saved model_fttransformer_v2.pt + scaler")

model.eval()
with torch.no_grad():
    d_pred_te = np.exp(model(X_te).cpu().numpy())
runtime = time.time() - t_total

ms, mf = d_test<1.0, d_test>=1.0
r2_te  = r2_score(d_test, d_pred_te)
mae_te = mean_absolute_error(d_test, d_pred_te)
mape_te = mape(d_test, d_pred_te)
summary = (
    f"Task: {LABEL}\nArchitecture: FT-Transformer E={E} heads={N_HEADS} layers={N_LAYERS} dim_ff={DIM_FF}\n"
    f"Optimizer: AdamW lr={LR} wd={WD}  Scheduler: CosineAnnealingWarmRestarts T0={T0}\n"
    f"Batch: {BATCH_SIZE}  MaxEpochs: {MAX_EPOCHS}  Patience: {PATIENCE}\n"
    f"Device: {device}  Params: {n_params:,}  Features: {X_train.shape[1]}\n"
    f"Train: {len(d_train)}  Test: {len(d_test)}\n\n"
    f"Test R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred_te[ms]):.4f}  MAPE={mape(d_test[ms],d_pred_te[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred_te[mf]):.4f}  MAPE={mape(d_test[mf],d_pred_te[mf]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt"); print("Done.")
