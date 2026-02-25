"""
Improvement: 1D Transformer on raw traces (patch-based) + engineered feature fusion.
Divides 4096-point trace into 64 patches of 64 points each.
Uses 50k split with existing caches.
"""
import numpy as np, time, math, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL        = 'pt_transformer'
BATCH_SIZE   = 256
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATCH_SIZE   = 64    # 4096 / 64 = 64 patches
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 256
DROPOUT      = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100


class PatchEmbedding(nn.Module):
    """Split trace into patches and project to d_model."""
    def __init__(self, patch_size=64, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        # x: (B, 4096) → (B, n_patches, d_model)
        B, L = x.shape
        n = L // self.patch_size
        x = x.reshape(B, n, self.patch_size)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TraceTransformer(nn.Module):
    def __init__(self, feat_dim=207, patch_size=64, d_model=128,
                 n_heads=4, n_layers=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model, max_len=4096//patch_size+4, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # d_model (trace CLS) + 128 (features) = 256
        self.head = nn.Sequential(
            nn.Linear(d_model + 128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, trace, feats):
        x = self.patch_embed(trace)           # (B, 64, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)               # (B, 64, d_model)
        x = x.mean(dim=1)                     # (B, d_model) — average pooling
        f = self.feat_branch(feats)            # (B, 128)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


t_total = time.time()
log(f"Device: {device}")
log("Loading 50k raw traces + features...")
i_train = np.load('cache_i_train.npy').astype(np.float32)
X_train = np.load('cache_X_train_psd.npy').astype(np.float32)
d_train = np.load('cache_d_train.npy')
i_test  = np.load('cache_i_test.npy').astype(np.float32)
X_test  = np.load('cache_X_test_psd.npy').astype(np.float32)
d_test  = np.load('cache_d_test.npy')

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

i_tr  = torch.tensor(i_train_n[tr_idx]).to(device); X_tr = torch.tensor(X_tr_sc[tr_idx]).to(device)
y_tr  = torch.tensor(log_d[tr_idx]).to(device)
i_val = torch.tensor(i_train_n[val_idx]).to(device); X_val= torch.tensor(X_tr_sc[val_idx]).to(device)
y_val = torch.tensor(log_d[val_idx]).to(device)
i_te  = torch.tensor(i_test_n).to(device);           X_te = torch.tensor(X_te_sc).to(device)

loader = DataLoader(TensorDataset(i_tr,X_tr,y_tr), batch_size=BATCH_SIZE, shuffle=True)

model = TraceTransformer(feat_dim=X_tr.shape[1], patch_size=PATCH_SIZE,
                         d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                         d_ff=D_FF, dropout=DROPOUT).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  patches={4096//PATCH_SIZE}  d_model={D_MODEL}  layers={N_LAYERS}...")
best_val, best_state, patience_count = float('inf'), None, 0
train_losses, val_losses = [], []

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); ep=0.0
    for ib,xb,yb in loader:
        opt.zero_grad(); loss=crit(model(ib,xb),yb); loss.backward(); opt.step()
        ep += loss.item()*len(ib)
    ep /= len(i_tr)
    model.eval()
    with torch.no_grad(): vl = crit(model(i_val,X_val),y_val).item()
    sched.step(vl)
    train_losses.append(ep); val_losses.append(vl)
    if vl < best_val-1e-6: best_val,best_state,patience_count = vl,{k:v.clone() for k,v in model.state_dict().items()},0
    else: patience_count+=1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count>=PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

model.eval()
i_tr_all = torch.tensor(i_train_n).to(device); X_tr_all = torch.tensor(X_tr_sc).to(device)
with torch.no_grad():
    d_pred_tr = np.exp(model(i_tr_all,X_tr_all).cpu().numpy())
    d_pred_te = np.exp(model(i_te,X_te).cpu().numpy())
runtime = time.time()-t_total

r2_tr,mae_tr,mape_tr = r2_score(d_train,d_pred_tr),mean_absolute_error(d_train,d_pred_tr),mape(d_train,d_pred_tr)
r2_te,mae_te,mape_te = r2_score(d_test,d_pred_te),mean_absolute_error(d_test,d_pred_te),mape(d_test,d_pred_te)

summary = (f"Task: {LABEL}\nArchitecture: Transformer (64 patches × 64pts, d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}) + FC(207)\n"
           f"Optimizer: AdamW lr={LR} wd={WEIGHT_DECAY}  Scheduler: ReduceLROnPlateau\n"
           f"Batch: {BATCH_SIZE}  MaxEpochs: {MAX_EPOCHS}  Patience: {PATIENCE}\n"
           f"Device: {device}  Params: {n_params:,}\nTrain: {len(d_train)}  Test: {len(d_test)}\n\n"
           f"Train  R²={r2_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape_tr:.1f}%\n"
           f"Test   R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\nRuntime: {runtime:.1f}s\n")
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")

fig,axes=plt.subplots(1,2,figsize=(12,5))
ax=axes[0]; ax.scatter(d_test,d_pred_te,alpha=0.2,s=5,color='crimson')
lims=[min(d_test.min(),d_pred_te.min()),max(d_test.max(),d_pred_te.max())]
ax.plot(lims,lims,'r--',lw=1); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d'); ax.set_title(f'{LABEL} R²={r2_te:.4f}')
ax=axes[1]; ax.plot(train_losses,label='Train'); ax.plot(val_losses,label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
plt.tight_layout(); plt.savefig(f'{LABEL}.png',dpi=150)
log(f"Saved {LABEL}.png"); print("Done.")
