"""
Specialist model for fast diffusion: trained on d >= 0.5 (90% split).
Overlaps with slow model in [0.5, 1.5] for smooth soft-gating.
Saves weights + scaler for router evaluation.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL        = 'pt_specialist_fast'
BATCH_SIZE   = 1024
MAX_EPOCHS   = 500
PATIENCE     = 40
LR           = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.15
T0           = 50
D_LOW        = 0.5   # train on d >= this (generous overlap)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100


class MLP_BN(nn.Module):
    def __init__(self, in_dim, hidden=(2048,1024,512,256,128), dropout=0.15):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)


t_total = time.time()
log(f"Device: {device}")
log("Loading 90% train cache...")
X_train_all = np.load('cache_X_train_90pct.npy'); d_train_all = np.load('cache_d_train_90pct.npy')
X_test_all  = np.load('cache_X_test_90pct.npy');  d_test_all  = np.load('cache_d_test_90pct.npy')

# Filter training to d >= D_LOW (with overlap), test to d <= 10
mask_tr = (d_train_all >= D_LOW) & (d_train_all <= 10)
mask_te = (d_test_all  >  0)    & (d_test_all  <= 10)
X_train, d_train = X_train_all[mask_tr], d_train_all[mask_tr]
X_test,  d_test  = X_test_all[mask_te],  d_test_all[mask_te]
log(f"Train (d>={D_LOW}): {len(d_train)}  Test (d<=10): {len(d_test)}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(X_tr_sc), size=int(0.1*len(X_tr_sc)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(X_tr_sc)), val_idx)

X_tr = torch.tensor(X_tr_sc[tr_idx]).to(device); y_tr = torch.tensor(log_d[tr_idx]).to(device)
X_val= torch.tensor(X_tr_sc[val_idx]).to(device); y_val= torch.tensor(log_d[val_idx]).to(device)
X_te = torch.tensor(X_te_sc).to(device)
loader = DataLoader(TensorDataset(X_tr,y_tr), batch_size=BATCH_SIZE, shuffle=True)

model = MLP_BN(X_tr.shape[1], dropout=DROPOUT).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  samples={len(d_train)}...")
best_val, best_state, patience_count = float('inf'), None, 0
train_losses, val_losses = [], []

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); ep=0.0
    for xb,yb in loader:
        opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step()
        ep += loss.item()*len(xb)
    ep /= len(X_tr)
    model.eval()
    with torch.no_grad(): vl = crit(model(X_val),y_val).item()
    sched.step(epoch-1)
    train_losses.append(ep); val_losses.append(vl)
    if vl < best_val-1e-6: best_val,best_state,patience_count = vl,{k:v.clone() for k,v in model.state_dict().items()},0
    else: patience_count+=1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count>=PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

# Save weights + scaler for ensemble/routing
torch.save(model.state_dict(), 'model_specialist_fast.pt')
np.save('scaler_specialist_fast_mean.npy', scaler.mean_)
np.save('scaler_specialist_fast_scale.npy', scaler.scale_)
log("Saved model_specialist_fast.pt + scaler")

model.eval()
with torch.no_grad():
    d_pred_tr = np.exp(model(torch.tensor(X_tr_sc).to(device)).cpu().numpy())
    d_pred_te = np.exp(model(X_te).cpu().numpy())
runtime = time.time()-t_total

# Overall metrics
r2_tr  = r2_score(d_train, d_pred_tr);   mae_tr  = mean_absolute_error(d_train, d_pred_tr)
r2_te  = r2_score(d_test,  d_pred_te);   mae_te  = mean_absolute_error(d_test,  d_pred_te)
# Per-regime test metrics
m1 = d_test <  1;  m2 = d_test >= 1
r2_slow  = r2_score(d_test[m1], d_pred_te[m1]) if m1.sum()>1 else float('nan')
r2_fast  = r2_score(d_test[m2], d_pred_te[m2]) if m2.sum()>1 else float('nan')
mape_slow = mape(d_test[m1], d_pred_te[m1]) if m1.sum()>0 else float('nan')
mape_fast = mape(d_test[m2], d_pred_te[m2]) if m2.sum()>0 else float('nan')

summary = (f"Task: {LABEL}\nArchitecture: 2048-1024-512-256-128 + BN + Dropout({DROPOUT})\n"
           f"Training filter: d >= {D_LOW}  Train samples: {len(d_train)}\n"
           f"Optimizer: AdamW lr={LR} wd={WEIGHT_DECAY}  Scheduler: CosineAnnealingWarmRestarts T0={T0}\n"
           f"Batch: {BATCH_SIZE}  MaxEpochs: {MAX_EPOCHS}  Patience: {PATIENCE}\n"
           f"Device: {device}  Params: {n_params:,}\n\n"
           f"Train  R²={r2_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape(d_train,d_pred_tr):.1f}%\n"
           f"Test (full, d<=10)  R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape(d_test,d_pred_te):.1f}%\n"
           f"  d < 1    ({m1.sum():5d} samples): R²={r2_slow:.4f}  MAPE={mape_slow:.1f}%\n"
           f"  d >= 1   ({m2.sum():5d} samples): R²={r2_fast:.4f}  MAPE={mape_fast:.1f}%\n"
           f"Runtime: {runtime:.1f}s\n")
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")

fig,axes=plt.subplots(1,2,figsize=(12,5))
ax=axes[0]; ax.scatter(d_test,d_pred_te,alpha=0.2,s=5,c=np.where(d_test<1,'steelblue','darkorange'))
lims=[min(d_test.min(),d_pred_te.min()),max(d_test.max(),d_pred_te.max())]
ax.plot(lims,lims,'r--',lw=1); ax.axvline(1,color='gray',lw=0.8,ls=':'); ax.axhline(1,color='gray',lw=0.8,ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d'); ax.set_title(f'{LABEL} R²={r2_te:.4f}')
ax=axes[1]; ax.plot(train_losses,label='Train'); ax.plot(val_losses,label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
plt.tight_layout(); plt.savefig(f'{LABEL}.png',dpi=150)
log(f"Saved {LABEL}.png"); print("Done.")
