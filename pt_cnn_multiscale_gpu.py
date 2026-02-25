"""
Improvement: Multi-scale 1D CNN with 3 parallel branches (fine/medium/coarse)
+ feature branch, all fused to predict log(d).
Uses 50k split with existing caches.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL        = 'pt_cnn_multiscale'
BATCH_SIZE   = 256
EVAL_BATCH   = 512   # smaller batches for val/test inference to avoid OOM
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

def batch_predict(model_fn, *cpu_tensors, bs=512):
    """Run model on CPU tensors in small batches to avoid GPU OOM."""
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)


class FineBranch(nn.Module):
    """Fine-scale: captures fast fluctuations (kernel=8)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8,  stride=2, padding=3),  nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=4, stride=2, padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,128,kernel_size=4, stride=2, padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x.unsqueeze(1)).squeeze(2)


class MediumBranch(nn.Module):
    """Medium-scale: captures diffusion timescale (kernel=32)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=32, stride=8,  padding=12), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=2,  padding=3),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=4, stride=2,  padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x.unsqueeze(1)).squeeze(2)


class CoarseBranch(nn.Module):
    """Coarse-scale: captures slow dynamics (kernel=128)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=128, stride=16, padding=56), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=16, stride=4,  padding=6),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=4,  stride=2,  padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x.unsqueeze(1)).squeeze(2)


class MultiScaleFusion(nn.Module):
    def __init__(self, feat_dim=207):
        super().__init__()
        self.fine   = FineBranch()
        self.medium = MediumBranch()
        self.coarse = CoarseBranch()
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        # 128 + 128 + 128 + 128 = 512 fused
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        f = torch.cat([self.fine(trace), self.medium(trace),
                       self.coarse(trace), self.feat_branch(feats)], dim=1)
        return self.head(f).squeeze(1)


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

# Keep all data on CPU; move batches to GPU in training loop to avoid OOM
i_tr  = torch.tensor(i_train_n[tr_idx]);  X_tr = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d[tr_idx])
i_val = torch.tensor(i_train_n[val_idx]); X_val= torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d[val_idx])
i_te  = torch.tensor(i_test_n);           X_te = torch.tensor(X_te_sc)

loader = DataLoader(TensorDataset(i_tr,X_tr,y_tr), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = MultiScaleFusion(feat_dim=X_tr.shape[1]).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}...")
best_val, best_state, patience_count = float('inf'), None, 0
train_losses, val_losses = [], []

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); ep=0.0
    for ib,xb,yb in loader:
        ib,xb,yb = ib.to(device),xb.to(device),yb.to(device)
        opt.zero_grad(); loss=crit(model(ib,xb),yb); loss.backward(); opt.step()
        ep += loss.item()*len(ib)
    ep /= len(i_tr)
    model.eval()
    with torch.no_grad():
        vl = crit(batch_predict(lambda i,x: model(i,x), i_val, X_val, bs=EVAL_BATCH), y_val).item()
    sched.step(vl)
    train_losses.append(ep); val_losses.append(vl)
    if vl < best_val-1e-6: best_val,best_state,patience_count = vl,{k:v.clone() for k,v in model.state_dict().items()},0
    else: patience_count+=1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count>=PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done — best val={best_val:.5f}")

model.eval()
i_tr_all = torch.tensor(i_train_n); X_tr_all = torch.tensor(X_tr_sc)
with torch.no_grad():
    d_pred_tr = np.exp(batch_predict(lambda i,x: model(i,x), i_tr_all, X_tr_all, bs=EVAL_BATCH).numpy())
    d_pred_te = np.exp(batch_predict(lambda i,x: model(i,x), i_te,     X_te,     bs=EVAL_BATCH).numpy())
runtime = time.time()-t_total

r2_tr,mae_tr,mape_tr = r2_score(d_train,d_pred_tr),mean_absolute_error(d_train,d_pred_tr),mape(d_train,d_pred_tr)
r2_te,mae_te,mape_te = r2_score(d_test,d_pred_te),mean_absolute_error(d_test,d_pred_te),mape(d_test,d_pred_te)

summary = (f"Task: {LABEL}\nArchitecture: MultiScale-CNN (fine k=8 + medium k=32 + coarse k=128) + FC(207) → head\n"
           f"Optimizer: AdamW lr={LR} wd={WEIGHT_DECAY}  Scheduler: ReduceLROnPlateau\n"
           f"Batch: {BATCH_SIZE}  MaxEpochs: {MAX_EPOCHS}  Patience: {PATIENCE}\n"
           f"Device: {device}  Params: {n_params:,}\nTrain: {len(d_train)}  Test: {len(d_test)}\n\n"
           f"Train  R²={r2_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape_tr:.1f}%\n"
           f"Test   R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\nRuntime: {runtime:.1f}s\n")
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")

fig,axes=plt.subplots(1,2,figsize=(12,5))
ax=axes[0]; ax.scatter(d_test,d_pred_te,alpha=0.2,s=5,color='purple')
lims=[min(d_test.min(),d_pred_te.min()),max(d_test.max(),d_pred_te.max())]
ax.plot(lims,lims,'r--',lw=1); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d'); ax.set_title(f'{LABEL} R²={r2_te:.4f}')
ax=axes[1]; ax.plot(train_losses,label='Train'); ax.plot(val_losses,label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
plt.tight_layout(); plt.savefig(f'{LABEL}.png',dpi=150)
log(f"Saved {LABEL}.png"); print("Done.")
