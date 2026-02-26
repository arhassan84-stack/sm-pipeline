"""
CNN fusion model trained on PEAK-CENTERED CROPPED traces (CROP_W=512 pts).
Motivation: for d>=1, the burst is only ~33 pts wide in a 4096-pt trace.
Cropping to ±256 around the intensity peak focuses the CNN on the signal region
and removes ~3584 pts of background noise that dilute the learning signal.

Uses the same FusionNet architecture (AdaptiveAvgPool handles any trace length).
Saves model_cnn_peakcrop.pt for ensemble use.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL      = 'pt_cnn_peakcrop'
BATCH_SIZE = 512
MAX_EPOCHS = 300
PATIENCE   = 30
LR         = 1e-3
WD         = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

class CNN_Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=16,stride=4,padding=6),   nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=8,stride=4,padding=2),   nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=4,stride=2,padding=1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,kernel_size=4,stride=2,padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x.unsqueeze(1)).squeeze(2)

class Feature_Branch(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.cnn_branch  = CNN_Branch()
        self.feat_branch = Feature_Branch(in_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256,64),  nn.ReLU(), nn.Linear(64,1),
        )
    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace), self.feat_branch(feats)], dim=1)).squeeze(1)

t_total = time.time()
log(f"Device: {device}")
log("Loading peak-cropped traces + feature caches...")
i_train = np.load('cache_i_train_peakcrop.npy').astype(np.float32)
i_test  = np.load('cache_i_test_peakcrop.npy').astype(np.float32)
X_train = np.load('cache_X_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy')
d_train = np.load('cache_d_train_90pct.npy')
d_test  = np.load('cache_d_test_90pct.npy')
log(f"Trace shape: {i_train.shape}  Feature shape: {X_train.shape}")

mask_tr = d_train<=10; i_train,X_train,d_train = i_train[mask_tr],X_train[mask_tr],d_train[mask_tr]
mask_te = d_test<=10;  i_test, X_test, d_test  = i_test[mask_te], X_test[mask_te], d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  CropW: {i_train.shape[1]}  Features: {X_train.shape[1]}")

# Normalize traces per-sample (zero-mean, unit std)
i_tr_n = (i_train - i_train.mean(1,keepdims=True)) / (i_train.std(1,keepdims=True)+1e-8)
i_te_n = (i_test  - i_test.mean(1,keepdims=True))  / (i_test.std(1,keepdims=True)+1e-8)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = scaler.transform(X_test).astype(np.float32)
log_d = np.log(d_train).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1*len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

def to_t(*arrs): return [torch.tensor(a) for a in arrs]
I_tr,X_tr,y_tr = to_t(i_tr_n[tr_idx], X_tr_sc[tr_idx], log_d[tr_idx])
I_val,X_val,y_val = to_t(i_tr_n[val_idx], X_tr_sc[val_idx], log_d[val_idx])
I_tr=I_tr.to(device); X_tr=X_tr.to(device); y_tr=y_tr.to(device)
I_val=I_val.to(device); X_val=X_val.to(device); y_val=y_val.to(device)
I_te=torch.tensor(i_te_n); X_te=torch.tensor(X_te_sc)

loader = DataLoader(TensorDataset(I_tr,X_tr,y_tr), batch_size=BATCH_SIZE, shuffle=True)

model = FusionNet(feat_dim=X_train.shape[1]).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50)
crit  = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")
best_val, best_state, patience_ct = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); ep = 0.0
    for ib,xb,yb in loader:
        opt.zero_grad(); loss=crit(model(ib,xb),yb); loss.backward(); opt.step()
        ep += loss.item()*len(ib)
    ep /= len(I_tr)
    model.eval()
    with torch.no_grad(): vl = crit(model(I_val,X_val),y_val).item()
    sched.step(epoch-1)
    if vl < best_val-1e-6: best_val,best_state,patience_ct = vl,{k:v.clone() for k,v in model.state_dict().items()},0
    else: patience_ct += 1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  patience={patience_ct}")
    if patience_ct >= PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
torch.save(model.state_dict(), 'model_cnn_peakcrop.pt')
np.save('scaler_cnn_peakcrop_mean.npy', scaler.mean_)
np.save('scaler_cnn_peakcrop_scale.npy', scaler.scale_)
log("Saved model_cnn_peakcrop.pt + scaler")

def batch_predict(m, I_cpu, X_cpu, bs=512):
    n = len(I_cpu); outs = []
    for i in range(0, n, bs):
        I_b = I_cpu[i:i+bs].to(device); X_b = X_cpu[i:i+bs].to(device)
        outs.append(m(I_b, X_b).detach().cpu())
    return torch.cat(outs).numpy()

model.eval()
log_pred = batch_predict(model, I_te, X_te)
d_pred   = np.exp(log_pred)
runtime  = time.time() - t_total

ms, mf = d_test<1.0, d_test>=1.0
r2_te  = r2_score(d_test, d_pred)
mae_te = mean_absolute_error(d_test, d_pred)
summary = (
    f"Task: {LABEL}\nCrop: peak-centered W={i_train.shape[1]}\n"
    f"Features: {X_train.shape[1]}  Train: {len(d_train)}  Test: {len(d_test)}\n\n"
    f"Test R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape(d_test,d_pred):.1f}%\n"
    f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred[ms]):.4f}  MAPE={mape(d_test[ms],d_pred[ms]):.1f}%\n"
    f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred[mf]):.4f}  MAPE={mape(d_test[mf],d_pred[mf]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt"); print("Done.")
