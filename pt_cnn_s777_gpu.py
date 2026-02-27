"""CNN fusion (90k) with seed=777 — identical architecture to model_cnn_90pct, different init."""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL='pt_cnn_s777'; SEED=777; BATCH_SIZE=256; EVAL_BATCH=512
MAX_EPOCHS=200; PATIENCE=25; LR=5e-4; WEIGHT_DECAY=1e-4; T0=30

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

def batch_predict(model_fn, *cpu_tensors, bs=512):
    n=len(cpu_tensors[0]); outs=[]
    for i in range(0,n,bs):
        batch=[t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)

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
        self.net = nn.Sequential(nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU())
    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.cnn_branch=CNN_Branch(); self.feat_branch=Feature_Branch(feat_dim)
        self.head=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.1),
                                nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace),self.feat_branch(feats)],dim=1)).squeeze(1)

t_total=time.time(); log(f"Device: {device}  Seed: {SEED}")
i_train=np.load('cache_i_train_90pct.npy').astype(np.float32)
X_train=np.load('cache_X_train_90pct.npy').astype(np.float32)
d_train=np.load('cache_d_train_90pct.npy')
i_test =np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test =np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test =np.load('cache_d_test_90pct.npy')

mask_tr=d_train<=10; i_train,X_train,d_train=i_train[mask_tr],X_train[mask_tr],d_train[mask_tr]
mask_te=d_test<=10;  i_test, X_test, d_test =i_test[mask_te], X_test[mask_te], d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Features: {X_train.shape[1]}")

i_tr_n=(i_train-i_train.mean(1,keepdims=True))/(i_train.std(1,keepdims=True)+1e-8)
i_te_n=(i_test -i_test.mean(1, keepdims=True))/(i_test.std(1, keepdims=True)+1e-8)
sc=StandardScaler(); X_tr_sc=sc.fit_transform(X_train).astype(np.float32); X_te_sc=sc.transform(X_test).astype(np.float32)
log_d=np.log(d_train).astype(np.float32)

rng=np.random.default_rng(SEED)
val_idx=rng.choice(len(d_train),size=int(0.1*len(d_train)),replace=False)
tr_idx =np.setdiff1d(np.arange(len(d_train)),val_idx)

i_tr=torch.tensor(i_tr_n[tr_idx]); X_tr=torch.tensor(X_tr_sc[tr_idx]); y_tr=torch.tensor(log_d[tr_idx])
i_val=torch.tensor(i_tr_n[val_idx]);X_val=torch.tensor(X_tr_sc[val_idx]);y_val=torch.tensor(log_d[val_idx])
i_te=torch.tensor(i_te_n);          X_te=torch.tensor(X_te_sc)

loader=DataLoader(TensorDataset(i_tr,X_tr,y_tr),batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
model=FusionNet(feat_dim=X_tr.shape[1]).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=T0)
crit=nn.MSELoss()
n_params=sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")

best_val,best_state,patience_count=float('inf'),None,0
for epoch in range(1,MAX_EPOCHS+1):
    model.train(); ep=0.0
    for ib,xb,yb in loader:
        ib,xb,yb=ib.to(device),xb.to(device),yb.to(device)
        opt.zero_grad(); loss=crit(model(ib,xb),yb); loss.backward(); opt.step()
        ep+=loss.item()*len(ib)
    ep/=len(i_tr)
    model.eval()
    with torch.no_grad():
        vl=crit(batch_predict(lambda i,x:model(i,x),i_val,X_val,bs=EVAL_BATCH),y_val).item()
    sched.step(epoch-1)
    if vl<best_val-1e-6: best_val,best_state,patience_count=vl,{k:v.clone() for k,v in model.state_dict().items()},0
    else: patience_count+=1
    if epoch%10==0: log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count>=PATIENCE: log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
torch.save(model.state_dict(),'model_cnn_s777.pt')
np.save('scaler_cnn_s777_mean.npy',sc.mean_); np.save('scaler_cnn_s777_scale.npy',sc.scale_)
log("Saved model_cnn_s777.pt + scaler")

model.eval()
with torch.no_grad():
    d_pred_te=np.exp(batch_predict(lambda i,x:model(i,x),i_te,X_te,bs=EVAL_BATCH).numpy())
runtime=time.time()-t_total
ms,mf=d_test<1.0,d_test>=1.0
r2_te=r2_score(d_test,d_pred_te); mae_te=mean_absolute_error(d_test,d_pred_te); mape_te=mape(d_test,d_pred_te)
summary=(f"Task: {LABEL}\nSeed: {SEED}\nTest R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\n"
         f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms],d_pred_te[ms]):.4f}  MAPE={mape(d_test[ms],d_pred_te[ms]):.1f}%\n"
         f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf],d_pred_te[mf]):.4f}  MAPE={mape(d_test[mf],d_pred_te[mf]):.1f}%\n"
         f"Runtime: {runtime:.1f}s\n")
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(f'results_{LABEL}.txt','w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")
