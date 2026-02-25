"""
Ensemble: geometric mean of predictions from 3 trained models:
  1. MLP+BN+Dropout with CosineAnnealing  (model_mlp_cosine.pt)
  2. ResNet-MLP with CosineAnnealing       (model_resnet_cosine.pt)
  3. CNN fusion trained on 90k data        (model_cnn_90pct.pt)
Averaging in log-space = geometric mean in original space.
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100

# ──────────────────────────────────────────────────────────────────────────────
# Model definitions (must match the saved checkpoints)
# ──────────────────────────────────────────────────────────────────────────────
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


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim,dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.block(x))

class ResNetMLP(nn.Module):
    def __init__(self, in_dim, dim=512, n_blocks=6, dropout=0.1):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_dim,dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(dim,dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(dim,1)
    def forward(self, x): return self.head(self.blocks(self.stem(x))).squeeze(1)


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
    def __init__(self, in_dim=207):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, feat_dim=207):
        super().__init__()
        self.cnn_branch  = CNN_Branch()
        self.feat_branch = Feature_Branch(in_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256,64),  nn.ReLU(),
            nn.Linear(64,1),
        )
    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace), self.feat_branch(feats)], dim=1)).squeeze(1)


t_total = time.time()
log(f"Device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
# Load test data (90pct split — all 3 models were trained on this)
# ──────────────────────────────────────────────────────────────────────────────
log("Loading 90pct test data...")
X_test  = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test  = np.load('cache_d_test_90pct.npy')
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)
mask_te = d_test<=10
X_test, d_test, i_test = X_test[mask_te], d_test[mask_te], i_test[mask_te]
log(f"Test: {len(d_test)}")

# Per-sample trace normalisation
i_test_n = (i_test - i_test.mean(1,keepdims=True)) / (i_test.std(1,keepdims=True)+1e-8)

# ──────────────────────────────────────────────────────────────────────────────
# Model 1: MLP Cosine
# ──────────────────────────────────────────────────────────────────────────────
log("Loading MLP cosine model...")
scaler_mlp = StandardScaler()
scaler_mlp.mean_  = np.load('scaler_mlp_cosine_mean.npy')
scaler_mlp.scale_ = np.load('scaler_mlp_cosine_scale.npy')
X_te_mlp = torch.tensor(scaler_mlp.transform(X_test).astype(np.float32)).to(device)

model_mlp = MLP_BN(in_dim=X_test.shape[1]).to(device)
model_mlp.load_state_dict(torch.load('model_mlp_cosine.pt', map_location=device))
model_mlp.eval()
with torch.no_grad():
    log_pred_mlp = model_mlp(X_te_mlp).cpu().numpy()
log(f"  MLP cosine: MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# Model 2: ResNet Cosine
# ──────────────────────────────────────────────────────────────────────────────
log("Loading ResNet cosine model...")
scaler_res = StandardScaler()
scaler_res.mean_  = np.load('scaler_resnet_cosine_mean.npy')
scaler_res.scale_ = np.load('scaler_resnet_cosine_scale.npy')
X_te_res = torch.tensor(scaler_res.transform(X_test).astype(np.float32)).to(device)

model_res = ResNetMLP(in_dim=X_test.shape[1]).to(device)
model_res.load_state_dict(torch.load('model_resnet_cosine.pt', map_location=device))
model_res.eval()
with torch.no_grad():
    log_pred_res = model_res(X_te_res).cpu().numpy()
log(f"  ResNet cosine: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# Model 3: CNN 90pct fusion
# ──────────────────────────────────────────────────────────────────────────────
log("Loading CNN 90pct fusion model...")
scaler_cnn = StandardScaler()
scaler_cnn.mean_  = np.load('scaler_cnn90pct_mean.npy')
scaler_cnn.scale_ = np.load('scaler_cnn90pct_scale.npy')
X_te_cnn = torch.tensor(scaler_cnn.transform(X_test).astype(np.float32)).to(device)
i_te_cnn = torch.tensor(i_test_n).to(device)

model_cnn = FusionNet(feat_dim=X_test.shape[1]).to(device)
model_cnn.load_state_dict(torch.load('model_cnn_90pct.pt', map_location=device))
model_cnn.eval()
with torch.no_grad():
    log_pred_cnn = model_cnn(i_te_cnn, X_te_cnn).cpu().numpy()
log(f"  CNN 90pct: MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble: average in log space
# ──────────────────────────────────────────────────────────────────────────────
log_pred_ensemble = (log_pred_mlp + log_pred_res + log_pred_cnn) / 3.0
d_pred_te = np.exp(log_pred_ensemble)
runtime = time.time()-t_total

r2_te   = r2_score(d_test, d_pred_te)
mae_te  = mean_absolute_error(d_test, d_pred_te)
mape_te = mape(d_test, d_pred_te)

summary = (f"Task: pt_ensemble\n"
           f"Components: mlp_cosine + resnet_cosine + cnn_90pct (geometric mean of predictions)\n"
           f"Test: {len(d_test)}\n\n"
           f"Individual Test MAPEs:\n"
           f"  mlp_cosine:    MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%\n"
           f"  resnet_cosine: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%\n"
           f"  cnn_90pct:     MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%\n\n"
           f"Ensemble Test R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\n"
           f"Runtime: {runtime:.1f}s\n")
print(f"\n{'='*55}\n{summary}{'='*55}")
with open('results_pt_ensemble.txt','w') as f: f.write(summary)
log("Saved results_pt_ensemble.txt")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(d_test,d_pred_te,alpha=0.2,s=5,color='goldenrod')
lims=[min(d_test.min(),d_pred_te.min()),max(d_test.max(),d_pred_te.max())]
ax.plot(lims,lims,'r--',lw=1); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'Ensemble (3 models) R²={r2_te:.4f}  MAPE={mape_te:.1f}%')
plt.tight_layout(); plt.savefig('pt_ensemble.png',dpi=150)
log("Saved pt_ensemble.png"); print("Done.")
