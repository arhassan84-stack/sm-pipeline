"""
Balanced ensemble: geometric mean of predictions from:
  1. Balanced MLP+BN (model_balanced_mlp.pt)
  2. Balanced ResNet-MLP (model_balanced_resnet.pt)
  3. Original CNN fusion 90k (model_cnn_90pct.pt) — unbalanced, kept for diversity
Reports overall + per-regime (d<1 vs d>=1) metrics.
Compares against original unbalanced ensemble for reference.
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime_str(d_true, d_pred):
    m_s = d_true < 1.0; m_f = d_true >= 1.0
    return (f"  d <  1  ({m_s.sum():5d}): R²={r2_score(d_true[m_s],d_pred[m_s]):.4f}  MAPE={mape(d_true[m_s],d_pred[m_s]):.1f}%\n"
            f"  d >= 1  ({m_f.sum():5d}): R²={r2_score(d_true[m_f],d_pred[m_f]):.4f}  MAPE={mape(d_true[m_f],d_pred[m_f]):.1f}%")

# ── Model definitions ─────────────────────────────────────────────────────────
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
            nn.Dropout(dropout), nn.Linear(dim,dim), nn.BatchNorm1d(dim),
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
            nn.Linear(256,64),  nn.ReLU(), nn.Linear(64,1),
        )
    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace), self.feat_branch(feats)], dim=1)).squeeze(1)

def batch_predict(model_fn, *cpu_tensors, bs=512):
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).detach().cpu())
    return torch.cat(outs).numpy()

# ── Load test data ────────────────────────────────────────────────────────────
t_total = time.time()
log(f"Device: {device}")
log("Loading 90pct test data...")
X_test = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test = np.load('cache_d_test_90pct.npy')
i_test = np.load('cache_i_test_90pct.npy').astype(np.float32)
mask_te = d_test<=10
X_test, d_test, i_test = X_test[mask_te], d_test[mask_te], i_test[mask_te]
log(f"Test: {len(d_test)}")
i_test_n = (i_test - i_test.mean(1,keepdims=True)) / (i_test.std(1,keepdims=True)+1e-8)
X_te_t = torch.tensor(X_test)
i_te_t = torch.tensor(i_test_n)

# ── Model 1: Balanced MLP ─────────────────────────────────────────────────────
log("Loading balanced MLP...")
sc_mlp = StandardScaler()
sc_mlp.mean_  = np.load('scaler_balanced_mlp_mean.npy')
sc_mlp.scale_ = np.load('scaler_balanced_mlp_scale.npy')
X_te_mlp = torch.tensor(sc_mlp.transform(X_test).astype(np.float32))
m_mlp = MLP_BN(in_dim=X_test.shape[1]).to(device)
m_mlp.load_state_dict(torch.load('model_balanced_mlp.pt', map_location=device))
m_mlp.eval()
log_pred_mlp = batch_predict(lambda x: m_mlp(x), X_te_mlp)
log(f"  Balanced MLP: MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%")

# ── Model 2: Balanced ResNet ──────────────────────────────────────────────────
log("Loading balanced ResNet...")
sc_res = StandardScaler()
sc_res.mean_  = np.load('scaler_balanced_resnet_mean.npy')
sc_res.scale_ = np.load('scaler_balanced_resnet_scale.npy')
X_te_res = torch.tensor(sc_res.transform(X_test).astype(np.float32))
m_res = ResNetMLP(in_dim=X_test.shape[1]).to(device)
m_res.load_state_dict(torch.load('model_balanced_resnet.pt', map_location=device))
m_res.eval()
log_pred_res = batch_predict(lambda x: m_res(x), X_te_res)
log(f"  Balanced ResNet: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%")

# ── Model 3: Original CNN 90pct ───────────────────────────────────────────────
log("Loading original CNN 90pct...")
sc_cnn = StandardScaler()
sc_cnn.mean_  = np.load('scaler_cnn90pct_mean.npy')
sc_cnn.scale_ = np.load('scaler_cnn90pct_scale.npy')
X_te_cnn = torch.tensor(sc_cnn.transform(X_test).astype(np.float32))
m_cnn = FusionNet(feat_dim=X_test.shape[1]).to(device)
m_cnn.load_state_dict(torch.load('model_cnn_90pct.pt', map_location=device))
m_cnn.eval()
log_pred_cnn = batch_predict(lambda i, x: m_cnn(i, x), i_te_t, X_te_cnn)
log(f"  CNN 90pct: MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%")

# ── Ensembles ─────────────────────────────────────────────────────────────────
# 2-model: balanced MLP + balanced ResNet
log_pred_2 = (log_pred_mlp + log_pred_res) / 2.0
d_pred_2   = np.exp(log_pred_2)
# 3-model: balanced MLP + balanced ResNet + original CNN
log_pred_3 = (log_pred_mlp + log_pred_res + log_pred_cnn) / 3.0
d_pred_3   = np.exp(log_pred_3)

runtime = time.time()-t_total

summary = (f"Task: pt_balanced_ensemble\n"
           f"Components: balanced_mlp + balanced_resnet + cnn_90pct (geometric mean)\n"
           f"Test: {len(d_test)}\n\n"
           f"Individual Test MAPEs:\n"
           f"  balanced_mlp:    MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%\n"
           f"  balanced_resnet: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%\n"
           f"  cnn_90pct:       MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%\n\n"
           f"2-model ensemble (balanced_mlp + balanced_resnet):\n"
           f"  Overall: R²={r2_score(d_test,d_pred_2):.4f}  MAE={mean_absolute_error(d_test,d_pred_2):.4f}  MAPE={mape(d_test,d_pred_2):.1f}%\n"
           f"{regime_str(d_test, d_pred_2)}\n\n"
           f"3-model ensemble (+ original CNN 90pct):\n"
           f"  Overall: R²={r2_score(d_test,d_pred_3):.4f}  MAE={mean_absolute_error(d_test,d_pred_3):.4f}  MAPE={mape(d_test,d_pred_3):.1f}%\n"
           f"{regime_str(d_test, d_pred_3)}\n"
           f"Runtime: {runtime:.1f}s\n")
print(f"\n{'='*60}\n{summary}{'='*60}")
with open('results_pt_balanced_ensemble.txt','w') as f: f.write(summary)
log("Saved results_pt_balanced_ensemble.txt")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax, (d_pred, title) in zip(axes, [
    (d_pred_2, f'2-model balanced ensemble\nR²={r2_score(d_test,d_pred_2):.4f}  MAPE={mape(d_test,d_pred_2):.1f}%'),
    (d_pred_3, f'3-model ensemble (bal+CNN)\nR²={r2_score(d_test,d_pred_3):.4f}  MAPE={mape(d_test,d_pred_3):.1f}%'),
]):
    m_s = d_test<1.0; m_f = d_test>=1.0
    ax.scatter(d_test[m_s],d_pred[m_s],alpha=0.2,s=5,color='steelblue',label=f'd<1 MAPE={mape(d_test[m_s],d_pred[m_s]):.1f}%')
    ax.scatter(d_test[m_f],d_pred[m_f],alpha=0.2,s=5,color='tomato',   label=f'd>=1 MAPE={mape(d_test[m_f],d_pred[m_f]):.1f}%')
    lims=[min(d_test.min(),d_pred.min()),max(d_test.max(),d_pred.max())]
    ax.plot(lims,lims,'k--',lw=1); ax.axvline(1,color='gray',lw=0.8,ls=':'); ax.axhline(1,color='gray',lw=0.8,ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('True d'); ax.set_ylabel('Predicted d'); ax.set_title(title)
    ax.legend(markerscale=3,fontsize=9)
plt.tight_layout(); plt.savefig('pt_balanced_ensemble.png',dpi=150)
log("Saved pt_balanced_ensemble.png"); print("Done.")
