"""
Evaluate peak-crop CNN individually and in ensemble with the 3-NN models.
Reports per-regime breakdown and all sub-combinations.
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime(d_true, d_pred):
    ms, mf = d_true<1.0, d_true>=1.0
    return (f"  d <  1  ({ms.sum():5d}): R²={r2_score(d_true[ms],d_pred[ms]):.4f}  MAPE={mape(d_true[ms],d_pred[ms]):.1f}%\n"
            f"  d >= 1  ({mf.sum():5d}): R²={r2_score(d_true[mf],d_pred[mf]):.4f}  MAPE={mape(d_true[mf],d_pred[mf]):.1f}%")

# ── Model definitions ───────────────────────────────────────────────────────
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
        self.block = nn.Sequential(nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(dim,dim), nn.BatchNorm1d(dim))
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
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU())
    def forward(self, x): return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.cnn_branch  = CNN_Branch()
        self.feat_branch = Feature_Branch(in_dim=feat_dim)
        self.head = nn.Sequential(nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace), self.feat_branch(feats)], dim=1)).squeeze(1)

def batch_predict_feat(model_fn, X_cpu, bs=512):
    n = len(X_cpu); outs = []
    for i in range(0, n, bs):
        outs.append(model_fn(X_cpu[i:i+bs].to(device)).detach().cpu())
    return torch.cat(outs).numpy()

def batch_predict_fusion(model_fn, I_cpu, X_cpu, bs=512):
    n = len(I_cpu); outs = []
    for i in range(0, n, bs):
        outs.append(model_fn(I_cpu[i:i+bs].to(device), X_cpu[i:i+bs].to(device)).detach().cpu())
    return torch.cat(outs).numpy()

# ── Load data ──────────────────────────────────────────────────────────────
t0 = time.time()
log(f"Device: {device}")
log("Loading test caches...")
X_test = np.load('cache_X_test_90pct.npy')
d_test = np.load('cache_d_test_90pct.npy')
i_test_full = np.load('cache_i_test_90pct.npy').astype(np.float32)
i_test_crop = np.load('cache_i_test_peakcrop.npy').astype(np.float32)
mask = d_test<=10
X_test,d_test,i_test_full,i_test_crop = X_test[mask],d_test[mask],i_test_full[mask],i_test_crop[mask]
log(f"Test: {len(d_test)}  Features: {X_test.shape[1]}")

i_full_n = (i_test_full - i_test_full.mean(1,keepdims=True)) / (i_test_full.std(1,keepdims=True)+1e-8)
i_crop_n = (i_test_crop - i_test_crop.mean(1,keepdims=True)) / (i_test_crop.std(1,keepdims=True)+1e-8)
X_te_t   = torch.tensor(X_test.astype(np.float32))
I_full_t = torch.tensor(i_full_n)
I_crop_t = torch.tensor(i_crop_n)

# ── Load MLP ──────────────────────────────────────────────────────────────
log("Loading MLP...")
sc_mlp = StandardScaler(); sc_mlp.mean_=np.load('scaler_mlp_cosine_mean.npy'); sc_mlp.scale_=np.load('scaler_mlp_cosine_scale.npy')
X_te_mlp = torch.tensor(sc_mlp.transform(X_test).astype(np.float32))
m_mlp = MLP_BN(X_test.shape[1]).to(device); m_mlp.load_state_dict(torch.load('model_mlp_cosine.pt', map_location=device)); m_mlp.eval()
log_pred_mlp = batch_predict_feat(lambda x: m_mlp(x), X_te_mlp)
log(f"  MLP: MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%")

# ── Load ResNet ───────────────────────────────────────────────────────────
log("Loading ResNet...")
sc_res = StandardScaler(); sc_res.mean_=np.load('scaler_resnet_cosine_mean.npy'); sc_res.scale_=np.load('scaler_resnet_cosine_scale.npy')
X_te_res = torch.tensor(sc_res.transform(X_test).astype(np.float32))
m_res = ResNetMLP(X_test.shape[1]).to(device); m_res.load_state_dict(torch.load('model_resnet_cosine.pt', map_location=device)); m_res.eval()
log_pred_res = batch_predict_feat(lambda x: m_res(x), X_te_res)
log(f"  ResNet: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%")

# ── Load CNN (full trace) ─────────────────────────────────────────────────
log("Loading CNN full-trace...")
sc_cnn = StandardScaler(); sc_cnn.mean_=np.load('scaler_cnn90pct_mean.npy'); sc_cnn.scale_=np.load('scaler_cnn90pct_scale.npy')
X_te_cnn = torch.tensor(sc_cnn.transform(X_test).astype(np.float32))
m_cnn = FusionNet(feat_dim=X_test.shape[1]).to(device); m_cnn.load_state_dict(torch.load('model_cnn_90pct.pt', map_location=device)); m_cnn.eval()
log_pred_cnn = batch_predict_fusion(lambda i,x: m_cnn(i,x), I_full_t, X_te_cnn)
log(f"  CNN full: MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%")

# ── Load CNN (peak-crop) ──────────────────────────────────────────────────
log("Loading CNN peak-crop...")
sc_crop = StandardScaler(); sc_crop.mean_=np.load('scaler_cnn_peakcrop_mean.npy'); sc_crop.scale_=np.load('scaler_cnn_peakcrop_scale.npy')
X_te_crop = torch.tensor(sc_crop.transform(X_test).astype(np.float32))
m_crop = FusionNet(feat_dim=X_test.shape[1]).to(device); m_crop.load_state_dict(torch.load('model_cnn_peakcrop.pt', map_location=device)); m_crop.eval()
log_pred_crop = batch_predict_fusion(lambda i,x: m_crop(i,x), I_crop_t, X_te_crop)
log(f"  CNN crop: MAPE={mape(d_test, np.exp(log_pred_crop)):.1f}%")

# ── Combos ────────────────────────────────────────────────────────────────
preds = {'mlp': log_pred_mlp, 'resnet': log_pred_res, 'cnn': log_pred_cnn, 'crop': log_pred_crop}
combos = [
    ('3-NN (mlp+resnet+cnn)',              ['mlp','resnet','cnn']),
    ('3-NN+crop (mlp+resnet+cnn+crop)',    ['mlp','resnet','cnn','crop']),
    ('crop only',                          ['crop']),
    ('3-NN swap (mlp+resnet+crop)',        ['mlp','resnet','crop']),
]

lines = [
    f"Task: pt_ensemble_peakcrop",
    f"Peak-crop window: {i_test_crop.shape[1]} pts",
    f"Features: {X_test.shape[1]}  Test: {len(d_test)}",
    f"",
    f"Individual Test results:",
    f"  MLP cosine:   MAPE={mape(d_test,np.exp(log_pred_mlp)):.1f}%",
    f"  ResNet cosine: MAPE={mape(d_test,np.exp(log_pred_res)):.1f}%",
    f"  CNN full:     MAPE={mape(d_test,np.exp(log_pred_cnn)):.1f}%",
    f"  CNN peak-crop: MAPE={mape(d_test,np.exp(log_pred_crop)):.1f}%",
    f"",
]
for name, keys in combos:
    lp = np.mean([preds[k] for k in keys], axis=0)
    dp = np.exp(lp)
    lines += [
        f"{name}:",
        f"  Overall: R²={r2_score(d_test,dp):.4f}  MAE={mean_absolute_error(d_test,dp):.4f}  MAPE={mape(d_test,dp):.1f}%",
        regime(d_test, dp),
        f"",
    ]
lines.append(f"Runtime: {time.time()-t0:.1f}s")
summary = "\n".join(lines)
print(f"\n{'='*60}\n{summary}\n{'='*60}")
with open('results_pt_ensemble_peakcrop.txt','w') as f: f.write(summary)
log("Saved results_pt_ensemble_peakcrop.txt"); print("Done.")
