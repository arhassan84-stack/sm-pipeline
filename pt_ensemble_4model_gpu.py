"""
4-model ensemble: HistGBR (212 features, retrained) + MLP cosine + ResNet cosine + CNN 90pct.
HistGBR trained on CPU in this script; PyTorch models loaded from saved weights.
Geometric mean (average in log-space) across all 4 models.
Also reports all sub-combinations to show HistGBR's contribution.
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime(d_true, d_pred):
    ms, mf = d_true<1.0, d_true>=1.0
    return (f"  d <  1  ({ms.sum():5d}): R²={r2_score(d_true[ms],d_pred[ms]):.4f}  MAPE={mape(d_true[ms],d_pred[ms]):.1f}%\n"
            f"  d >= 1  ({mf.sum():5d}): R²={r2_score(d_true[mf],d_pred[mf]):.4f}  MAPE={mape(d_true[mf],d_pred[mf]):.1f}%")

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

def batch_predict(model_fn, *cpu_tensors, bs=512):
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).detach().cpu())
    return torch.cat(outs).numpy()

# ── Load data ─────────────────────────────────────────────────────────────────
t_total = time.time()
log(f"Device: {device}")
log("Loading 90pct train + test caches (283 features)...")
X_train = np.load('cache_X_train_90pct.npy');  d_train = np.load('cache_d_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy');   d_test  = np.load('cache_d_test_90pct.npy')
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)

mask_tr = d_train<=10; X_train, d_train = X_train[mask_tr], d_train[mask_tr]
mask_te = d_test<=10;  X_test,  d_test  = X_test[mask_te],  d_test[mask_te]
i_test  = i_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Features: {X_train.shape[1]}")
assert X_train.shape[1] == 283, f"Expected 283 features, got {X_train.shape[1]}"

i_test_n = (i_test - i_test.mean(1,keepdims=True)) / (i_test.std(1,keepdims=True)+1e-8)
X_te_t   = torch.tensor(X_test.astype(np.float32))
i_te_t   = torch.tensor(i_test_n)

# ── Model 1: HistGBR (212 features, retrained) ────────────────────────────────
log("Training HistGBR on 212 features (CPU)...")
log_d_train = np.log(d_train)
hgbr = HistGradientBoostingRegressor(
    max_iter=800, max_depth=5, learning_rate=0.08,
    min_samples_leaf=80, max_leaf_nodes=127, l2_regularization=0.0,
    random_state=42, verbose=0,
)
t0 = time.time()
hgbr.fit(X_train, log_d_train)
log(f"  HistGBR trained in {time.time()-t0:.1f}s")
log_pred_hgbr = hgbr.predict(X_test)
log(f"  HistGBR: MAPE={mape(d_test, np.exp(log_pred_hgbr)):.1f}%  "
    f"R²={r2_score(d_test, np.exp(log_pred_hgbr)):.4f}")

# ── Model 2: MLP cosine (212 features) ───────────────────────────────────────
log("Loading MLP cosine model...")
sc_mlp = StandardScaler()
sc_mlp.mean_  = np.load('scaler_mlp_cosine_mean.npy')
sc_mlp.scale_ = np.load('scaler_mlp_cosine_scale.npy')
X_te_mlp = torch.tensor(sc_mlp.transform(X_test).astype(np.float32))
m_mlp = MLP_BN(in_dim=X_test.shape[1]).to(device)
m_mlp.load_state_dict(torch.load('model_mlp_cosine.pt', map_location=device))
m_mlp.eval()
log_pred_mlp = batch_predict(lambda x: m_mlp(x), X_te_mlp)
log(f"  MLP cosine: MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%")

# ── Model 3: ResNet cosine (212 features) ────────────────────────────────────
log("Loading ResNet cosine model...")
sc_res = StandardScaler()
sc_res.mean_  = np.load('scaler_resnet_cosine_mean.npy')
sc_res.scale_ = np.load('scaler_resnet_cosine_scale.npy')
X_te_res = torch.tensor(sc_res.transform(X_test).astype(np.float32))
m_res = ResNetMLP(in_dim=X_test.shape[1]).to(device)
m_res.load_state_dict(torch.load('model_resnet_cosine.pt', map_location=device))
m_res.eval()
log_pred_res = batch_predict(lambda x: m_res(x), X_te_res)
log(f"  ResNet cosine: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%")

# ── Model 4: CNN 90pct (212 features) ─────────────────────────────────────────
log("Loading CNN 90pct fusion model...")
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
log("Computing ensembles...")
preds = {
    'hgbr':   log_pred_hgbr,
    'mlp':    log_pred_mlp,
    'resnet': log_pred_res,
    'cnn':    log_pred_cnn,
}

combos = [
    ('3-NN (mlp+resnet+cnn)',             ['mlp','resnet','cnn']),
    ('4-model (hgbr+mlp+resnet+cnn)',     ['hgbr','mlp','resnet','cnn']),
    ('2-model (hgbr+cnn)',                ['hgbr','cnn']),
    ('2-model (hgbr+mlp)',                ['hgbr','mlp']),
]

runtime = time.time() - t_total

lines = [
    f"Task: pt_ensemble_4model",
    f"HistGBR params: max_iter=800, max_depth=5, lr=0.08, min_samples_leaf=80, max_leaf_nodes=127",
    f"Features: {X_train.shape[1]}  Train: {len(d_train)}  Test: {len(d_test)}",
    f"",
    f"Individual Test results:",
    f"  HistGBR (283f):  R²={r2_score(d_test,np.exp(log_pred_hgbr)):.4f}  MAPE={mape(d_test,np.exp(log_pred_hgbr)):.1f}%",
    f"  MLP cosine:      R²={r2_score(d_test,np.exp(log_pred_mlp)):.4f}   MAPE={mape(d_test,np.exp(log_pred_mlp)):.1f}%",
    f"  ResNet cosine:   R²={r2_score(d_test,np.exp(log_pred_res)):.4f}   MAPE={mape(d_test,np.exp(log_pred_res)):.1f}%",
    f"  CNN 90pct:       R²={r2_score(d_test,np.exp(log_pred_cnn)):.4f}   MAPE={mape(d_test,np.exp(log_pred_cnn)):.1f}%",
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

lines.append(f"Runtime: {runtime:.1f}s")
summary = "\n".join(lines)
print(f"\n{'='*60}\n{summary}\n{'='*60}")
with open('results_pt_ensemble_4model.txt','w') as f: f.write(summary)
log("Saved results_pt_ensemble_4model.txt")

# ── Plot: 4-model ensemble ────────────────────────────────────────────────────
lp_4 = np.mean([preds[k] for k in ['hgbr','mlp','resnet','cnn']], axis=0)
dp_4 = np.exp(lp_4)
ms, mf = d_test<1.0, d_test>=1.0

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12,5))
ax = axes[0]
ax.scatter(d_test[ms], dp_4[ms], alpha=0.2, s=5, color='steelblue',
           label=f"d<1  MAPE={mape(d_test[ms],dp_4[ms]):.1f}%")
ax.scatter(d_test[mf], dp_4[mf], alpha=0.2, s=5, color='tomato',
           label=f"d>=1 MAPE={mape(d_test[mf],dp_4[mf]):.1f}%")
lims = [min(d_test.min(),dp_4.min()), max(d_test.max(),dp_4.max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.axvline(1, color='gray', lw=0.8, ls=':'); ax.axhline(1, color='gray', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'4-model ensemble (HistGBR+MLP+ResNet+CNN)\n'
             f'R²={r2_score(d_test,dp_4):.4f}  MAPE={mape(d_test,dp_4):.1f}%')
ax.legend(markerscale=3, fontsize=9)

# Per-model comparison bar chart
ax2 = axes[1]
model_names = ['HistGBR', 'MLP', 'ResNet', 'CNN', '3-NN\nensemble', '4-model\nensemble']
mapes = [
    mape(d_test, np.exp(log_pred_hgbr)),
    mape(d_test, np.exp(log_pred_mlp)),
    mape(d_test, np.exp(log_pred_res)),
    mape(d_test, np.exp(log_pred_cnn)),
    mape(d_test, np.exp(np.mean([preds[k] for k in ['mlp','resnet','cnn']], axis=0))),
    mape(d_test, dp_4),
]
colors = ['#4e8abf','#4e8abf','#4e8abf','#4e8abf','#e88030','#2ca02c']
bars = ax2.bar(model_names, mapes, color=colors)
ax2.set_ylabel('Test MAPE (%)'); ax2.set_title('MAPE by model')
for bar, v in zip(bars, mapes):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{v:.1f}%',
             ha='center', va='bottom', fontsize=9)
ax2.set_ylim(0, max(mapes)*1.15)
plt.tight_layout(); plt.savefig('pt_ensemble_4model.png', dpi=150)
log("Saved pt_ensemble_4model.png"); print("Done.")
