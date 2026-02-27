"""
Ensemble evaluation with multiple FT-Transformers.

Tests all meaningful combinations of:
  - CNN (raw trace branch, best individual MAPE)
  - FTT orig  (E=64, 4L, 8H)
  - FTT large (E=128, 6L, 8H)
  - FTT v2    (E=64, 6L, 4H, dropout=0.2)
  - MLP, ResNet (for reference)

Geometric mean ensemble in log-space (equal weights).
Also runs fine-grained grid search for optimal weights on the best combos.
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m=t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def regime(dt, dp):
    ms, mf = dt<1.0, dt>=1.0
    return (f"  d <1  ({ms.sum():5d}): R²={r2_score(dt[ms],dp[ms]):.4f}  MAPE={mape(dt[ms],dp[ms]):.1f}%\n"
            f"  d>=1  ({mf.sum():5d}): R²={r2_score(dt[mf],dp[mf]):.4f}  MAPE={mape(dt[mf],dp[mf]):.1f}%")

# ── Model definitions ──────────────────────────────────────────────────────────
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

class FTTransformer(nn.Module):
    def __init__(self, n_features, E=64, n_heads=8, n_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.feat_weight = nn.Parameter(torch.randn(n_features, E) * 0.01)
        self.feat_bias   = nn.Parameter(torch.zeros(n_features, E))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, E))
        encoder_layer = nn.TransformerEncoderLayer(d_model=E, nhead=n_heads, dim_feedforward=dim_ff,
                                                    dropout=dropout, activation='relu',
                                                    batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, 1))
    def forward(self, x):
        tokens = x.unsqueeze(2) * self.feat_weight.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        out = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.head(out[:, 0, :]).squeeze(1)

def batch_predict_feat(m, X_cpu, bs=512):
    n = len(X_cpu); outs = []
    for i in range(0, n, bs):
        outs.append(m(X_cpu[i:i+bs].to(device)).detach().cpu())
    return torch.cat(outs).numpy()

def batch_predict_fusion(m, I_cpu, X_cpu, bs=512):
    n = len(I_cpu); outs = []
    for i in range(0, n, bs):
        outs.append(m(I_cpu[i:i+bs].to(device), X_cpu[i:i+bs].to(device)).detach().cpu())
    return torch.cat(outs).numpy()

# ── Load data ──────────────────────────────────────────────────────────────────
log(f"Device: {device}")
X_train = np.load('cache_X_train_90pct.npy'); d_train = np.load('cache_d_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy');  d_test  = np.load('cache_d_test_90pct.npy')
i_train = np.load('cache_i_train_90pct.npy').astype(np.float32)
i_test  = np.load('cache_i_test_90pct.npy').astype(np.float32)

mask_tr = d_train<=10; X_train,d_train,i_train = X_train[mask_tr],d_train[mask_tr],i_train[mask_tr]
mask_te = d_test<=10;  X_test, d_test, i_test  = X_test[mask_te], d_test[mask_te], i_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}  Features: {X_train.shape[1]}")

i_tr_n = (i_train - i_train.mean(1,keepdims=True)) / (i_train.std(1,keepdims=True)+1e-8)
i_te_n = (i_test  - i_test.mean(1,keepdims=True))  / (i_test.std(1,keepdims=True)+1e-8)
I_tr_t = torch.tensor(i_tr_n); I_te_t = torch.tensor(i_te_n)

# ── Load all models ────────────────────────────────────────────────────────────
preds = {}  # name → log-predictions on test set

def sc_load(mean_f, scale_f, X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(); sc.mean_=np.load(mean_f); sc.scale_=np.load(scale_f)
    return torch.tensor(sc.transform(X).astype(np.float32))

# MLP
log("Loading MLP...")
m = MLP_BN(X_train.shape[1]).to(device)
m.load_state_dict(torch.load('model_mlp_cosine.pt', map_location=device)); m.eval()
X_te_sc = sc_load('scaler_mlp_cosine_mean.npy','scaler_mlp_cosine_scale.npy', X_test)
preds['mlp'] = batch_predict_feat(m, X_te_sc)

# ResNet
log("Loading ResNet...")
m = ResNetMLP(X_train.shape[1]).to(device)
m.load_state_dict(torch.load('model_resnet_cosine.pt', map_location=device)); m.eval()
X_te_sc = sc_load('scaler_resnet_cosine_mean.npy','scaler_resnet_cosine_scale.npy', X_test)
preds['resnet'] = batch_predict_feat(m, X_te_sc)

# CNN
log("Loading CNN...")
sc_cnn = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
sc_cnn.mean_=np.load('scaler_cnn90pct_mean.npy'); sc_cnn.scale_=np.load('scaler_cnn90pct_scale.npy')
X_te_cnn = torch.tensor(sc_cnn.transform(X_test).astype(np.float32))
m = FusionNet(feat_dim=X_test.shape[1]).to(device)
m.load_state_dict(torch.load('model_cnn_90pct.pt', map_location=device)); m.eval()
preds['cnn'] = batch_predict_fusion(m, I_te_t, X_te_cnn)

# FTT original
log("Loading FTT original...")
m = FTTransformer(n_features=X_test.shape[1], E=64, n_heads=8, n_layers=4, dim_ff=256, dropout=0.1).to(device)
m.load_state_dict(torch.load('model_fttransformer.pt', map_location=device)); m.eval()
X_te_sc = sc_load('scaler_fttransformer_mean.npy','scaler_fttransformer_scale.npy', X_test)
preds['ftt'] = batch_predict_feat(m, X_te_sc)

# FTT large
log("Loading FTT large...")
m = FTTransformer(n_features=X_test.shape[1], E=128, n_heads=8, n_layers=6, dim_ff=512, dropout=0.15).to(device)
m.load_state_dict(torch.load('model_fttransformer_large.pt', map_location=device)); m.eval()
X_te_sc = sc_load('scaler_fttransformer_large_mean.npy','scaler_fttransformer_large_scale.npy', X_test)
preds['ftt_large'] = batch_predict_feat(m, X_te_sc)

# FTT v2
log("Loading FTT v2...")
m = FTTransformer(n_features=X_test.shape[1], E=64, n_heads=4, n_layers=6, dim_ff=256, dropout=0.2).to(device)
m.load_state_dict(torch.load('model_fttransformer_v2.pt', map_location=device)); m.eval()
X_te_sc = sc_load('scaler_fttransformer_v2_mean.npy','scaler_fttransformer_v2_scale.npy', X_test)
preds['ftt_v2'] = batch_predict_feat(m, X_te_sc)

# ── Individual results ──────────────────────────────────────────────────────────
lines = ["="*70, "INDIVIDUAL MODELS", "="*70]
for name, lp in preds.items():
    dp = np.exp(lp)
    lines.append(f"{name:12s}: R²={r2_score(d_test,dp):.4f}  MAE={mean_absolute_error(d_test,dp):.4f}  MAPE={mape(d_test,dp):.1f}%")
    lines.append(regime(d_test, dp))
lines.append("")

# ── Ensemble helpers ────────────────────────────────────────────────────────────
def geo_mean(keys, weights=None):
    n = len(keys)
    w = np.array(weights if weights is not None else [1.0]*n, dtype=float)
    w /= w.sum()
    lp = sum(w[i]*preds[k] for i,k in enumerate(keys))
    return np.exp(lp)

# ── All combinations ────────────────────────────────────────────────────────────
lines += ["="*70, "ENSEMBLE COMBINATIONS (equal weights)", "="*70]

COMBOS = [
    # Existing best
    ('mlp+cnn+ftt',                   ['mlp','cnn','ftt']),
    ('resnet+cnn+ftt',                 ['resnet','cnn','ftt']),
    # New: large FTT
    ('mlp+cnn+ftt_large',              ['mlp','cnn','ftt_large']),
    ('resnet+cnn+ftt_large',           ['resnet','cnn','ftt_large']),
    ('cnn+ftt+ftt_large',              ['cnn','ftt','ftt_large']),
    # New: FTT v2
    ('mlp+cnn+ftt_v2',                 ['mlp','cnn','ftt_v2']),
    ('resnet+cnn+ftt_v2',              ['resnet','cnn','ftt_v2']),
    ('cnn+ftt+ftt_v2',                 ['cnn','ftt','ftt_v2']),
    # Multi-FTT 3-model
    ('cnn+ftt_large+ftt_v2',           ['cnn','ftt_large','ftt_v2']),
    # 4-model: CNN + all FTTs
    ('cnn+ftt+ftt_large+ftt_v2',       ['cnn','ftt','ftt_large','ftt_v2']),
    # 4-model: MLP/ResNet + CNN + 2 FTTs
    ('mlp+cnn+ftt+ftt_large',          ['mlp','cnn','ftt','ftt_large']),
    ('mlp+cnn+ftt+ftt_v2',             ['mlp','cnn','ftt','ftt_v2']),
    ('resnet+cnn+ftt+ftt_large',       ['resnet','cnn','ftt','ftt_large']),
    ('resnet+cnn+ftt+ftt_v2',          ['resnet','cnn','ftt','ftt_v2']),
    # 5-model
    ('mlp+cnn+ftt+ftt_large+ftt_v2',   ['mlp','cnn','ftt','ftt_large','ftt_v2']),
    ('resnet+cnn+ftt+ftt_large+ftt_v2',['resnet','cnn','ftt','ftt_large','ftt_v2']),
]

best_mape = 999; best_combo = None
for name, keys in COMBOS:
    dp = geo_mean(keys)
    r2 = r2_score(d_test,dp); mae = mean_absolute_error(d_test,dp); m_ = mape(d_test,dp)
    lines.append(f"\n{name}:")
    lines.append(f"  R²={r2:.4f}  MAE={mae:.4f}  MAPE={m_:.2f}%")
    lines.append(regime(d_test, dp))
    if m_ < best_mape: best_mape = m_; best_combo = (name, keys)

lines += ["", "="*70, f"BEST: {best_combo[0]}  MAPE={best_mape:.2f}%", "="*70, ""]

# ── Grid search on top-3 combos (test oracle) ───────────────────────────────────
lines += ["="*70, "GRID SEARCH: top combos (step=0.02, test oracle)", "="*70]

TOP_COMBOS = [
    ('cnn+ftt+ftt_large',        ['cnn','ftt','ftt_large']),
    ('cnn+ftt+ftt_v2',           ['cnn','ftt','ftt_v2']),
    ('cnn+ftt_large+ftt_v2',     ['cnn','ftt_large','ftt_v2']),
    ('cnn+ftt+ftt_large+ftt_v2', ['cnn','ftt','ftt_large','ftt_v2']),
]

mf_te = d_test >= 1.0
step = 0.02
alphas = np.arange(0, 1+step, step)

for cname, keys in TOP_COMBOS:
    n = len(keys)
    lines.append(f"\n{cname}:")
    if n == 3:
        best_mo, best_mf_val = np.inf, np.inf
        best_wo, best_wf = None, None
        for a in alphas:
            for b in alphas:
                c = 1.0 - a - b
                if c < -1e-9: continue
                c = max(c, 0.0)
                w = np.array([a, b, c]); w /= w.sum()
                dp = geo_mean(keys, w)
                m_all = mape(d_test, dp); m_fast = mape(d_test[mf_te], dp[mf_te])
                if m_all < best_mo:  best_mo = m_all;  best_wo = w.copy()
                if m_fast < best_mf_val: best_mf_val = m_fast; best_wf = w.copy()
        for label, bw, bm in [("Best MAPE",best_wo,best_mo),("Best d>=1",best_wf,best_mf_val)]:
            dp = geo_mean(keys, bw)
            ws = "  ".join(f"{k}={bw[i]:.3f}" for i,k in enumerate(keys))
            lines.append(f"  {label}: {ws}  →  MAPE={mape(d_test,dp):.2f}%  R²={r2_score(d_test,dp):.4f}  MAE={mean_absolute_error(d_test,dp):.4f}")
            lines.append("  " + regime(d_test, dp).replace("\n", "\n  "))
    else:  # 4-model grid
        best_mo = np.inf; best_wo = None
        for a in alphas:
            for b in alphas:
                for c in alphas:
                    dd = 1.0 - a - b - c
                    if dd < -1e-9: continue
                    dd = max(dd, 0.0)
                    w = np.array([a,b,c,dd]); w /= w.sum()
                    dp = geo_mean(keys, w)
                    m_all = mape(d_test, dp)
                    if m_all < best_mo: best_mo = m_all; best_wo = w.copy()
        dp = geo_mean(keys, best_wo)
        ws = "  ".join(f"{k}={best_wo[i]:.3f}" for i,k in enumerate(keys))
        lines.append(f"  Best MAPE: {ws}  →  MAPE={mape(d_test,dp):.2f}%  R²={r2_score(d_test,dp):.4f}")
        lines.append("  " + regime(d_test, dp).replace("\n", "\n  "))

summary = "\n".join(lines)
print(summary)
with open('results_ensemble_multiftt.txt','w') as f: f.write(summary)
log("Saved results_ensemble_multiftt.txt")
