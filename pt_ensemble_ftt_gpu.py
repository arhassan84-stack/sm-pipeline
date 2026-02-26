"""
Ensemble evaluation: FT-Transformer vs 3-NN vs 3-NN+FTT vs FTT replacing each member.
Reports per-regime breakdown for all combinations.
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

def batch_predict(model_fn, X_cpu, bs=512):
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
X_test = np.load('cache_X_test_90pct.npy')
d_test = np.load('cache_d_test_90pct.npy')
i_test = np.load('cache_i_test_90pct.npy').astype(np.float32)
mask = d_test<=10
X_test,d_test,i_test = X_test[mask],d_test[mask],i_test[mask]
i_te_n = (i_test - i_test.mean(1,keepdims=True)) / (i_test.std(1,keepdims=True)+1e-8)
log(f"Test: {len(d_test)}  Features: {X_test.shape[1]}")

# MLP
sc_mlp=StandardScaler(); sc_mlp.mean_=np.load('scaler_mlp_cosine_mean.npy'); sc_mlp.scale_=np.load('scaler_mlp_cosine_scale.npy')
X_te_mlp=torch.tensor(sc_mlp.transform(X_test).astype(np.float32))
m_mlp=MLP_BN(X_test.shape[1]).to(device); m_mlp.load_state_dict(torch.load('model_mlp_cosine.pt',map_location=device)); m_mlp.eval()
log_pred_mlp=batch_predict(lambda x: m_mlp(x), X_te_mlp)
log(f"  MLP: MAPE={mape(d_test, np.exp(log_pred_mlp)):.1f}%")

# ResNet
sc_res=StandardScaler(); sc_res.mean_=np.load('scaler_resnet_cosine_mean.npy'); sc_res.scale_=np.load('scaler_resnet_cosine_scale.npy')
X_te_res=torch.tensor(sc_res.transform(X_test).astype(np.float32))
m_res=ResNetMLP(X_test.shape[1]).to(device); m_res.load_state_dict(torch.load('model_resnet_cosine.pt',map_location=device)); m_res.eval()
log_pred_res=batch_predict(lambda x: m_res(x), X_te_res)
log(f"  ResNet: MAPE={mape(d_test, np.exp(log_pred_res)):.1f}%")

# CNN
sc_cnn=StandardScaler(); sc_cnn.mean_=np.load('scaler_cnn90pct_mean.npy'); sc_cnn.scale_=np.load('scaler_cnn90pct_scale.npy')
X_te_cnn=torch.tensor(sc_cnn.transform(X_test).astype(np.float32)); I_te=torch.tensor(i_te_n)
m_cnn=FusionNet(feat_dim=X_test.shape[1]).to(device); m_cnn.load_state_dict(torch.load('model_cnn_90pct.pt',map_location=device)); m_cnn.eval()
log_pred_cnn=batch_predict_fusion(lambda i,x: m_cnn(i,x), I_te, X_te_cnn)
log(f"  CNN: MAPE={mape(d_test, np.exp(log_pred_cnn)):.1f}%")

# FT-Transformer
sc_ftt=StandardScaler(); sc_ftt.mean_=np.load('scaler_fttransformer_mean.npy'); sc_ftt.scale_=np.load('scaler_fttransformer_scale.npy')
X_te_ftt=torch.tensor(sc_ftt.transform(X_test).astype(np.float32))
m_ftt=FTTransformer(n_features=X_test.shape[1]).to(device); m_ftt.load_state_dict(torch.load('model_fttransformer.pt',map_location=device)); m_ftt.eval()
log_pred_ftt=batch_predict(lambda x: m_ftt(x), X_te_ftt)
log(f"  FTT: MAPE={mape(d_test, np.exp(log_pred_ftt)):.1f}%")

# ── Ensembles ──────────────────────────────────────────────────────────────
preds = {'mlp': log_pred_mlp, 'resnet': log_pred_res, 'cnn': log_pred_cnn, 'ftt': log_pred_ftt}
combos = [
    ('3-NN (mlp+resnet+cnn)',              ['mlp','resnet','cnn']),
    ('FTT only',                           ['ftt']),
    ('4-model (mlp+resnet+cnn+ftt)',       ['mlp','resnet','cnn','ftt']),
    ('3-model swap: mlp+resnet+ftt',       ['mlp','resnet','ftt']),
    ('3-model swap: mlp+cnn+ftt',          ['mlp','cnn','ftt']),
    ('3-model swap: resnet+cnn+ftt',       ['resnet','cnn','ftt']),
]
lines = [
    f"Task: pt_ensemble_ftt",
    f"Features: {X_test.shape[1]}  Test: {len(d_test)}",
    f"",
    f"Individual results:",
    f"  MLP cosine:    MAPE={mape(d_test,np.exp(log_pred_mlp)):.1f}%  R²={r2_score(d_test,np.exp(log_pred_mlp)):.4f}",
    f"  ResNet cosine: MAPE={mape(d_test,np.exp(log_pred_res)):.1f}%  R²={r2_score(d_test,np.exp(log_pred_res)):.4f}",
    f"  CNN 90pct:     MAPE={mape(d_test,np.exp(log_pred_cnn)):.1f}%  R²={r2_score(d_test,np.exp(log_pred_cnn)):.4f}",
    f"  FT-Transformer:MAPE={mape(d_test,np.exp(log_pred_ftt)):.1f}%  R²={r2_score(d_test,np.exp(log_pred_ftt)):.4f}",
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
with open('results_pt_ensemble_ftt.txt','w') as f: f.write(summary)
log("Saved results_pt_ensemble_ftt.txt"); print("Done.")
