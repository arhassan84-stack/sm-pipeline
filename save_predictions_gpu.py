"""
Run all 4 models (MLP, ResNet, CNN, FTT) on both train and test sets
and save raw log-predictions + true d values to disk.
Used for weight optimization via scipy.optimize locally (no test leakage).
"""
import numpy as np, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
X_tr_t = torch.tensor(X_train.astype(np.float32))
X_te_t = torch.tensor(X_test.astype(np.float32))

np.save('pred_d_train.npy', d_train)
np.save('pred_d_test.npy',  d_test)

# ── MLP ────────────────────────────────────────────────────────────────────
log("MLP...")
sc = StandardScaler(); sc.mean_=np.load('scaler_mlp_cosine_mean.npy'); sc.scale_=np.load('scaler_mlp_cosine_scale.npy')
X_tr_sc = torch.tensor(sc.transform(X_train).astype(np.float32))
X_te_sc = torch.tensor(sc.transform(X_test).astype(np.float32))
m = MLP_BN(X_train.shape[1]).to(device); m.load_state_dict(torch.load('model_mlp_cosine.pt', map_location=device)); m.eval()
np.save('pred_logd_mlp_train.npy', batch_predict_feat(lambda x: m(x), X_tr_sc))
np.save('pred_logd_mlp_test.npy',  batch_predict_feat(lambda x: m(x), X_te_sc))
log("  MLP done")

# ── ResNet ─────────────────────────────────────────────────────────────────
log("ResNet...")
sc = StandardScaler(); sc.mean_=np.load('scaler_resnet_cosine_mean.npy'); sc.scale_=np.load('scaler_resnet_cosine_scale.npy')
X_tr_sc = torch.tensor(sc.transform(X_train).astype(np.float32))
X_te_sc = torch.tensor(sc.transform(X_test).astype(np.float32))
m = ResNetMLP(X_train.shape[1]).to(device); m.load_state_dict(torch.load('model_resnet_cosine.pt', map_location=device)); m.eval()
np.save('pred_logd_resnet_train.npy', batch_predict_feat(lambda x: m(x), X_tr_sc))
np.save('pred_logd_resnet_test.npy',  batch_predict_feat(lambda x: m(x), X_te_sc))
log("  ResNet done")

# ── CNN ────────────────────────────────────────────────────────────────────
log("CNN...")
sc = StandardScaler(); sc.mean_=np.load('scaler_cnn90pct_mean.npy'); sc.scale_=np.load('scaler_cnn90pct_scale.npy')
X_tr_cnn = torch.tensor(sc.transform(X_train).astype(np.float32))
X_te_cnn = torch.tensor(sc.transform(X_test).astype(np.float32))
m = FusionNet(feat_dim=X_train.shape[1]).to(device); m.load_state_dict(torch.load('model_cnn_90pct.pt', map_location=device)); m.eval()
np.save('pred_logd_cnn_train.npy', batch_predict_fusion(lambda i,x: m(i,x), I_tr_t, X_tr_cnn))
np.save('pred_logd_cnn_test.npy',  batch_predict_fusion(lambda i,x: m(i,x), I_te_t, X_te_cnn))
log("  CNN done")

# ── FT-Transformer ─────────────────────────────────────────────────────────
log("FTT...")
sc = StandardScaler(); sc.mean_=np.load('scaler_fttransformer_mean.npy'); sc.scale_=np.load('scaler_fttransformer_scale.npy')
X_tr_sc = torch.tensor(sc.transform(X_train).astype(np.float32))
X_te_sc = torch.tensor(sc.transform(X_test).astype(np.float32))
m = FTTransformer(n_features=X_train.shape[1]).to(device); m.load_state_dict(torch.load('model_fttransformer.pt', map_location=device)); m.eval()
np.save('pred_logd_ftt_train.npy', batch_predict_feat(lambda x: m(x), X_tr_sc))
np.save('pred_logd_ftt_test.npy',  batch_predict_feat(lambda x: m(x), X_te_sc))
log("  FTT done")

log(f"All predictions saved. Runtime: {time.time()-t0:.1f}s")
log("Files: pred_d_train/test.npy, pred_logd_mlp/resnet/cnn/ftt_train/test.npy")
