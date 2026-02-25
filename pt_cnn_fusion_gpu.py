"""
Phase 5 — PyTorch 1D CNN on raw traces + engineered feature fusion.
GPU-enabled: automatically uses CUDA if available.
"""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL        = 'pt_cnn_fusion'
RESULT_FILE  = f'results_{LABEL}.txt'
BATCH_SIZE   = 256
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 5e-4
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape(true, pred):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100


class CNN_Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(2)


class Feature_Branch(nn.Module):
    def __init__(self, in_dim=207):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, feat_dim=207):
        super().__init__()
        self.cnn_branch  = CNN_Branch()
        self.feat_branch = Feature_Branch(in_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        return self.head(torch.cat([self.cnn_branch(trace), self.feat_branch(feats)], dim=1)).squeeze(1)


t_total = time.time()
log(f"Device: {device}")

log("Loading raw traces + features...")
i_train_50k = np.load('cache_i_train.npy').astype(np.float32)
X_train_50k = np.load('cache_X_train_psd.npy').astype(np.float32)
d_train_50k = np.load('cache_d_train.npy')

i_test_5k   = np.load('cache_i_test.npy').astype(np.float32)
X_test_5k   = np.load('cache_X_test_psd.npy').astype(np.float32)
d_test_5k   = np.load('cache_d_test.npy')

train_mask = d_train_50k <= 10
i_train = i_train_50k[train_mask]
X_train = X_train_50k[train_mask]
d_train = d_train_50k[train_mask]

test_mask = d_test_5k <= 10
i_test  = i_test_5k[test_mask]
X_test  = X_test_5k[test_mask]
d_test  = d_test_5k[test_mask]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

# Per-sample normalisation of raw traces
i_train_n = (i_train - i_train.mean(1, keepdims=True)) / (i_train.std(1, keepdims=True) + 1e-8)
i_test_n  = (i_test  - i_test.mean(1,  keepdims=True)) / (i_test.std(1,  keepdims=True) + 1e-8)

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)

log_d_train = np.log(d_train).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(d_train)), val_idx)

i_tr   = torch.tensor(i_train_n[tr_idx]).to(device)
X_tr   = torch.tensor(X_tr_sc[tr_idx]).to(device)
y_tr   = torch.tensor(log_d_train[tr_idx]).to(device)
i_val  = torch.tensor(i_train_n[val_idx]).to(device)
X_val  = torch.tensor(X_tr_sc[val_idx]).to(device)
y_val  = torch.tensor(log_d_train[val_idx]).to(device)
i_te   = torch.tensor(i_test_n).to(device)
X_te   = torch.tensor(X_te_sc).to(device)

train_loader = DataLoader(TensorDataset(i_tr, X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

model     = FusionNet(feat_dim=X_tr.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}...")

best_val_loss  = float('inf')
best_state     = None
patience_count = 0
train_losses   = []
val_losses     = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for ib, xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(ib, xb), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(ib)
    ep_loss /= len(i_tr)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(i_val, X_val), y_val).item()
    scheduler.step(val_loss)
    train_losses.append(ep_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss  = val_loss
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep_loss:.5f}  val={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  patience={patience_count}")

    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done — best val_loss={best_val_loss:.5f}")

model.eval()
i_tr_all = torch.tensor(i_train_n).to(device)
X_tr_all = torch.tensor(X_tr_sc).to(device)
with torch.no_grad():
    d_pred_train = np.exp(model(i_tr_all, X_tr_all).cpu().numpy())
    d_pred_test  = np.exp(model(i_te, X_te).cpu().numpy())
runtime = time.time() - t_total

r2_tr   = r2_score(d_train, d_pred_train)
mae_tr  = mean_absolute_error(d_train, d_pred_train)
mape_tr = mape(d_train, d_pred_train)
r2_te   = r2_score(d_test, d_pred_test)
mae_te  = mean_absolute_error(d_test, d_pred_test)
mape_te = mape(d_test, d_pred_test)

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: CNN (4096 raw trace) + FC (207 features) → fusion head\n"
    f"Optimizer: AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}\n"
    f"Batch size: {BATCH_SIZE}  Max epochs: {MAX_EPOCHS}  Early stopping patience: {PATIENCE}\n"
    f"Device: {device}  Params: {n_params:,}\n"
    f"Train size: {len(d_train)}  Test size: {len(d_test)}\n\n"
    f"Train  R²={r2_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape_tr:.1f}%\n"
    f"Test   R²={r2_te:.4f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'='*55}\n{summary}{'='*55}")
with open(RESULT_FILE, 'w') as fh:
    fh.write(summary)
log(f"Saved {RESULT_FILE}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(d_test, d_pred_test, alpha=0.2, s=5, color='seagreen')
lims = [min(d_test.min(), d_pred_test.min()), max(d_test.max(), d_pred_test.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'{LABEL}  R²={r2_te:.4f}')
ax = axes[1]
ax.plot(train_losses, label='Train loss', color='seagreen')
ax.plot(val_losses,   label='Val loss',   color='crimson')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.set_title('Training Curve'); ax.legend()
plt.tight_layout()
plt.savefig(f'{LABEL}.png', dpi=150)
log(f"Saved {LABEL}.png")
print("Done.")
