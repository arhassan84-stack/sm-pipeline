"""
Phase 4 — PyTorch ResNet-style MLP with residual connections.
Architecture: 207 → Linear(512) → [ResBlock × 6] → Linear(1)
Each ResBlock: Linear(512→512) → BN → ReLU → Linear(512→512) → BN + skip
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

LABEL        = 'pt_resnet_mlp'
RESULT_FILE  = f'results_{LABEL}.txt'
BATCH_SIZE   = 512
MAX_EPOCHS   = 300
PATIENCE     = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DIM          = 512
N_BLOCKS     = 6

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape(true, pred):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetMLP(nn.Module):
    def __init__(self, in_dim, dim=512, n_blocks=6, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, dim), nn.BatchNorm1d(dim), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(dim, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(1)


t_total = time.time()

log("Loading 90% train cache...")
X_train = np.load('cache_X_train_90pct.npy')
d_train = np.load('cache_d_train_90pct.npy')
X_test  = np.load('cache_X_test_90pct.npy')
d_test  = np.load('cache_d_test_90pct.npy')

train_mask = d_train <= 10
X_train, d_train = X_train[train_mask], d_train[train_mask]
test_mask  = d_test <= 10
X_test, d_test = X_test[test_mask], d_test[test_mask]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

log_d_train = np.log(d_train)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = scaler.transform(X_test).astype(np.float32)

rng = np.random.default_rng(42)
val_idx = rng.choice(len(X_tr_sc), size=int(0.1 * len(X_tr_sc)), replace=False)
tr_idx  = np.setdiff1d(np.arange(len(X_tr_sc)), val_idx)

X_tr  = torch.tensor(X_tr_sc[tr_idx])
y_tr  = torch.tensor(log_d_train[tr_idx].astype(np.float32))
X_val = torch.tensor(X_tr_sc[val_idx])
y_val = torch.tensor(log_d_train[val_idx].astype(np.float32))
X_te  = torch.tensor(X_te_sc)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

model = ResNetMLP(in_dim=X_tr.shape[1], dim=DIM, n_blocks=N_BLOCKS)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=False)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}  dim={DIM}  blocks={N_BLOCKS}...")

best_val_loss  = float('inf')
best_state     = None
patience_count = 0
train_losses   = []
val_losses     = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_tr)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
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
with torch.no_grad():
    d_pred_train = np.exp(model(torch.tensor(X_tr_sc)).numpy())
    d_pred_test  = np.exp(model(X_te).numpy())
runtime = time.time() - t_total

r2_tr   = r2_score(d_train, d_pred_train)
mae_tr  = mean_absolute_error(d_train, d_pred_train)
mape_tr = mape(d_train, d_pred_train)
r2_te   = r2_score(d_test, d_pred_test)
mae_te  = mean_absolute_error(d_test, d_pred_test)
mape_te = mape(d_test, d_pred_test)

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: ResNet-MLP  dim={DIM}  n_blocks={N_BLOCKS}\n"
    f"Optimizer: AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}\n"
    f"Scheduler: ReduceLROnPlateau  factor=0.5  patience=10\n"
    f"Batch size: {BATCH_SIZE}  Max epochs: {MAX_EPOCHS}  Early stopping patience: {PATIENCE}\n"
    f"Params: {n_params:,}\n"
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
ax.scatter(d_test, d_pred_test, alpha=0.2, s=5, color='darkorange')
lims = [min(d_test.min(), d_pred_test.min()), max(d_test.max(), d_pred_test.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'{LABEL}  R²={r2_te:.4f}')
ax = axes[1]
ax.plot(train_losses, label='Train loss', color='darkorange')
ax.plot(val_losses,   label='Val loss',   color='purple')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.set_title('Training Curve'); ax.legend()
plt.tight_layout()
plt.savefig(f'{LABEL}.png', dpi=150)
log(f"Saved {LABEL}.png")
print("Done.")
