"""
Router evaluation: trains a binary classifier router, then evaluates 3 routing strategies:
  Option A+D: Soft gating — P(slow) × pred_slow + P(fast) × pred_fast  (in log space)
              Specialists trained with overlap: slow on d<=1.5, fast on d>=0.5
  Option B:   Physical router — threshold on half-decay ACF lag (feature index 9)
  Option C:   General model router — use existing MLP cosine prediction to decide regime

All evaluated on full test set with per-regime (d<1 vs d>=1) breakdown.
"""
import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t>0; return np.mean(np.abs((p[m]-t[m])/t[m]))*100
def report(label, d_true, d_pred):
    m1 = d_true < 1; m2 = d_true >= 1
    lines = [
        f"  Overall   ({len(d_true):5d}): R²={r2_score(d_true,d_pred):.4f}  MAE={mean_absolute_error(d_true,d_pred):.4f}  MAPE={mape(d_true,d_pred):.1f}%",
        f"  d <  1    ({m1.sum():5d}): R²={r2_score(d_true[m1],d_pred[m1]):.4f}  MAE={mean_absolute_error(d_true[m1],d_pred[m1]):.4f}  MAPE={mape(d_true[m1],d_pred[m1]):.1f}%",
        f"  d >= 1    ({m2.sum():5d}): R²={r2_score(d_true[m2],d_pred[m2]):.4f}  MAE={mean_absolute_error(d_true[m2],d_pred[m2]):.4f}  MAPE={mape(d_true[m2],d_pred[m2]):.1f}%",
    ]
    print(f"\n{label}"); [print(l) for l in lines]
    return "\n".join([label]+lines)

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

class RouterMLP(nn.Module):
    """Binary classifier: outputs P(d < 1) = P(slow)."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64),     nn.ReLU(),
            nn.Linear(64, 1),       nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(1)

t_total = time.time()
log(f"Device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────────
log("Loading 90% cache...")
X_train_all = np.load('cache_X_train_90pct.npy'); d_train_all = np.load('cache_d_train_90pct.npy')
X_test_all  = np.load('cache_X_test_90pct.npy');  d_test_all  = np.load('cache_d_test_90pct.npy')

mask_tr = (d_train_all > 0) & (d_train_all <= 10)
mask_te = (d_test_all  > 0) & (d_test_all  <= 10)
X_train, d_train = X_train_all[mask_tr], d_train_all[mask_tr]
X_test,  d_test  = X_test_all[mask_te],  d_test_all[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

# ── Train router classifier ────────────────────────────────────────────────────
log("Training router classifier (d<1 vs d>=1)...")
ROUTER_LR = 1e-3; ROUTER_EPOCHS = 200; ROUTER_PATIENCE = 25; ROUTER_BS = 1024

scaler_r = StandardScaler()
X_tr_r = scaler_r.fit_transform(X_train).astype(np.float32)
X_te_r = scaler_r.transform(X_test).astype(np.float32)
y_cls_tr = (d_train < 1.0).astype(np.float32)

rng = np.random.default_rng(42)
val_idx_r = rng.choice(len(X_tr_r), size=int(0.1*len(X_tr_r)), replace=False)
tr_idx_r  = np.setdiff1d(np.arange(len(X_tr_r)), val_idx_r)

Xr_tr = torch.tensor(X_tr_r[tr_idx_r]).to(device); yr_tr = torch.tensor(y_cls_tr[tr_idx_r]).to(device)
Xr_val= torch.tensor(X_tr_r[val_idx_r]).to(device); yr_val= torch.tensor(y_cls_tr[val_idx_r]).to(device)
Xr_te = torch.tensor(X_te_r).to(device)
rloader = DataLoader(TensorDataset(Xr_tr,yr_tr), batch_size=ROUTER_BS, shuffle=True)

router = RouterMLP(X_train.shape[1]).to(device)
r_opt  = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR, weight_decay=1e-4)
r_sched= torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(r_opt, T_0=50)
r_crit = nn.BCELoss()

best_rv, best_rstate, r_patience = float('inf'), None, 0
for epoch in range(1, ROUTER_EPOCHS+1):
    router.train(); ep=0.0
    for xb,yb in rloader:
        r_opt.zero_grad(); loss=r_crit(router(xb),yb); loss.backward(); r_opt.step()
        ep += loss.item()*len(xb)
    router.eval()
    with torch.no_grad(): vl = r_crit(router(Xr_val),yr_val).item()
    r_sched.step(epoch-1)
    if vl < best_rv-1e-6: best_rv,best_rstate,r_patience = vl,{k:v.clone() for k,v in router.state_dict().items()},0
    else: r_patience+=1
    if epoch%20==0: log(f"  Router epoch {epoch:3d}  val={vl:.5f}  patience={r_patience}")
    if r_patience>=ROUTER_PATIENCE: log(f"  Router early stop at epoch {epoch}"); break

router.load_state_dict(best_rstate)
torch.save(router.state_dict(), 'model_router.pt')
np.save('scaler_router_mean.npy', scaler_r.mean_)
np.save('scaler_router_scale.npy', scaler_r.scale_)

router.eval()
with torch.no_grad():
    p_slow_te = router(Xr_te).cpu().numpy()  # P(d < 1) for each test sample
y_cls_te = (d_test < 1.0).astype(float)
router_acc = accuracy_score(y_cls_te, p_slow_te > 0.5)
log(f"Router trained — accuracy={router_acc*100:.2f}%  (threshold=0.5)")

# ── Load specialist models ─────────────────────────────────────────────────────
log("Loading specialist models...")
def load_mlp_with_scaler(model_file, mean_file, scale_file, in_dim):
    m = MLP_BN(in_dim).to(device); m.load_state_dict(torch.load(model_file, map_location=device)); m.eval()
    sc = StandardScaler(); sc.mean_ = np.load(mean_file); sc.scale_ = np.load(scale_file)
    return m, sc

model_slow, sc_slow = load_mlp_with_scaler('model_specialist_slow.pt','scaler_specialist_slow_mean.npy','scaler_specialist_slow_scale.npy', X_test.shape[1])
model_fast, sc_fast = load_mlp_with_scaler('model_specialist_fast.pt','scaler_specialist_fast_mean.npy','scaler_specialist_fast_scale.npy', X_test.shape[1])
model_gen,  sc_gen  = load_mlp_with_scaler('model_mlp_cosine.pt',     'scaler_mlp_cosine_mean.npy',     'scaler_mlp_cosine_scale.npy',      X_test.shape[1])
log("  Loaded slow, fast, general specialists.")

# Precompute test set predictions from each specialist
with torch.no_grad():
    log_pred_slow = model_slow(torch.tensor(sc_slow.transform(X_test).astype(np.float32)).to(device)).cpu().numpy()
    log_pred_fast = model_fast(torch.tensor(sc_fast.transform(X_test).astype(np.float32)).to(device)).cpu().numpy()
    log_pred_gen  = model_gen( torch.tensor(sc_gen.transform(X_test).astype(np.float32)).to(device)).cpu().numpy()

# ── Option A+D: Soft gating ────────────────────────────────────────────────────
log("\n=== Option A+D: Soft gating (learned router, probability blending) ===")
log_pred_AD = p_slow_te * log_pred_slow + (1 - p_slow_te) * log_pred_fast
d_pred_AD   = np.exp(log_pred_AD)
res_AD = report("Option A+D (soft gating)", d_test, d_pred_AD)

# ── Option B: Physical router (half-decay ACF lag, feature index 9) ────────────
log("\n=== Option B: Physical router (half-decay lag threshold) ===")
HAD_IDX = 9  # half-decay lag is feature 9 in the 207-feature vector
lags_train = X_train[:, HAD_IDX]
lags_test  = X_test[:,  HAD_IDX]
labels_tr  = (d_train < 1.0)

# Find optimal threshold: scan unique lag values on training set
unique_lags = np.sort(np.unique(lags_train))
best_thresh, best_acc_B = 0, 0
for thr in unique_lags:
    pred_cls = lags_train > thr   # high lag → slow diffusion → d < 1
    acc = (pred_cls == labels_tr).mean()
    if acc > best_acc_B: best_acc_B, best_thresh = acc, thr

pred_slow_B  = lags_test > best_thresh
acc_B = accuracy_score(y_cls_te, pred_slow_B)
log(f"  Optimal lag threshold: {best_thresh:.1f}  train_acc={best_acc_B*100:.2f}%  test_acc={acc_B*100:.2f}%")

log_pred_B = np.where(pred_slow_B, log_pred_slow, log_pred_fast)
d_pred_B   = np.exp(log_pred_B)
res_B = report("Option B (physical lag threshold)", d_test, d_pred_B)

# ── Option C: General model as router ─────────────────────────────────────────
log("\n=== Option C: General model (MLP cosine) as router ===")
d_pred_gen_route = np.exp(log_pred_gen)
pred_slow_C = d_pred_gen_route < 1.0
acc_C = accuracy_score(y_cls_te, pred_slow_C)
log(f"  General model routing accuracy: {acc_C*100:.2f}%")

log_pred_C = np.where(pred_slow_C, log_pred_slow, log_pred_fast)
d_pred_C   = np.exp(log_pred_C)
res_C = report("Option C (general model router)", d_test, d_pred_C)

# ── Baseline: general model alone ─────────────────────────────────────────────
res_base = report("Baseline: MLP cosine (no routing)", d_test, np.exp(log_pred_gen))

runtime = time.time()-t_total

# ── Save summary ──────────────────────────────────────────────────────────────
summary = (f"Task: pt_router_all\nRuntime: {runtime:.1f}s\n"
           f"Router accuracy (learned, threshold=0.5): {router_acc*100:.2f}%\n"
           f"Router accuracy (physical lag):  {acc_B*100:.2f}%\n"
           f"Router accuracy (general model): {acc_C*100:.2f}%\n\n"
           + res_base + "\n\n" + res_AD + "\n\n" + res_B + "\n\n" + res_C + "\n")
print(f"\n{'='*60}\n{summary}{'='*60}")
with open('results_pt_router_all.txt','w') as f: f.write(summary)
log("Saved results_pt_router_all.txt")

# ── Comparison plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
configs = [
    (axes[0,0], np.exp(log_pred_gen), 'Baseline (MLP cosine)', 'steelblue'),
    (axes[0,1], d_pred_AD,            'Option A+D (soft gate)', 'seagreen'),
    (axes[1,0], d_pred_B,             'Option B (lag threshold)', 'darkorange'),
    (axes[1,1], d_pred_C,             'Option C (general router)', 'crimson'),
]
for ax, dp, title, col in configs:
    ax.scatter(d_test, dp, alpha=0.15, s=3, c=np.where(d_test<1, col, 'gray'))
    lims = [min(d_test.min(), dp.min()), max(d_test.max(), dp.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.axvline(1, color='black', lw=0.8, ls=':'); ax.axhline(1, color='black', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
    r2 = r2_score(d_test, dp); mp = mape(d_test, dp)
    ax.set_title(f'{title}\nR²={r2:.4f}  MAPE={mp:.1f}%')
plt.tight_layout(); plt.savefig('pt_router_comparison.png', dpi=150)
log("Saved pt_router_comparison.png"); print("Done.")
