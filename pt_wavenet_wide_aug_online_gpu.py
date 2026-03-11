"""
WaveNet wide — online background augmentation approach.

Identical architecture and training data to pt_wavenet_wide_aug_gpu.py, but adds
online Poisson background noise to the trace branch during training.

Online approach:
  Each batch, sample b_pct ~ Uniform[0, BG_MAX_PCT] (0–20% of MAX_RATE).
  Convert to background photon rate: b_bin = b_pct * MAX_RATE * DT (photons/bin).
  Draw Poisson(b_bin) independently per bin and add to raw trace before z-score norm.
  Feature branch receives pre-computed clean features (cache) — trace branch only.

Test evaluation:
  Four fixed background levels (b=0%, 5%, 10%, 20% of MAX_RATE) applied to the
  same 9,502-sample test set, each with a fixed seed for reproducibility.
  Baseline wavenet_wide_aug (trained without online augmentation) is implicitly
  comparable at b=0% level.
"""

import numpy as np, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL        = 'pt_wavenet_wide_aug_online'
CHANNELS     = 256
BATCH_SIZE   = 128
EVAL_BATCH   = 256
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]

# ── Online background augmentation parameters ─────────────────────────────────
MAX_RATE   = 50_000.0   # counts/s  (matches simulation MAX_RATE)
DT         = 1e-3       # s         (dt = 1 ms)
BG_MAX_PCT = 0.20       # Uniform[0, 20%] of MAX_RATE during training

# Fixed background levels for test evaluation (% of MAX_RATE)
BG_EVAL_PCTS = [0.00, 0.05, 0.10, 0.20]   # → 0, 2.5, 5, 10 photons/bin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


def add_background(traces_raw, b_pct, rng):
    """
    Add Poisson background to raw intensity traces (counts/s).

    Parameters
    ----------
    traces_raw : ndarray (N, T), float32  — raw intensity in counts/s
    b_pct      : float — background as fraction of MAX_RATE (e.g. 0.10 = 10%)
    rng        : np.random.Generator

    Returns
    -------
    traces_noisy : ndarray (N, T), float32  — with background added, still counts/s
    """
    if b_pct <= 0.0:
        return traces_raw.copy()
    b_bin = b_pct * MAX_RATE * DT          # expected background photons per bin
    bg    = rng.poisson(b_bin, size=traces_raw.shape).astype(np.float32) / DT
    return traces_raw + bg


def norm_traces(traces):
    """Per-trace z-score normalisation (applied after background addition)."""
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return (traces - mu) / sig


def batch_predict(model_fn, *cpu_tensors, bs=256):
    n = len(cpu_tensors[0]); outs = []
    for i in range(0, n, bs):
        batch = [t[i:i+bs].to(device) for t in cpu_tensors]
        outs.append(model_fn(*batch).cpu())
    return torch.cat(outs)


# ── Model ─────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn   = nn.BatchNorm1d(channels)
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


class WaveNet1D(nn.Module):
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trace, feats):
        x = self.input_proj(trace.unsqueeze(1))
        x = self.blocks(x)
        x = self.gap(x).squeeze(2)
        f = self.feat_branch(feats)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


# ── Data loading ──────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log("Loading training data (raw traces kept unnormalised for online augmentation)...")

# Raw traces — kept un-normalised; background added per batch before z-scoring
i_train_raw = np.load('cache_i_train_aug.npy').astype(np.float32)
X_train     = np.load('cache_X_train_aug.npy').astype(np.float32)
d_train     = np.load('cache_d_train_aug.npy')

i_test_raw  = np.load('cache_i_test_90pct.npy').astype(np.float32)
X_test      = np.load('cache_X_test_90pct.npy').astype(np.float32)
d_test      = np.load('cache_d_test_90pct.npy')

mask_tr = d_train <= 10
i_train_raw, X_train, d_train = i_train_raw[mask_tr], X_train[mask_tr], d_train[mask_tr]
mask_te = d_test <= 10
i_test_raw,  X_test,  d_test  = i_test_raw[mask_te],  X_test[mask_te],  d_test[mask_te]
log(f"Train: {len(d_train)}  Test: {len(d_test)}")

# Feature scaler fitted on clean training features
feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)
log_d   = np.log(d_train).astype(np.float32)

# Train/val split (indices only — traces normalised on the fly)
rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

# Validation traces: clean (b=0), normalised once
i_val_n = norm_traces(i_train_raw[val_idx])
X_val   = torch.tensor(X_tr_sc[val_idx])
y_val   = torch.tensor(log_d[val_idx])

# Index DataLoader for training (background applied per batch)
tr_idx_t  = torch.arange(len(tr_idx))
idx_loader = DataLoader(TensorDataset(tr_idx_t),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

X_tr_t = torch.tensor(X_tr_sc[tr_idx])
y_tr_t = torch.tensor(log_d[tr_idx])

# RNG for online background augmentation (separate from split RNG)
rng_aug = np.random.default_rng(0)

# ── Model + optimiser ─────────────────────────────────────────────────────────

model = WaveNet1D(feat_dim=X_tr_t.shape[1], channels=CHANNELS, dilations=DILATIONS).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit  = nn.MSELoss()

n_params   = sum(p.numel() for p in model.parameters())
rf_original = sum(2 * d for d in DILATIONS) * 4

log(f"Training {LABEL}  params={n_params:,}")
log(f"  channels={CHANNELS}  dilations={DILATIONS}  RF={rf_original} original lags")
log(f"  Online background: Uniform[0, {BG_MAX_PCT*100:.0f}%] of MAX_RATE "
    f"= Uniform[0, {BG_MAX_PCT*MAX_RATE*DT:.1f}] photons/bin")

# ── Training loop ─────────────────────────────────────────────────────────────

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep = 0.0

    for (batch_local_idx,) in idx_loader:
        # Global indices into i_train_raw / X_tr_t / y_tr_t
        global_idx = tr_idx[batch_local_idx.numpy()]

        # ── Online background augmentation (trace branch only) ────────────────
        b_pct      = rng_aug.uniform(0.0, BG_MAX_PCT)
        ib_raw     = i_train_raw[global_idx]           # (B, T) counts/s, raw
        ib_noisy   = add_background(ib_raw, b_pct, rng_aug)
        ib_n       = norm_traces(ib_noisy)             # z-score after background
        # ─────────────────────────────────────────────────────────────────────

        ib = torch.tensor(ib_n,                      dtype=torch.float32).to(device)
        xb = X_tr_t[batch_local_idx].to(device)
        yb = y_tr_t[batch_local_idx].to(device)

        opt.zero_grad()
        loss = crit(model(ib, xb), yb)
        loss.backward()
        opt.step()
        ep += loss.item() * len(ib)

    ep /= len(tr_idx)

    # Validation on clean traces (b=0)
    model.eval()
    i_val_t = torch.tensor(i_val_n)
    with torch.no_grad():
        vl = crit(
            batch_predict(lambda i, x: model(i, x), i_val_t, X_val, bs=EVAL_BATCH),
            y_val,
        ).item()

    sched.step(epoch - 1)

    if vl < best_val - 1e-6:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep:.5f}  val={vl:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.5f}")

# ── Save model + scaler ───────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_)
log(f"Saved model_{LABEL}.pt + scaler")

# ── Evaluate at four background levels ────────────────────────────────────────

model.eval()
rng_eval = np.random.default_rng(999)   # fixed seed → deterministic test background

log("Evaluating at fixed background levels...")
results_lines = [
    f"Task: {LABEL}",
    f"Architecture: WaveNet1D channels={CHANNELS} dilations={DILATIONS}",
    f"  Receptive field: {rf_original} original lags",
    f"Params: {n_params:,}",
    f"Online approach: background Uniform[0, {BG_MAX_PCT*100:.0f}%] of MAX_RATE per batch",
    f"  MAX_RATE={MAX_RATE:.0f} counts/s  DT={DT*1e3:.1f} ms",
    f"  Background range: 0 – {BG_MAX_PCT*MAX_RATE*DT:.1f} photons/bin",
    f"Train: {len(d_train)}  Test: {len(d_test)}",
    "",
]

X_te_t = torch.tensor(X_te_sc)

for b_pct in BG_EVAL_PCTS:
    b_bin = b_pct * MAX_RATE * DT
    label_b = f"b{int(b_pct*100):02d}pct"

    i_te_noisy = add_background(i_test_raw, b_pct, rng_eval)
    i_te_n     = norm_traces(i_te_noisy)
    i_te_t     = torch.tensor(i_te_n)

    with torch.no_grad():
        logd_pred = batch_predict(
            lambda i, x: model(i, x), i_te_t, X_te_t, bs=EVAL_BATCH
        ).numpy()

    d_pred = np.exp(logd_pred)
    np.save(f'pred_logd_{LABEL}_test_{label_b}.npy', logd_pred)

    ms, mf = d_test < 1.0, d_test >= 1.0
    block = (
        f"Background: {b_pct*100:.0f}% of MAX_RATE  "
        f"({b_bin:.2f} photons/bin)\n"
        f"  Test  R²={r2_score(d_test, d_pred):.4f}  "
        f"MAE={mean_absolute_error(d_test, d_pred):.4f}  "
        f"MAPE={mape(d_test, d_pred):.1f}%\n"
        f"  d <1  ({ms.sum():5d}): R²={r2_score(d_test[ms], d_pred[ms]):.4f}  "
        f"MAPE={mape(d_test[ms], d_pred[ms]):.1f}%\n"
        f"  d>=1  ({mf.sum():5d}): R²={r2_score(d_test[mf], d_pred[mf]):.4f}  "
        f"MAPE={mape(d_test[mf], d_pred[mf]):.1f}%"
    )
    log(block.replace('\n', '  |  '))
    results_lines.append(block)
    results_lines.append("")

# Also save clean training-set predictions for meta-learner compatibility
i_tr_all_n = norm_traces(i_train_raw)          # clean (b=0)
i_tr_all_t = torch.tensor(i_tr_all_n)
X_tr_all_t = torch.tensor(X_tr_sc)
with torch.no_grad():
    logd_pred_tr = batch_predict(
        lambda i, x: model(i, x), i_tr_all_t, X_tr_all_t, bs=EVAL_BATCH
    ).numpy()

np.save(f'pred_logd_{LABEL}_train.npy', logd_pred_tr)
log(f"Saved pred_logd_{LABEL}_train.npy")

runtime = time.time() - t_total
results_lines.append(f"Runtime: {runtime:.1f}s")

summary = "\n".join(results_lines)
print(f"\n{'='*60}\n{summary}\n{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
