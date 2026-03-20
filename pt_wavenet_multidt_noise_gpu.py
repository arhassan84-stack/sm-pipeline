"""
Multi-dt WaveNet — Noise: predict background level n.

Background n is added on-the-fly to the existing noise0 training data:
  n_frac ~ |N(0, NOISE_SIGMA)| clipped at NOISE_MAX_FRAC (0.50)
  n_abs  = n_frac * max(I_dt100_scaled)
  Traces are counts/s → expected counts per dt010 bin = n_abs × DT_MIN_S (1e-4)
  bg_counts ~ Poisson(n_abs × DT_MIN_S); bg_010 = bg_counts / DT_MIN_S (counts/s)
  bg_050/100 = average of 5/10 consecutive dt010 bg bins (same photon stream)

Key changes vs multidt_B:
  1. 6 extra input features: [raw_mean, raw_std] per dt channel (primary n signal)
     appended to the 906 cached features → feat_dim = 912
  2. Head output: Softplus  (enforces n_pred >= 0)
  3. Loss: MSE on n_abs
  4. Target: n_abs  (absolute background, same units as intensity trace)

The raw stats are the critical features: after trace normalization the WaveNet
branches are blind to absolute intensity, so raw_mean and raw_std carry the
background information the model needs.

Label: wavenet_multidt_noise
"""

import numpy as np, time, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

LABEL          = 'wavenet_multidt_noise'
CHANNELS       = 256
BATCH_SIZE     = 128
DL_WORKERS     = 6
MAX_EPOCHS     = 200
PATIENCE       = 25
LR             = 5e-4
WEIGHT_DECAY   = 1e-4
T0             = 30
DILATIONS      = [1, 2, 4, 8, 16, 32, 64, 128]
AUG_SCALE_LO   = 0.85
AUG_SCALE_HI   = 1.15
AUG_NOISE_SD   = 0.03
AUG_SHIFT      = 128
EVAL_BATCH     = 128
POOL010        = 16

NOISE_SIGMA    = 0.15   # std of half-normal for n_frac
NOISE_MAX_FRAC = 0.50   # hard clip: n_abs <= 50% of max intensity
DT_MIN_S       = 1e-4   # dt010 bin width in seconds (0.1 ms)
                         # traces stored as counts/s → Poisson draw needs × DT_MIN_S
N_SCALE        = 1e3    # predict n in kHz: target = n_abs / N_SCALE ∈ [0,~84]
                         # keeps MSE well within FP16 range (avoids AMP NaN overflow)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape_safe(t, p):
    """MAPE only for samples where n_true > 1% of mean(n_true)."""
    thr = max(1e-2 * t.mean(), 1e-8)
    m = t > thr
    return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100 if m.sum() > 0 else float('nan')


# ── Dataset ────────────────────────────────────────────────────────────────────

class NoiseFCSDataset(Dataset):
    """On-the-fly background augmentation.

    For each sample:
      1. Scale augmentation (brightness jitter)
      2. Draw n_frac ~ |N(0, NOISE_SIGMA)|, clip at NOISE_MAX_FRAC
      3. n_abs = n_frac * max(t010_scaled)  [counts per dt010 bin, finest resolution]
      4. Draw bg_010[i] ~ Poisson(n_abs) for all 40960 dt010 bins
         bg_050[j] = mean(bg_010[5j : 5j+5])   [average 5 → dt050]
         bg_100[k] = mean(bg_010[10k: 10k+10])  [average 10 → dt100]
         → noise is physically correlated across channels (same photon stream)
      5. Record raw mean/std per channel BEFORE Gaussian noise  (6 scalars → primary n signal)
      6. Add per-bin Gaussian noise, circular shift, normalize  (for WaveNet input)
    Target: n_abs  (counts per dt010 bin)
    """
    def __init__(self, i010, i010_idx, i050, i100, X_tensor,
                 raw_stat_mean, raw_stat_std):
        self.i010 = i010
        self.i010_idx = i010_idx
        self.i050 = i050
        self.i100 = i100
        self.X = X_tensor
        self.rsm = raw_stat_mean   # (6,) float32
        self.rss = raw_stat_std    # (6,) float32
        self._rng = None

    def __len__(self): return len(self.i050)

    @property
    def rng(self):
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            seed = int(wi.seed % (2**31)) if wi is not None else 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, idx):
        t010 = self.i010[self.i010_idx[idx]]   # (40960,) raw float32
        t050 = self.i050[idx]                   # (8192,)
        t100 = self.i100[idx]                   # (4096,)

        # 1. Scale augmentation
        scale = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        t010s = t010 * scale
        t050s = t050 * scale
        t100s = t100 * scale

        # 2. Background level — defined at finest dt (dt010 = 0.1ms)
        n_frac = float(min(abs(self.rng.normal(0.0, NOISE_SIGMA)), NOISE_MAX_FRAC))
        n_abs  = np.float32(n_frac * float(t010s.max()))

        # 3. Correlated Poisson background — matching simulation units (counts/s)
        #    Traces are counts/s → expected counts per dt010 bin = n_abs * DT_MIN_S
        #    Draw integer counts, divide back to counts/s, then bin down by averaging
        bg_counts = self.rng.poisson(n_abs * DT_MIN_S, size=t010s.shape).astype(np.float32)
        bg_010 = bg_counts / DT_MIN_S                                  # counts/s at dt010
        bg_050 = bg_010.reshape(8192,  5).mean(axis=1).astype(np.float32)   # avg 5  → dt050
        bg_100 = bg_010.reshape(4096, 10).mean(axis=1).astype(np.float32)   # avg 10 → dt100
        t010n = t010s + bg_010
        t050n = t050s + bg_050
        t100n = t100s + bg_100

        # 4. Raw statistics (before per-bin Gaussian noise): primary signal for n
        raw = np.array([t010n.mean(), t010n.std(),
                        t050n.mean(), t050n.std(),
                        t100n.mean(), t100n.std()], dtype=np.float32)
        raw_sc = (raw - self.rsm) / self.rss

        # 5. Per-bin noise + shift + normalize
        s100 = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))

        def aug(t, shift):
            a = t + self.rng.normal(0.0, AUG_NOISE_SD, t.shape).astype(np.float32)
            a = np.roll(a, shift)
            return (a - a.mean()) / (a.std() + 1e-8)

        a010 = aug(t010n, s100 * 10)
        a050 = aug(t050n, s100 * 2)
        a100 = aug(t100n, s100)

        x_all = np.concatenate([self.X[idx].numpy(), raw_sc])
        return (torch.from_numpy(a010.astype(np.float32)),
                torch.from_numpy(a050.astype(np.float32)),
                torch.from_numpy(a100.astype(np.float32)),
                torch.tensor(x_all,              dtype=torch.float32),
                torch.tensor(n_abs / N_SCALE,    dtype=torch.float32))  # kHz


def norm_traces(traces):
    mu  = traces.mean(axis=1, keepdims=True)
    sig = traces.std(axis=1,  keepdims=True) + 1e-8
    return ((traces - mu) / sig).astype(np.float32)


# ── Model ──────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)) + x)


class WaveNetBackbone(nn.Module):
    def __init__(self, channels=256, dilations=None, avgpool=1):
        super().__init__()
        if dilations is None: dilations = [1,2,4,8,16,32,64,128]
        proj = []
        if avgpool > 1:
            proj.append(nn.AvgPool1d(kernel_size=avgpool, stride=avgpool))
        proj += [nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
                 nn.BatchNorm1d(channels), nn.ReLU()]
        self.input_proj = nn.Sequential(*proj)
        self.blocks = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap    = nn.AdaptiveAvgPool1d(1)

    def forward(self, trace):
        x = self.input_proj(trace.unsqueeze(1))
        return self.gap(self.blocks(x)).squeeze(2)


class WaveNetMultiDT_Noise(nn.Module):
    """Same 3-branch backbone as multidt_B, Softplus output for noise prediction."""
    def __init__(self, feat_dim, channels=256, dilations=None):
        super().__init__()
        self.branch010 = WaveNetBackbone(channels, dilations, avgpool=POOL010)
        self.branch050 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),   # n_pred >= 0
        )
        self.s010 = torch.cuda.Stream()
        self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        cur = torch.cuda.current_stream()
        self.s010.wait_stream(cur)
        self.s050.wait_stream(cur)
        with torch.cuda.stream(self.s010): e010 = self.branch010(t010)
        with torch.cuda.stream(self.s050): e050 = self.branch050(t050)
        e100 = self.branch100(t100)
        f    = self.feat_branch(feats)
        cur.wait_stream(self.s010)
        cur.wait_stream(self.s050)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


# ── Data loading ───────────────────────────────────────────────────────────────

t_total = time.time()
log(f"Device: {device}")
log("Loading multi-dt caches (RAM)...")

i010_tr = np.load('cache_i_train_multidt_dt010.npy').astype(np.float32)
i050_tr = np.load('cache_i_train_multidt_dt050.npy').astype(np.float32)
i100_tr = np.load('cache_i_train_multidt_dt100.npy').astype(np.float32)
X010_tr = np.load('cache_X_train_multidt_dt010.npy').astype(np.float32)
X050_tr = np.load('cache_X_train_multidt_dt050.npy').astype(np.float32)
X100_tr = np.load('cache_X_train_multidt_dt100.npy').astype(np.float32)
d_train = np.load('cache_d_train_multidt.npy')

i010_te = np.load('cache_i_test_multidt_dt010.npy').astype(np.float32)
i050_te = np.load('cache_i_test_multidt_dt050.npy').astype(np.float32)
i100_te = np.load('cache_i_test_multidt_dt100.npy').astype(np.float32)
X010_te = np.load('cache_X_test_multidt_dt010.npy').astype(np.float32)
X050_te = np.load('cache_X_test_multidt_dt050.npy').astype(np.float32)
X100_te = np.load('cache_X_test_multidt_dt100.npy').astype(np.float32)
d_test  = np.load('cache_d_test_multidt.npy')

X_train = np.concatenate([X010_tr, X050_tr, X100_tr], axis=1)
X_test  = np.concatenate([X010_te, X050_te, X100_te], axis=1)

# Apply d<=10 mask (same as D-prediction models)
mask_tr = d_train <= 10
tr_global = np.where(mask_tr)[0]
i050_tr = i050_tr[mask_tr]; i100_tr = i100_tr[mask_tr]
X_train = X_train[mask_tr]; d_train = d_train[mask_tr]
# i010_tr is the full array indexed via tr_global

mask_te = d_test <= 10
te_global = np.where(mask_te)[0]
i050_te = i050_te[mask_te]; i100_te = i100_te[mask_te]
X_test  = X_test[mask_te];  d_test  = d_test[mask_te]

log(f"Train: {len(d_train)}  Test: {len(d_test)}")

feat_scaler = StandardScaler()
X_tr_sc = feat_scaler.fit_transform(X_train).astype(np.float32)
X_te_sc = feat_scaler.transform(X_test).astype(np.float32)

# Train / val split
rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

# ── Raw stat scaler (fitted on clean training data, no noise) ──────────────────

log("Computing raw stat normalization (5k subsample of clean training traces)...")
rng_rs   = np.random.default_rng(0)
rs_samp  = rng_rs.choice(len(tr_idx), size=min(5000, len(tr_idx)), replace=False)
rs_i010  = i010_tr[tr_global[tr_idx[rs_samp]]]   # (5000, 40960)
rs_i050  = i050_tr[tr_idx[rs_samp]]
rs_i100  = i100_tr[tr_idx[rs_samp]]

raw_stat_sample = np.column_stack([
    rs_i010.mean(axis=1), rs_i010.std(axis=1),
    rs_i050.mean(axis=1), rs_i050.std(axis=1),
    rs_i100.mean(axis=1), rs_i100.std(axis=1),
]).astype(np.float32)
del rs_i010

raw_stat_scaler = StandardScaler().fit(raw_stat_sample)
raw_stat_mean   = raw_stat_scaler.mean_.astype(np.float32)
raw_stat_std    = raw_stat_scaler.scale_.astype(np.float32)
log(f"  raw_stat_mean (mean,std per channel): {np.round(raw_stat_mean, 3)}")

# ── Validation set: pre-add fixed noise for stable early stopping ──────────────

log("Pre-processing validation set with fixed noise...")
rng_vn      = np.random.default_rng(99)
n_frac_val  = np.abs(rng_vn.normal(0.0, NOISE_SIGMA, size=len(val_idx))).clip(0, NOISE_MAX_FRAC).astype(np.float32)

i010_val_raw = i010_tr[tr_global[val_idx]]   # (N_val, 40960)
i050_val_raw = i050_tr[val_idx]
i100_val_raw = i100_tr[val_idx]

n_abs_val = (n_frac_val * i010_val_raw.max(axis=1)).astype(np.float32)

# Correlated Poisson background: generate at dt010, bin down — chunked for dt010
i010_val_nsy = np.empty_like(i010_val_raw)
i050_val_nsy = np.empty_like(i050_val_raw)
i100_val_nsy = np.empty_like(i100_val_raw)
_VCHUNK = 256
for _s in range(0, len(val_idx), _VCHUNK):
    _e = min(_s + _VCHUNK, len(val_idx))
    _n = _e - _s
    _lam = np.broadcast_to(n_abs_val[_s:_e, None] * DT_MIN_S, (_n, 40960)).copy()
    _bg_counts = rng_vn.poisson(_lam).astype(np.float32)   # integer counts per dt010 bin
    _bg010 = _bg_counts / DT_MIN_S                          # back to counts/s
    i010_val_nsy[_s:_e] = i010_val_raw[_s:_e] + _bg010
    i050_val_nsy[_s:_e] = i050_val_raw[_s:_e] + _bg010.reshape(_n, 8192,  5).mean(axis=2)
    i100_val_nsy[_s:_e] = i100_val_raw[_s:_e] + _bg010.reshape(_n, 4096, 10).mean(axis=2)

raw_stats_val = np.column_stack([
    i010_val_nsy.mean(axis=1), i010_val_nsy.std(axis=1),
    i050_val_nsy.mean(axis=1), i050_val_nsy.std(axis=1),
    i100_val_nsy.mean(axis=1), i100_val_nsy.std(axis=1),
]).astype(np.float32)
raw_stats_val_sc = (raw_stats_val - raw_stat_mean) / raw_stat_std

i010_val_n = norm_traces(i010_val_nsy)
i050_val_n = norm_traces(i050_val_nsy)
i100_val_n = norm_traces(i100_val_nsy)
X_val_combined = np.concatenate([X_tr_sc[val_idx], raw_stats_val_sc], axis=1)
y_val = torch.tensor(n_abs_val / N_SCALE)   # kHz

del i010_val_raw, i050_val_raw, i100_val_raw, i010_val_nsy, i050_val_nsy, i100_val_nsy
log(f"  Val n_abs: mean={n_abs_val.mean():.4f}  max={n_abs_val.max():.4f}")

# ── DataLoader ─────────────────────────────────────────────────────────────────

train_ds = NoiseFCSDataset(
    i010=i010_tr, i010_idx=tr_global[tr_idx],
    i050=i050_tr[tr_idx], i100=i100_tr[tr_idx],
    X_tensor=torch.tensor(X_tr_sc[tr_idx]),
    raw_stat_mean=raw_stat_mean, raw_stat_std=raw_stat_std,
)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=DL_WORKERS, pin_memory=True,
                    persistent_workers=True, prefetch_factor=2)

# ── Model + optimiser ──────────────────────────────────────────────────────────

feat_dim = X_tr_sc.shape[1] + 6   # 906 cached + 6 raw stats = 912
model  = WaveNetMultiDT_Noise(feat_dim=feat_dim, channels=CHANNELS, dilations=DILATIONS).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit   = nn.MSELoss()
scaler = GradScaler()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")
log(f"  feat_dim={feat_dim} ({feat_dim-6} cached + 6 raw stats)  output=Softplus  target=kHz")
log(f"  Noise: |N(0,{NOISE_SIGMA})| clipped at {NOISE_MAX_FRAC}  AMP | CUDA streams")

# ── Training loop ──────────────────────────────────────────────────────────────

i010_val_t = torch.tensor(i010_val_n)
i050_val_t = torch.tensor(i050_val_n)
i100_val_t = torch.tensor(i100_val_n)
X_val_t    = torch.tensor(X_val_combined)

best_val, best_state, patience_count = float('inf'), None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); ep = 0.0
    for t010b, t050b, t100b, xb, nb in loader:
        t010b = t010b.to(device, non_blocking=True)
        t050b = t050b.to(device, non_blocking=True)
        t100b = t100b.to(device, non_blocking=True)
        xb    = xb.to(device, non_blocking=True)
        nb    = nb.to(device, non_blocking=True)
        opt.zero_grad()
        with autocast():
            loss = crit(model(t010b, t050b, t100b, xb), nb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()
        ep += loss.item() * len(nb)
    ep /= len(tr_idx)

    model.eval()
    preds_val = []
    with torch.no_grad(), autocast():
        for i in range(0, len(val_idx), EVAL_BATCH):
            preds_val.append(model(
                i010_val_t[i:i+EVAL_BATCH].to(device),
                i050_val_t[i:i+EVAL_BATCH].to(device),
                i100_val_t[i:i+EVAL_BATCH].to(device),
                X_val_t[i:i+EVAL_BATCH].to(device)).cpu())
    pred_val = torch.cat(preds_val)
    vl = crit(pred_val, y_val).item()

    sched.step(epoch - 1)
    if vl < best_val - 1e-8:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

    if epoch % 10 == 0:
        log(f"Epoch {epoch:3d}  train={ep:.6f}  val={vl:.6f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.6f}")

# ── Save ───────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',    feat_scaler.mean_)
np.save(f'scaler_{LABEL}_scale.npy',   feat_scaler.scale_)
np.save(f'raw_stat_{LABEL}_mean.npy',  raw_stat_mean)
np.save(f'raw_stat_{LABEL}_std.npy',   raw_stat_std)

# ── Inference helper ───────────────────────────────────────────────────────────

def predict_n(i010_all, i010_idx, i050, i100, X_sc, rng_seed):
    """Add fixed-noise augmentation to a set of traces and predict n_abs."""
    N = len(i050)
    rng = np.random.default_rng(rng_seed)
    n_frac = np.abs(rng.normal(0.0, NOISE_SIGMA, size=N)).clip(0, NOISE_MAX_FRAC).astype(np.float32)

    n_abs  = (n_frac * i010_all[i010_idx].max(axis=1)).astype(np.float32)
    n_pred = []
    CHUNK  = 256

    # Allocate output arrays
    i050_nsy  = np.empty_like(i050)
    i100_nsy  = np.empty_like(i100)
    i010_n    = np.empty((N, i010_all.shape[1]), dtype=np.float32)
    raw_stats = np.empty((N, 6), dtype=np.float32)

    # Correlated Poisson background: generate at dt010, bin down (chunked)
    for s in range(0, N, CHUNK):
        e   = min(s + CHUNK, N)
        n   = e - s
        # Correct units: traces are counts/s → Poisson(n_abs * DT_MIN_S) gives counts/bin
        lam = np.broadcast_to(n_abs[s:e, None] * DT_MIN_S, (n, 40960)).copy()
        bg_counts = rng.poisson(lam).astype(np.float32)   # integer counts per dt010 bin
        bg010 = bg_counts / DT_MIN_S                       # back to counts/s
        # bin down to dt050 and dt100
        i050_nsy[s:e] = i050[s:e] + bg010.reshape(n, 8192,  5).mean(axis=2)
        i100_nsy[s:e] = i100[s:e] + bg010.reshape(n, 4096, 10).mean(axis=2)
        # noisy dt010 trace + raw stats
        chunk = i010_all[i010_idx[s:e]] + bg010
        raw_stats[s:e, 0] = chunk.mean(axis=1)
        raw_stats[s:e, 1] = chunk.std(axis=1)
        mu  = chunk.mean(axis=1, keepdims=True)
        sig = chunk.std(axis=1,  keepdims=True) + 1e-8
        i010_n[s:e] = (chunk - mu) / sig

    # Raw stats for dt050 and dt100 — computed AFTER loop fills arrays
    raw_stats[:, 2] = i050_nsy.mean(axis=1); raw_stats[:, 3] = i050_nsy.std(axis=1)
    raw_stats[:, 4] = i100_nsy.mean(axis=1); raw_stats[:, 5] = i100_nsy.std(axis=1)

    # Normalize dt050 and dt100 traces — computed AFTER loop fills arrays
    i050_n = norm_traces(i050_nsy)
    i100_n = norm_traces(i100_nsy)

    raw_stats_sc  = (raw_stats - raw_stat_mean) / raw_stat_std
    X_combined    = np.concatenate([X_sc, raw_stats_sc], axis=1)

    model.eval()
    with torch.no_grad(), autocast():
        for s in range(0, N, EVAL_BATCH):
            e = min(s + EVAL_BATCH, N)
            n_pred.append(model(
                torch.tensor(i010_n[s:e]).to(device),
                torch.tensor(i050_n[s:e]).to(device),
                torch.tensor(i100_n[s:e]).to(device),
                torch.tensor(X_combined[s:e]).to(device)).cpu())
    # unscale: model predicts in kHz, convert back to counts/s
    return np.concatenate([t.numpy() for t in n_pred]) * N_SCALE, n_abs


# ── Evaluate ───────────────────────────────────────────────────────────────────

log("Predicting test set...")
n_pred_te, n_true_te = predict_n(i010_te, te_global, i050_te, i100_te, X_te_sc, rng_seed=42)
log("Predicting train set...")
n_pred_tr, n_true_tr = predict_n(i010_tr, tr_global, i050_tr, i100_tr, X_tr_sc, rng_seed=123)

np.save(f'pred_n_{LABEL}_test.npy',   n_pred_te)
np.save(f'pred_n_{LABEL}_train.npy',  n_pred_tr)
np.save(f'true_n_{LABEL}_test.npy',   n_true_te)
np.save(f'true_n_{LABEL}_train.npy',  n_true_tr)

# Stratified MAPE by n_frac bin
n_frac_te = n_true_te / (i100_te.max(axis=1) + 1e-8)   # approximate n_frac for reporting
bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5)]

runtime = time.time() - t_total

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 3 WaveNet branches (same as multidt_B) + Softplus output\n"
    f"  feat_dim={feat_dim} (906 cached + 6 raw stats)  channels={CHANNELS}\n"
    f"  Noise augmentation: |N(0,{NOISE_SIGMA})| clipped at {NOISE_MAX_FRAC}\n"
    f"Params: {n_params:,}\n"
    f"Train: {len(d_train)}  Test: {len(d_test)}\n"
    f"\nTrain  R²={r2_score(n_true_tr, n_pred_tr):.4f}  "
    f"MAE={mean_absolute_error(n_true_tr, n_pred_tr):.5f}  "
    f"MAPE={mape_safe(n_true_tr, n_pred_tr):.1f}%\n"
    f"Test   R²={r2_score(n_true_te, n_pred_te):.4f}  "
    f"MAE={mean_absolute_error(n_true_te, n_pred_te):.5f}  "
    f"MAPE={mape_safe(n_true_te, n_pred_te):.1f}%\n"
    f"\nTest stratified by n_frac:\n"
)
for lo, hi in bins:
    m = (n_frac_te >= lo) & (n_frac_te < hi)
    if m.sum() > 0:
        summary += (f"  n_frac [{lo:.1f},{hi:.1f})  n={m.sum():6d}  "
                    f"MAE={mean_absolute_error(n_true_te[m], n_pred_te[m]):.5f}  "
                    f"MAPE={mape_safe(n_true_te[m], n_pred_te[m]):.1f}%\n")
summary += f"Runtime: {runtime:.1f}s\n"

print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f: f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
