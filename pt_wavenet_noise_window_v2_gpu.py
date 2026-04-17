"""
Windowed noise inference: predict background level n from a centered W-ms window.

v2 — fixes unit mismatch:  simulated traces are stored in Hz (counts/s) but
real FCS traces arrive in kHz.  All cache arrays are now divided by 1000 after
loading so that the entire training pipeline operates in kHz, matching inference.

Concretely vs v1:
  - N_SCALE = 1.0   (label = n_abs in kHz; range ~0–25, same numerical range as
                     v1's n_abs/1000 because v1 had n_abs in Hz ~0–25000)
  - Cache divided by 1000 immediately after np.load → kHz throughout
  - Poisson background rate = n_abs_kHz * 1000 * DT_MIN_S  (kHz→Hz→counts/bin)
  - bg_010 = bg_counts / DT_MIN_S / 1000               (counts/bin→Hz→kHz)
  - LABEL = wavenet_noise_w{WINDOW_MS}_v2

Usage:  python pt_wavenet_noise_window_v2_gpu.py --window_ms 512
Window choices: 256, 512, 1024, 2048 ms
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--window_ms', type=int, required=True,
                    choices=[256, 512, 1024, 2048])
args   = parser.parse_args()
WINDOW_MS = args.window_ms

# ── Window geometry ────────────────────────────────────────────────────────────
# Full trace: 40960 dt010 bins = 4096 ms
_C010 = 20480   # center bin of full trace at dt010
_C050 =  4096   # center at dt050
_C100 =  2048   # center at dt100
W010  = WINDOW_MS * 10    # bins at 0.1 ms/bin  (e.g. 512 ms → 5120 bins)
W050  = WINDOW_MS *  2    # bins at 0.5 ms/bin
W100  = WINDOW_MS         # bins at 1.0 ms/bin
S010  = _C010 - W010 // 2
S050  = _C050 - W050 // 2
S100  = _C100 - W100 // 2

# ── Hyper-parameters ───────────────────────────────────────────────────────────
LABEL        = f'wavenet_noise_w{WINDOW_MS}_v2'
CHANNELS     = 256
BATCH_SIZE   = 128
DL_WORKERS   = 2
MAX_EPOCHS   = 200
PATIENCE     = 25
LR           = 5e-4
WEIGHT_DECAY = 1e-4
T0           = 30
DILATIONS    = [1, 2, 4, 8, 16, 32, 64, 128]
AUG_SCALE_LO = 0.85
AUG_SCALE_HI = 1.15
AUG_NOISE_SD = 0.03
AUG_SHIFT    = max(1, W100 // 32)   # ~3% of window (dt100 bins)
EVAL_BATCH   = 128
POOL010      = max(1, WINDOW_MS // 256)   # keeps ~640 bins entering WaveNet

NOISE_SIGMA    = 0.15
NOISE_MAX_FRAC = 0.50
DT_MIN_S       = 1e-4     # dt010 bin width in seconds
# v2: traces are in kHz → label is n_abs in kHz → N_SCALE = 1.0
N_SCALE        = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU — aborting. Use --gres=gpu:... in SLURM.")


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mape_safe(t, p):
    thr = max(1e-2 * t.mean(), 1e-8)
    m   = t > thr
    return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100 if m.sum() > 0 else float('nan')


# ── Dataset ────────────────────────────────────────────────────────────────────

class WindowedNoiseFCSDataset(Dataset):
    """On-the-fly background augmentation on traces cropped to WINDOW_MS ms.

    All input arrays (i010, i050, i100) are expected in kHz.
    n_abs  = n_frac * max(full_dt010_kHz)   [kHz; consistent with inference]
    Poisson background drawn at dt010 resolution (40960 bins), then cropped + binned.
    Raw stats (mean, std per channel) computed from the cropped window only.
    """

    def __init__(self, i010, i010_idx, i050, i100,
                 raw_stat_mean, raw_stat_std):
        self.i010     = i010         # (N_all, 40960)  kHz
        self.i010_idx = i010_idx     # (N_tr,) global row indices into i010
        self.i050     = i050         # (N_tr, 8192)    kHz
        self.i100     = i100         # (N_tr, 4096)    kHz
        self.rsm      = raw_stat_mean   # (6,)
        self.rss      = raw_stat_std    # (6,)
        self._rng     = None

    def __len__(self): return len(self.i050)

    @property
    def rng(self):
        if self._rng is None:
            wi   = torch.utils.data.get_worker_info()
            seed = int(wi.seed % (2**31)) if wi is not None else 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, idx):
        # Arrays already in kHz (divided at load time)
        t010_full = self.i010[self.i010_idx[idx]]   # (40960,)  kHz
        t050_full = self.i050[idx]                   # (8192,)   kHz
        t100_full = self.i100[idx]                   # (4096,)   kHz

        # 1. Scale augmentation (on full trace so n_abs is defined consistently)
        scale  = np.float32(self.rng.uniform(AUG_SCALE_LO, AUG_SCALE_HI))
        t010s  = t010_full * scale
        t050s  = t050_full * scale
        t100s  = t100_full * scale

        # 2. Background level from full-trace max (kHz)
        n_frac = float(min(abs(self.rng.normal(0.0, NOISE_SIGMA)), NOISE_MAX_FRAC))
        n_abs  = np.float32(n_frac * float(t010s.max()))  # kHz

        # 3. Correlated Poisson background at dt010 resolution (full 40960 bins)
        #    Poisson rate = n_abs_kHz * 1e3 * DT_MIN_S  (counts per bin)
        bg_counts = self.rng.poisson(n_abs * 1000.0 * DT_MIN_S,
                                     size=40960).astype(np.float32)
        bg_010    = bg_counts / DT_MIN_S / 1000.0   # kHz
        bg_050    = bg_010.reshape(8192,  5).mean(axis=1).astype(np.float32)
        bg_100    = bg_010.reshape(4096, 10).mean(axis=1).astype(np.float32)

        # 4. Add background, then crop to window
        t010n = (t010s + bg_010)[S010:S010 + W010]     # (W010,)  kHz
        t050n = (t050s + bg_050)[S050:S050 + W050]     # (W050,)  kHz
        t100n = (t100s + bg_100)[S100:S100 + W100]     # (W100,)  kHz

        # 5. Raw stats from cropped window (primary noise signal)
        raw    = np.array([t010n.mean(), t010n.std(),
                           t050n.mean(), t050n.std(),
                           t100n.mean(), t100n.std()], dtype=np.float32)
        raw_sc = (raw - self.rsm) / self.rss

        # 6. Per-bin Gaussian noise + circular shift + z-score normalize
        s100 = int(self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1))

        def aug(t, shift):
            a = t + self.rng.normal(0.0, AUG_NOISE_SD, t.shape).astype(np.float32)
            a = np.roll(a, shift)
            return (a - a.mean()) / (a.std() + 1e-8)

        a010 = aug(t010n, s100 * 10)
        a050 = aug(t050n, s100 *  2)
        a100 = aug(t100n, s100)

        return (torch.from_numpy(a010.astype(np.float32)),
                torch.from_numpy(a050.astype(np.float32)),
                torch.from_numpy(a100.astype(np.float32)),
                torch.tensor(raw_sc, dtype=torch.float32),
                torch.tensor(n_abs / N_SCALE, dtype=torch.float32))


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
        if dilations is None: dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        proj = []
        if avgpool > 1:
            proj.append(nn.AvgPool1d(kernel_size=avgpool, stride=avgpool))
        proj += [nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
                 nn.BatchNorm1d(channels), nn.ReLU()]
        self.input_proj = nn.Sequential(*proj)
        self.blocks     = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap        = nn.AdaptiveAvgPool1d(1)

    def forward(self, trace):
        x = self.input_proj(trace.unsqueeze(1))
        return self.gap(self.blocks(x)).squeeze(2)


class WaveNetNoise_W(nn.Module):
    """3-branch WaveNet for windowed noise inference.
    feat_dim = 6 (raw stats from cropped window only).
    Softplus output enforces n_pred >= 0.
    """

    def __init__(self, channels=256, dilations=None):
        super().__init__()
        self.branch010 = WaveNetBackbone(channels, dilations, avgpool=POOL010)
        self.branch050 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100 = WaveNetBackbone(channels, dilations, avgpool=1)
        self.feat_branch = nn.Sequential(
            nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 32, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
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
log(f"Device: {device}  |  Window: {WINDOW_MS} ms  [v2 — kHz units]")
log(f"  dt010: {W010} bins  [{S010}:{S010+W010}]  POOL010={POOL010}")
log(f"  dt050: {W050} bins  [{S050}:{S050+W050}]")
log(f"  dt100: {W100} bins  [{S100}:{S100+W100}]  AUG_SHIFT={AUG_SHIFT}")
log("Loading multi-dt caches (RAM)...")

# v2: divide by 1000 immediately → all arrays in kHz (matches real FCS traces)
i010_tr = np.load('cache_i_train_multidt_dt010.npy').astype(np.float32) / 1000.0
i050_tr = np.load('cache_i_train_multidt_dt050.npy').astype(np.float32) / 1000.0
i100_tr = np.load('cache_i_train_multidt_dt100.npy').astype(np.float32) / 1000.0
d_train = np.load('cache_d_train_multidt.npy')

i010_te = np.load('cache_i_test_multidt_dt010.npy').astype(np.float32) / 1000.0
i050_te = np.load('cache_i_test_multidt_dt050.npy').astype(np.float32) / 1000.0
i100_te = np.load('cache_i_test_multidt_dt100.npy').astype(np.float32) / 1000.0
d_test  = np.load('cache_d_test_multidt.npy')

# d <= 10 mask (consistent with D-prediction models)
mask_tr   = d_train <= 10
tr_global = np.where(mask_tr)[0]
i050_tr   = i050_tr[mask_tr]; i100_tr = i100_tr[mask_tr]
d_train   = d_train[mask_tr]

mask_te   = d_test <= 10
te_global = np.where(mask_te)[0]
i050_te   = i050_te[mask_te]; i100_te = i100_te[mask_te]
d_test    = d_test[mask_te]

log(f"Train: {len(d_train)}  Test: {len(d_test)}")

# Train / val split
rng_split = np.random.default_rng(42)
val_idx   = rng_split.choice(len(d_train), size=int(0.1 * len(d_train)), replace=False)
tr_idx    = np.setdiff1d(np.arange(len(d_train)), val_idx)

# ── Raw stat scaler (5k clean samples from cropped window) ─────────────────────
log("Computing raw stat normalization (5k clean subsample, cropped window)...")
rng_rs  = np.random.default_rng(0)
rs_samp = rng_rs.choice(len(tr_idx), size=min(5000, len(tr_idx)), replace=False)
rs_010  = i010_tr[tr_global[tr_idx[rs_samp]], S010:S010 + W010]  # (5k, W010)  kHz
rs_050  = i050_tr[tr_idx[rs_samp], S050:S050 + W050]
rs_100  = i100_tr[tr_idx[rs_samp], S100:S100 + W100]

raw_stat_sample = np.column_stack([
    rs_010.mean(axis=1), rs_010.std(axis=1),
    rs_050.mean(axis=1), rs_050.std(axis=1),
    rs_100.mean(axis=1), rs_100.std(axis=1),
]).astype(np.float32)
del rs_010

raw_stat_scaler = StandardScaler().fit(raw_stat_sample)
raw_stat_mean   = raw_stat_scaler.mean_.astype(np.float32)
raw_stat_std    = raw_stat_scaler.scale_.astype(np.float32)
log(f"  raw_stat_mean (kHz): {np.round(raw_stat_mean, 3)}")

# ── Validation set: fixed noise, cropped window ────────────────────────────────
log("Pre-processing validation set (fixed noise, cropped window)...")
rng_vn     = np.random.default_rng(99)
n_frac_val = np.abs(rng_vn.normal(0.0, NOISE_SIGMA, size=len(val_idx))).clip(0, NOISE_MAX_FRAC).astype(np.float32)

i010_val_full = i010_tr[tr_global[val_idx]]            # (N_val, 40960)  kHz
n_abs_val     = (n_frac_val * i010_val_full.max(axis=1)).astype(np.float32)  # kHz

N_val      = len(val_idx)
i010_val_w = np.empty((N_val, W010), dtype=np.float32)
i050_val_w = np.empty((N_val, W050), dtype=np.float32)
i100_val_w = np.empty((N_val, W100), dtype=np.float32)

_VCHUNK = 256
for _s in range(0, N_val, _VCHUNK):
    _e   = min(_s + _VCHUNK, N_val)
    _n   = _e - _s
    # Poisson rate = n_abs_kHz * 1e3 * DT_MIN_S  (counts per bin)
    _lam = np.broadcast_to(n_abs_val[_s:_e, None] * 1000.0 * DT_MIN_S,
                           (_n, 40960)).copy()
    _bg_counts = rng_vn.poisson(_lam).astype(np.float32)
    _bg010     = _bg_counts / DT_MIN_S / 1000.0   # kHz
    _bg050     = _bg010.reshape(_n, 8192,  5).mean(axis=2)
    _bg100     = _bg010.reshape(_n, 4096, 10).mean(axis=2)
    i010_val_w[_s:_e] = (i010_val_full[_s:_e] + _bg010)[:, S010:S010 + W010]
    i050_val_w[_s:_e] = (i050_tr[val_idx[_s:_e]] + _bg050)[:, S050:S050 + W050]
    i100_val_w[_s:_e] = (i100_tr[val_idx[_s:_e]] + _bg100)[:, S100:S100 + W100]

raw_stats_val    = np.column_stack([
    i010_val_w.mean(axis=1), i010_val_w.std(axis=1),
    i050_val_w.mean(axis=1), i050_val_w.std(axis=1),
    i100_val_w.mean(axis=1), i100_val_w.std(axis=1),
]).astype(np.float32)
raw_stats_val_sc = (raw_stats_val - raw_stat_mean) / raw_stat_std

i010_val_n = norm_traces(i010_val_w)
i050_val_n = norm_traces(i050_val_w)
i100_val_n = norm_traces(i100_val_w)
y_val      = torch.tensor(n_abs_val / N_SCALE)

del i010_val_full, i010_val_w, i050_val_w, i100_val_w
log(f"  Val n_abs (kHz): mean={n_abs_val.mean():.3f}  max={n_abs_val.max():.3f}")

# ── DataLoader ─────────────────────────────────────────────────────────────────
train_ds = WindowedNoiseFCSDataset(
    i010=i010_tr, i010_idx=tr_global[tr_idx],
    i050=i050_tr[tr_idx], i100=i100_tr[tr_idx],
    raw_stat_mean=raw_stat_mean, raw_stat_std=raw_stat_std,
)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=DL_WORKERS, pin_memory=True,
                    persistent_workers=False, prefetch_factor=2)

# ── Model + optimiser ──────────────────────────────────────────────────────────
model  = WaveNetNoise_W(channels=CHANNELS, dilations=DILATIONS).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=1)
crit   = nn.MSELoss()
scaler = GradScaler()

n_params = sum(p.numel() for p in model.parameters())
log(f"Training {LABEL}  params={n_params:,}")
log(f"  Window: {WINDOW_MS} ms  POOL010={POOL010}  AUG_SHIFT={AUG_SHIFT}  feat_dim=6 (raw stats only)")
log(f"  N_SCALE={N_SCALE}  (label = n_abs in kHz)")

# ── Training loop ──────────────────────────────────────────────────────────────
i010_val_t = torch.tensor(i010_val_n)
i050_val_t = torch.tensor(i050_val_n)
i100_val_t = torch.tensor(i100_val_n)
X_val_t    = torch.tensor(raw_stats_val_sc)

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
        for i in range(0, N_val, EVAL_BATCH):
            preds_val.append(model(
                i010_val_t[i:i + EVAL_BATCH].to(device),
                i050_val_t[i:i + EVAL_BATCH].to(device),
                i100_val_t[i:i + EVAL_BATCH].to(device),
                X_val_t[i:i + EVAL_BATCH].to(device)).cpu())
    pred_val = torch.cat(preds_val)
    vl = crit(pred_val, y_val).item()

    sched.step(epoch - 1)
    if vl < best_val - 1e-8:
        best_val, best_state, patience_count = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_count += 1

    log(f"Epoch {epoch:3d}  train={ep:.6f}  val={vl:.6f}  "
        f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")
    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}"); break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.6f}")

# ── Save ───────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(),       f'model_{LABEL}.pt')
np.save(f'raw_stat_{LABEL}_mean.npy', raw_stat_mean)
np.save(f'raw_stat_{LABEL}_std.npy',  raw_stat_std)

# ── Inference helper ───────────────────────────────────────────────────────────

def predict_n(i010_all, i010_idx, i050_full, i100_full, rng_seed):
    """Add fixed-noise augmentation, crop to window, predict n_abs (kHz).

    All input arrays expected in kHz (already divided at load time).
    Returns (n_pred_kHz, n_true_kHz).
    """
    N   = len(i050_full)
    rng = np.random.default_rng(rng_seed)
    n_frac = np.abs(rng.normal(0.0, NOISE_SIGMA, size=N)).clip(0, NOISE_MAX_FRAC).astype(np.float32)

    CHUNK     = 256
    n_abs     = np.empty(N, dtype=np.float32)
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        n_abs[s:e] = n_frac[s:e] * i010_all[i010_idx[s:e]].max(axis=1)  # kHz

    i010_n    = np.empty((N, W010), dtype=np.float32)
    i050_w    = np.empty((N, W050), dtype=np.float32)
    i100_w    = np.empty((N, W100), dtype=np.float32)
    raw_stats = np.empty((N, 6),    dtype=np.float32)

    for s in range(0, N, CHUNK):
        e   = min(s + CHUNK, N)
        n   = e - s
        # Poisson rate = n_abs_kHz * 1e3 * DT_MIN_S  (counts per bin)
        lam = np.broadcast_to(n_abs[s:e, None] * 1000.0 * DT_MIN_S,
                               (n, 40960)).copy()
        bg_counts = rng.poisson(lam).astype(np.float32)
        bg010     = bg_counts / DT_MIN_S / 1000.0   # kHz
        bg050     = bg010.reshape(n, 8192,  5).mean(axis=2)
        bg100     = bg010.reshape(n, 4096, 10).mean(axis=2)
        c010 = (i010_all[i010_idx[s:e]] + bg010)[:, S010:S010 + W010]
        raw_stats[s:e, 0] = c010.mean(axis=1); raw_stats[s:e, 1] = c010.std(axis=1)
        mu = c010.mean(axis=1, keepdims=True); sig = c010.std(axis=1, keepdims=True) + 1e-8
        i010_n[s:e] = (c010 - mu) / sig
        i050_w[s:e] = (i050_full[s:e] + bg050)[:, S050:S050 + W050]
        i100_w[s:e] = (i100_full[s:e] + bg100)[:, S100:S100 + W100]

    raw_stats[:, 2] = i050_w.mean(axis=1); raw_stats[:, 3] = i050_w.std(axis=1)
    raw_stats[:, 4] = i100_w.mean(axis=1); raw_stats[:, 5] = i100_w.std(axis=1)

    i050_n = norm_traces(i050_w); del i050_w
    i100_n = norm_traces(i100_w); del i100_w
    rs_sc  = (raw_stats - raw_stat_mean) / raw_stat_std

    n_pred = []
    model.eval()
    with torch.no_grad(), autocast():
        for s in range(0, N, EVAL_BATCH):
            e = min(s + EVAL_BATCH, N)
            n_pred.append(model(
                torch.tensor(i010_n[s:e]).to(device),
                torch.tensor(i050_n[s:e]).to(device),
                torch.tensor(i100_n[s:e]).to(device),
                torch.tensor(rs_sc[s:e]).to(device)).cpu())
    return np.concatenate([t.numpy() for t in n_pred]) * N_SCALE, n_abs  # both kHz

# ── Evaluate ───────────────────────────────────────────────────────────────────
log("Predicting test set...")
n_pred_te, n_true_te = predict_n(i010_te, te_global, i050_te, i100_te, rng_seed=42)
log("Predicting train set...")
n_pred_tr, n_true_tr = predict_n(i010_tr, tr_global, i050_tr, i100_tr, rng_seed=123)

np.save(f'pred_n_{LABEL}_test.npy',  n_pred_te)
np.save(f'pred_n_{LABEL}_train.npy', n_pred_tr)
np.save(f'true_n_{LABEL}_test.npy',  n_true_te)
np.save(f'true_n_{LABEL}_train.npy', n_true_tr)

n_frac_te = n_true_te / (i100_te[:, S100:S100 + W100].max(axis=1) + 1e-8)
bins      = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5)]
runtime   = time.time() - t_total

summary = (
    f"Task: {LABEL}\n"
    f"Window: {WINDOW_MS} ms  (dt010:{W010} bins, dt050:{W050}, dt100:{W100})\n"
    f"Architecture: 3-branch WaveNet  POOL010={POOL010}  channels={CHANNELS}"
    f"  feat_dim=6 (raw stats only)  output=Softplus\n"
    f"v2 fix: traces in kHz (/ 1000 at load)  N_SCALE={N_SCALE}\n"
    f"Params: {n_params:,}\n"
    f"Train: {len(d_train)}  Test: {len(d_test)}\n"
    f"\nTrain  R²={r2_score(n_true_tr, n_pred_tr):.4f}  "
    f"MAE={mean_absolute_error(n_true_tr, n_pred_tr):.4f} kHz  "
    f"MAPE={mape_safe(n_true_tr, n_pred_tr):.1f}%\n"
    f"Test   R²={r2_score(n_true_te, n_pred_te):.4f}  "
    f"MAE={mean_absolute_error(n_true_te, n_pred_te):.4f} kHz  "
    f"MAPE={mape_safe(n_true_te, n_pred_te):.1f}%\n"
    f"\nTest stratified by n_frac:\n"
)
for lo, hi in bins:
    m = (n_frac_te >= lo) & (n_frac_te < hi)
    if m.sum() > 0:
        summary += (f"  n_frac [{lo:.1f},{hi:.1f})  n={m.sum():6d}  "
                    f"MAE={mean_absolute_error(n_true_te[m], n_pred_te[m]):.4f} kHz  "
                    f"MAPE={mape_safe(n_true_te[m], n_pred_te[m]):.1f}%\n")
summary += f"Runtime: {runtime:.1f}s\n"

print(f"\n{'='*60}\n{summary}{'='*60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
