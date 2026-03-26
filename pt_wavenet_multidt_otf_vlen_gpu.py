"""
pt_wavenet_multidt_otf_vlen_gpu.py — Variable-window on-the-fly multi-dt WaveNet.

Simulates full 4096ms FCS traces on GPU, then extracts a centred window of
randomly sampled duration T_ms at each training batch.

Sampling distribution:
    Uniform over T_GRID_MS — each duration gets equal representation.

Crop:    start = (N_FULL − N_win) // 2   (centred)

Features:
    Reference lag grid (from full-trace constants) is reused for all window
    sizes so feat_dim stays constant regardless of T_ms.
    An extra feature  log(T_ms / 4096)  is appended as a duration indicator.

Label:   wavenet_multidt_otf_vlen
"""

import random, numpy as np, math, time, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from features_gpu import make_constants, compute_features_gpu

LABEL           = 'wavenet_multidt_otf_vlen'
CHANNELS        = 256
BATCH_SIZE      = 128
STEPS_PER_EPOCH = 2000
MAX_EPOCHS      = 200
PATIENCE        = 50
LR              = 5e-4
WEIGHT_DECAY    = 1e-4
T0              = 30
T_MULT          = 2
DILATIONS       = [1, 2, 4, 8, 16, 32, 64, 128]
POOL010         = 16
EVAL_BATCH      = 256
WARMUP_BATCHES  = 500
S2_CHUNK        = 30

# Window grid (ms) — N010_w = T*10, must be divisible by 16 for AvgPool16
T_GRID_MS  = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

# Validation: separate held-out sets per representative duration
VAL_T_GRID  = [128, 256, 512, 1024, 4096]
VAL_PER_T   = 2000                          # traces per validation duration

# FCS physical constants
EX_LAMBDA = 488.0;  NA = 1.2
WXY       = 1e-3 * 0.51 * EX_LAMBDA / NA   # 0.2074 µm
MAX_RATE  = 50_000.0
D0_MIN    = 0.01
D0_MAX    = 14.12537544622754
DT010_S   = 0.1e-3

# Full-trace dimensions
N010_FULL = 40960
N050_FULL = 8192
N100_FULL = 4096

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── Duration sampler ──────────────────────────────────────────────────────────

def sample_duration_ms() -> int:
    """Uniform over T_GRID_MS — equal representation for all durations."""
    return random.choice(T_GRID_MS)


# ── Full-trace simulation ─────────────────────────────────────────────────────

def simulate_batch_full(B: int) -> tuple:
    """Simulate B full-length (4096ms) FCS traces on GPU.

    Returns: I010_full (B, N010_FULL) float32, D (B,) float32
    """
    D = torch.empty(B, device=device).uniform_(
        math.log(D0_MIN), math.log(D0_MAX)).exp()

    step_std = (2.0 * D * DT010_S).sqrt()
    sx = step_std[:, None] * torch.randn(B, N010_FULL - 1, device=device)
    sy = step_std[:, None] * torch.randn(B, N010_FULL - 1, device=device)

    zeros = torch.zeros(B, 1, device=device)
    x = torch.cat([zeros, sx.cumsum(1)], dim=1)
    y = torch.cat([zeros, sy.cumsum(1)], dim=1)

    mid = N010_FULL // 2 - 1
    x = x - x[:, mid:mid + 1]
    y = y - y[:, mid:mid + 1]

    em     = torch.exp(-2.0 * ((x / WXY) ** 2 + (y / WXY) ** 2))
    counts = torch.poisson(DT010_S * MAX_RATE * em)
    I010   = counts / DT010_S
    return I010, D


def crop_to_window(I010_full: torch.Tensor, T_ms: int) -> tuple:
    """Centre-crop I010_full to a window of T_ms duration; derive I050, I100."""
    B      = I010_full.shape[0]
    N010_w = T_ms * 10
    N050_w = N010_w // 5
    N100_w = N010_w // 10
    start  = (N010_FULL - N010_w) // 2
    I010_w = I010_full[:, start: start + N010_w]
    I050_w = I010_w.reshape(B, N050_w,  5).mean(2)
    I100_w = I010_w.reshape(B, N100_w, 10).mean(2)
    return I010_w, I050_w, I100_w


def norm_traces(t: torch.Tensor) -> torch.Tensor:
    mu  = t.mean(1, keepdim=True)
    sig = t.std(1,  keepdim=True) + 1e-8
    return (t - mu) / sig


# ── Per-window constants (reference lag grid overridden for consistent feat_dim)

log("Precomputing reference (full-trace) constants for lag grid...")
C010_REF = make_constants(N010_FULL, device)
C050_REF = make_constants(N050_FULL, device)
C100_REF = make_constants(N100_FULL, device)
log(f"  ref n_lags: dt010={C010_REF['n_lags']}, "
    f"dt050={C050_REF['n_lags']}, dt100={C100_REF['n_lags']}")


def _override_lags(Cw: dict, Cref: dict) -> dict:
    """Replace lag arrays in Cw with reference values to maintain fixed feat_dim."""
    for key in ('lags', 'short_lags', 'n_lags', 'n_short'):
        Cw[key] = Cref[key]
    return Cw


log(f"Caching window constants for T_GRID_MS={T_GRID_MS} ms...")
WIN_CONSTS: dict = {}
for T in T_GRID_MS:
    N010_w = T * 10;  N050_w = N010_w // 5;  N100_w = N010_w // 10
    WIN_CONSTS[T] = (
        _override_lags(make_constants(N010_w, device), C010_REF),
        _override_lags(make_constants(N050_w, device), C050_REF),
        _override_lags(make_constants(N100_w, device), C100_REF),
    )
log("  Constants cache ready.")


def compute_batch_features(I010_w: torch.Tensor,
                           I050_w: torch.Tensor,
                           I100_w: torch.Tensor,
                           T_ms:   int) -> torch.Tensor:
    """Compute features + log(T_ms/4096) duration indicator."""
    B = I010_w.shape[0]
    C010_w, C050_w, C100_w = WIN_CONSTS[T_ms]
    X010 = compute_features_gpu(I010_w, C010_w, S2_CHUNK)
    X050 = compute_features_gpu(I050_w, C050_w, S2_CHUNK)
    X100 = compute_features_gpu(I100_w, C100_w, S2_CHUNK)
    t_ind = torch.full((B, 1), math.log(T_ms / 4096.0), device=device)
    return torch.cat([X010, X050, X100, t_ind], dim=1)


# ── Model ─────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, dilation=d, padding=d)
        self.bn   = nn.BatchNorm1d(ch);  self.act = nn.ReLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)) + x)


class WaveNetBackbone(nn.Module):
    def __init__(self, ch=256, dilations=None, avgpool=1):
        super().__init__()
        d = dilations or DILATIONS
        proj = []
        if avgpool > 1:
            proj.append(nn.AvgPool1d(avgpool, avgpool))
        proj += [nn.Conv1d(1, ch, 16, 4, 6), nn.BatchNorm1d(ch), nn.ReLU()]
        self.proj   = nn.Sequential(*proj)
        self.blocks = nn.Sequential(*[DilatedResBlock(ch, d_) for d_ in d])
        self.gap    = nn.AdaptiveAvgPool1d(1)

    def forward(self, t):
        return self.gap(self.blocks(self.proj(t.unsqueeze(1)))).squeeze(2)


class WaveNetMultiDT_VLen(nn.Module):
    """WaveNetMultiDT_B with variable-length-aware feature branch."""
    def __init__(self, feat_dim: int, ch: int = 256):
        super().__init__()
        self.b010 = WaveNetBackbone(ch, avgpool=POOL010)
        self.b050 = WaveNetBackbone(ch)
        self.b100 = WaveNetBackbone(ch)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(ch * 3 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.s010 = torch.cuda.Stream()
        self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        cur = torch.cuda.current_stream()
        self.s010.wait_stream(cur);  self.s050.wait_stream(cur)
        with torch.cuda.stream(self.s010): e010 = self.b010(t010)
        with torch.cuda.stream(self.s050): e050 = self.b050(t050)
        e100 = self.b100(t100);  f = self.feat_branch(feats)
        cur.wait_stream(self.s010);  cur.wait_stream(self.s050)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


# ── Feature scaler warm-up (diverse window durations) ────────────────────────

log(f"Warm-up: {WARMUP_BATCHES} × {BATCH_SIZE} traces (diverse windows) "
    f"to fit StandardScaler...")
t_wu   = time.time()
wu_buf = []
with torch.no_grad():
    for _ in range(WARMUP_BATCHES):
        T_ms              = sample_duration_ms()
        I010_full, _      = simulate_batch_full(BATCH_SIZE)
        I010_w, I050_w, I100_w = crop_to_window(I010_full, T_ms)
        del I010_full
        wu_buf.append(
            compute_batch_features(I010_w, I050_w, I100_w, T_ms)
            .cpu().float().numpy())

X_wu        = np.concatenate(wu_buf, axis=0).astype(np.float64)
feat_dim    = X_wu.shape[1]
feat_scaler = StandardScaler().fit(X_wu)
del wu_buf, X_wu

feat_mean  = torch.tensor(feat_scaler.mean_,  dtype=torch.float32, device=device)
feat_scale = torch.tensor(feat_scaler.scale_, dtype=torch.float32, device=device)
log(f"  feat_dim={feat_dim}  warm-up done in {time.time() - t_wu:.1f}s")


# ── Validation sets (one per representative duration, fixed once) ─────────────

log(f"Generating validation sets: {VAL_PER_T} traces × {VAL_T_GRID} ms ...")
t_vs     = time.time()
val_sets = {}  # T_ms → dict(t010, t050, t100, X, y, D)

for T_ms in VAL_T_GRID:
    v10=[]; v50=[]; v100=[]; vX=[]; vy=[]; vD=[]
    with torch.no_grad():
        for s in range(0, VAL_PER_T, BATCH_SIZE):
            e = min(s + BATCH_SIZE, VAL_PER_T)
            I010_full, Dv    = simulate_batch_full(e - s)
            I010_w, I050_w, I100_w = crop_to_window(I010_full, T_ms)
            del I010_full
            Xv = (compute_batch_features(I010_w, I050_w, I100_w, T_ms)
                  - feat_mean) / feat_scale
            v10.append(norm_traces(I010_w).cpu())
            v50.append(norm_traces(I050_w).cpu())
            v100.append(norm_traces(I100_w).cpu())
            vX.append(Xv.cpu())
            vy.append(torch.log(Dv).cpu())
            vD.append(Dv.cpu().numpy())
    val_sets[T_ms] = dict(
        t010=torch.cat(v10), t050=torch.cat(v50), t100=torch.cat(v100),
        X=torch.cat(vX), y=torch.cat(vy), D=np.concatenate(vD))
    log(f"  T={T_ms:5d}ms — {VAL_PER_T} traces ready")

log(f"  All validation sets ready in {time.time() - t_vs:.1f}s")


# ── Model + optimiser ─────────────────────────────────────────────────────────

model   = WaveNetMultiDT_VLen(feat_dim=feat_dim, ch=CHANNELS).to(device)
opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=T_MULT)
crit    = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
log(f"Model: {LABEL}  params={n_params:,}  feat_dim={feat_dim}")
log(f"  {STEPS_PER_EPOCH} steps/epoch × up to {MAX_EPOCHS} epochs, patience={PATIENCE}")
log(f"  CosineAnnealingWarmRestarts T0={T0} T_mult={T_MULT}  |  AvgPool{POOL010} on dt010 | CUDA streams")
log(f"  Sampling: uniform over T_GRID_MS={T_GRID_MS}")

t_total = time.time()


# ── Training loop ─────────────────────────────────────────────────────────────

best_val       = float('inf')
best_state     = {k: v.clone() for k, v in model.state_dict().items()}
patience_count = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        # Sample a window duration for this batch
        T_ms              = sample_duration_ms()
        I010_full, D      = simulate_batch_full(BATCH_SIZE)
        I010_w, I050_w, I100_w = crop_to_window(I010_full, T_ms)
        del I010_full

        with torch.no_grad():
            X_sc = ((compute_batch_features(I010_w, I050_w, I100_w, T_ms)
                     - feat_mean) / feat_scale)

        t010 = norm_traces(I010_w)
        t050 = norm_traces(I050_w)
        t100 = norm_traces(I100_w)
        y    = torch.log(D)

        opt.zero_grad()
        loss = crit(model(t010, t050, t100, X_sc), y)
        loss.backward()
        opt.step()
        ep_loss += loss.item()

    ep_loss /= STEPS_PER_EPOCH
    sched.step(epoch - 1)

    # ── Validation across all representative durations ──
    model.eval()
    vl_parts = []
    with torch.no_grad():
        for T_ms, vs in val_sets.items():
            preds = []
            for s in range(0, VAL_PER_T, EVAL_BATCH):
                e = min(s + EVAL_BATCH, VAL_PER_T)
                preds.append(model(
                    vs['t010'][s:e].to(device),
                    vs['t050'][s:e].to(device),
                    vs['t100'][s:e].to(device),
                    vs['X'][s:e].to(device)).cpu())
            vl_parts.append(crit(torch.cat(preds), vs['y']).item())
    vl = float(np.mean(vl_parts))

    if vl < best_val - 1e-6:
        best_val       = vl
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 5 == 0 or patience_count == 0:
        # Quick MAPE on the 512ms set (representative mid-range)
        with torch.no_grad():
            vs512 = val_sets[512]
            pv512 = []
            for s in range(0, VAL_PER_T, EVAL_BATCH):
                e = min(s + EVAL_BATCH, VAL_PER_T)
                pv512.append(model(
                    vs512['t010'][s:e].to(device),
                    vs512['t050'][s:e].to(device),
                    vs512['t100'][s:e].to(device),
                    vs512['X'][s:e].to(device)).cpu())
        D_pred_512 = torch.cat(pv512).exp().numpy()
        log(f"Epoch {epoch:3d}  train={ep_loss:.5f}  val={vl:.5f}  "
            f"MAPE@512ms={mape(vs512['D'], D_pred_512):.1f}%  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}")

    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.5f}")


# ── Save ──────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_.astype(np.float32))
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_.astype(np.float32))


# ── Final evaluation per duration ─────────────────────────────────────────────

model.eval()
runtime = time.time() - t_total
lines   = [
    f"Task: {LABEL}",
    f"Architecture: 3 WaveNet branches (dt010 AvgPool{POOL010}, dt050, dt100) + "
    f"{feat_dim}-dim features (GPU on-the-fly, reference lag grid)",
    f"  channels={CHANNELS}  dilations={DILATIONS}",
    f"  Sampling: uniform over T_GRID_MS",
    f"Params: {n_params:,}",
    f"Runtime: {runtime:.1f}s",
    "",
    f"{'T_ms':>8}  {'N':>6}  {'R2':>7}  {'MAE':>7}  {'MAPE':>7}  "
    f"{'MAPE<1':>8}  {'MAPE>=1':>8}",
]
with torch.no_grad():
    for T_ms, vs in sorted(val_sets.items()):
        preds = []
        for s in range(0, VAL_PER_T, EVAL_BATCH):
            e = min(s + EVAL_BATCH, VAL_PER_T)
            preds.append(model(
                vs['t010'][s:e].to(device), vs['t050'][s:e].to(device),
                vs['t100'][s:e].to(device), vs['X'][s:e].to(device)).cpu())
        D_pred = torch.cat(preds).exp().numpy()
        D_true = vs['D']
        ms_    = D_true < 1.0;  mf_ = D_true >= 1.0
        lines.append(
            f"{T_ms:>8}ms  {VAL_PER_T:>6}  "
            f"{r2_score(D_true, D_pred):>7.4f}  "
            f"{mean_absolute_error(D_true, D_pred):>7.4f}  "
            f"{mape(D_true, D_pred):>6.1f}%  "
            f"{mape(D_true[ms_], D_pred[ms_]):>7.1f}%  "
            f"{mape(D_true[mf_], D_pred[mf_]):>8.1f}%")

summary = "\n".join(lines) + "\n"
print(f"\n{'=' * 70}\n{summary}{'=' * 70}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
