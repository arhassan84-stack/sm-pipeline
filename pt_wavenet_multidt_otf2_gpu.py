"""
pt_wavenet_multidt_otf2_gpu.py — Improved on-the-fly multi-dt WaveNet.

Identical architecture to wn_otf / multidt_B.  Training-regime improvements
over the v1 otf script to exploit the infinite-data advantage:

  1. PATIENCE = 50 > T0 = 30  (critical: prevents early-stop on cosine restart)
  2. T_MULT   = 2             (expanding cycles: 30 → 60 → 120 epochs)
  3. STEPS_PER_EPOCH = 3000   (+50% gradient steps vs v1, only +50% wall-time)
  4. Gradient clipping        (stabilises fresh-data gradient noise)
  5. LR warmup (500 steps)    (avoids destructive early updates)

Why these matter with infinite data
-------------------------------------
  * With finite cached data: PATIENCE can be short because val loss is smooth
    (same samples → deterministic gradients).
  * With fresh batches each step: after a cosine LR restart the val loss
    *temporarily* rises for ~5-15 epochs before reaching a new best.  If
    PATIENCE < T0 the scheduler can fire an early stop inside a restart dip,
    discarding a model that would have improved further.
  * T_MULT=2 gives short exploratory cycles early and a long fine-tuning
    cycle at the end — better suited to an optimizer that never overfits.
  * Gradient clipping (max_norm=1.0) reduces the variance from fresh batches
    without reducing the effective learning rate the way a smaller LR would.

Label: wavenet_multidt_otf2
"""

import numpy as np, math, time, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from features_gpu import make_constants, compute_features_gpu

LABEL            = 'wavenet_multidt_otf2'
CHANNELS         = 256
BATCH_SIZE       = 128
STEPS_PER_EPOCH  = 3000          # ↑ from 2000  (+50% gradient steps/epoch)
MAX_EPOCHS       = 200
PATIENCE         = 50            # ↑ from 25  (must be > T0 to survive restarts)
LR               = 5e-4
WEIGHT_DECAY     = 1e-4
T0               = 30            # cosine cycle length (epochs)
T_MULT           = 2             # NEW: expanding cycles  30 → 60 → 120
GRAD_CLIP        = 1.0           # NEW: gradient clipping max-norm
WARMUP_STEPS     = 500           # NEW: linear LR warmup (steps, not epochs)
DILATIONS        = [1, 2, 4, 8, 16, 32, 64, 128]
POOL010          = 16
EVAL_BATCH       = 256
WARMUP_BATCHES   = 500
VAL_SIZE         = 10_000
S2_CHUNK         = 30

# ── FCS physical constants ─────────────────────────────────────────────────────
EX_LAMBDA = 488.0;  NA = 1.2
WXY       = 1e-3 * 0.51 * EX_LAMBDA / NA
MAX_RATE  = 50_000.0
D0_MIN    = 0.01
D0_MAX    = 14.12537544622754
DT010_S   = 0.1e-3
N010, N050, N100 = 40960, 8192, 4096

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def mape(t, p): m = t > 0; return np.mean(np.abs((p[m] - t[m]) / t[m])) * 100


# ── On-the-fly simulation ─────────────────────────────────────────────────────

def simulate_batch(B: int) -> tuple:
    D = torch.empty(B, device=device).uniform_(
        math.log(D0_MIN), math.log(D0_MAX)).exp()
    step_std = (2.0 * D * DT010_S).sqrt()
    sx = step_std[:, None] * torch.randn(B, N010 - 1, device=device)
    sy = step_std[:, None] * torch.randn(B, N010 - 1, device=device)
    zeros = torch.zeros(B, 1, device=device)
    x = torch.cat([zeros, sx.cumsum(1)], dim=1)
    y = torch.cat([zeros, sy.cumsum(1)], dim=1)
    mid = N010 // 2 - 1
    x = x - x[:, mid:mid + 1]
    y = y - y[:, mid:mid + 1]
    em     = torch.exp(-2.0 * ((x / WXY) ** 2 + (y / WXY) ** 2))
    counts = torch.poisson(DT010_S * MAX_RATE * em)
    I010   = counts / DT010_S
    I050   = I010.reshape(B, N050,  5).mean(2)
    I100   = I010.reshape(B, N100, 10).mean(2)
    return I010, I050, I100, D


def norm_traces(t: torch.Tensor) -> torch.Tensor:
    mu  = t.mean(1, keepdim=True)
    sig = t.std(1,  keepdim=True) + 1e-8
    return (t - mu) / sig


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


class WaveNetMultiDT_B(nn.Module):
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


# ── Feature constants ──────────────────────────────────────────────────────────

log("Precomputing scattering filter banks and lag arrays...")
C010 = make_constants(N010, device)
C050 = make_constants(N050, device)
C100 = make_constants(N100, device)
log(f"  dt010: {N010} bins ({C010['n_lags']} lags, {C010['n_short']} short) | "
    f"dt050: {N050} bins ({C050['n_lags']} lags, {C050['n_short']} short) | "
    f"dt100: {N100} bins ({C100['n_lags']} lags, {C100['n_short']} short)")


# ── Feature scaler warm-up ─────────────────────────────────────────────────────

log(f"Warm-up: {WARMUP_BATCHES} × {BATCH_SIZE} = "
    f"{WARMUP_BATCHES * BATCH_SIZE:,} traces to fit StandardScaler...")
t_wu   = time.time()
wu_buf = []
with torch.no_grad():
    for _ in range(WARMUP_BATCHES):
        I010, I050, I100, _ = simulate_batch(BATCH_SIZE)
        X010 = compute_features_gpu(I010, C010, S2_CHUNK)
        X050 = compute_features_gpu(I050, C050, S2_CHUNK)
        X100 = compute_features_gpu(I100, C100, S2_CHUNK)
        wu_buf.append(torch.cat([X010, X050, X100], dim=1).cpu().float().numpy())

X_wu        = np.concatenate(wu_buf, axis=0).astype(np.float64)
feat_dim    = X_wu.shape[1]
feat_scaler = StandardScaler().fit(X_wu)
del wu_buf, X_wu

feat_mean  = torch.tensor(feat_scaler.mean_,  dtype=torch.float32, device=device)
feat_scale = torch.tensor(feat_scaler.scale_, dtype=torch.float32, device=device)
log(f"  feat_dim={feat_dim}  warm-up done in {time.time() - t_wu:.1f}s")


# ── Validation set ─────────────────────────────────────────────────────────────

log(f"Generating validation set ({VAL_SIZE} traces)...")
t_vs  = time.time()
vs_10=[]; vs_50=[]; vs_100=[]; vs_X=[]; vs_y=[]; vs_D=[]
with torch.no_grad():
    for s in range(0, VAL_SIZE, BATCH_SIZE):
        e = min(s + BATCH_SIZE, VAL_SIZE)
        I010v, I050v, I100v, Dv = simulate_batch(e - s)
        X010v = compute_features_gpu(I010v, C010, S2_CHUNK)
        X050v = compute_features_gpu(I050v, C050, S2_CHUNK)
        X100v = compute_features_gpu(I100v, C100, S2_CHUNK)
        Xv    = ((torch.cat([X010v, X050v, X100v], dim=1) - feat_mean) / feat_scale).clamp(-100, 100)
        vs_10.append(norm_traces(I010v).cpu())
        vs_50.append(norm_traces(I050v).cpu())
        vs_100.append(norm_traces(I100v).cpu())
        vs_X.append(Xv.cpu()); vs_y.append(torch.log(Dv).cpu())
        vs_D.append(Dv.cpu().numpy())

t010_val = torch.cat(vs_10);  t050_val = torch.cat(vs_50)
t100_val = torch.cat(vs_100); X_val = torch.cat(vs_X)
y_val    = torch.cat(vs_y);   D_val = np.concatenate(vs_D)
del vs_10, vs_50, vs_100, vs_X, vs_y, vs_D
log(f"  Validation ready in {time.time() - t_vs:.1f}s")


# ── Model + optimiser ──────────────────────────────────────────────────────────

model   = WaveNetMultiDT_B(feat_dim=feat_dim, ch=CHANNELS).to(device)
opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=T0, T_mult=T_MULT)    # expanding cycles
crit    = nn.MSELoss()
gscaler = GradScaler()

n_params = sum(p.numel() for p in model.parameters())
log(f"Model: {LABEL}  params={n_params:,}  feat_dim={feat_dim}")
log(f"  {STEPS_PER_EPOCH} steps/epoch × up to {MAX_EPOCHS} epochs, patience={PATIENCE}")
log(f"  CosineAnnealingWarmRestarts T0={T0} T_mult={T_MULT}  |  grad_clip={GRAD_CLIP}")
log(f"  LR warmup over first {WARMUP_STEPS} steps  |  AvgPool{POOL010}  |  AMP")

t_total   = time.time()
total_steps = 0


# ── Training loop ─────────────────────────────────────────────────────────────

best_val       = float('inf')
best_state     = {k: v.clone() for k, v in model.state_dict().items()}
patience_count = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        # ── LR warmup (linear ramp over first WARMUP_STEPS) ──
        if total_steps < WARMUP_STEPS:
            scale = (total_steps + 1) / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * scale

        I010, I050, I100, D = simulate_batch(BATCH_SIZE)

        with torch.no_grad():
            X010 = compute_features_gpu(I010, C010, S2_CHUNK)
            X050 = compute_features_gpu(I050, C050, S2_CHUNK)
            X100 = compute_features_gpu(I100, C100, S2_CHUNK)
            X_sc = ((torch.cat([X010, X050, X100], dim=1) - feat_mean) / feat_scale).clamp(-100, 100)

        t010 = norm_traces(I010); t050 = norm_traces(I050); t100 = norm_traces(I100)
        y    = torch.log(D)

        opt.zero_grad()
        with autocast():
            loss = crit(model(t010, t050, t100, X_sc), y)

        if not math.isfinite(loss.item()):
            continue  # skip non-finite loss; no backward, no param update

        gscaler.scale(loss).backward()
        gscaler.step(opt);  gscaler.update()
        ep_loss += loss.item()
        total_steps += 1

    ep_loss /= STEPS_PER_EPOCH
    # Advance cosine schedule only after warmup is complete
    if total_steps >= WARMUP_STEPS:
        sched.step(epoch - 1)

    # ── Validation ──
    model.eval()
    val_preds = []
    with torch.no_grad():
        for s in range(0, VAL_SIZE, EVAL_BATCH):
            e = min(s + EVAL_BATCH, VAL_SIZE)
            val_preds.append(model(
                t010_val[s:e].to(device), t050_val[s:e].to(device),
                t100_val[s:e].to(device), X_val[s:e].to(device)).cpu())
    logd_pred_val = torch.cat(val_preds)
    vl = crit(logd_pred_val, y_val).item()

    if vl < best_val - 1e-6:
        best_val       = vl
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 5 == 0 or patience_count == 0:
        D_pred_v = logd_pred_val.exp().numpy()
        log(f"Epoch {epoch:3d}  train={ep_loss:.5f}  val={vl:.5f}  "
            f"MAPE={mape(D_val, D_pred_v):.1f}%  "
            f"lr={opt.param_groups[0]['lr']:.2e}  patience={patience_count}  "
            f"steps={total_steps:,}")

    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}  (patience={PATIENCE})")
        break

model.load_state_dict(best_state)
log(f"Done training — best val={best_val:.5f}")


# ── Save ───────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), f'model_{LABEL}.pt')
np.save(f'scaler_{LABEL}_mean.npy',  feat_scaler.mean_.astype(np.float32))
np.save(f'scaler_{LABEL}_scale.npy', feat_scaler.scale_.astype(np.float32))


# ── Final evaluation ───────────────────────────────────────────────────────────

model.eval()
final_preds = []
with torch.no_grad():
    for s in range(0, VAL_SIZE, EVAL_BATCH):
        e = min(s + EVAL_BATCH, VAL_SIZE)
        final_preds.append(model(
            t010_val[s:e].to(device), t050_val[s:e].to(device),
            t100_val[s:e].to(device), X_val[s:e].to(device)).cpu())

D_pred  = torch.cat(final_preds).exp().numpy()
runtime = time.time() - t_total
ms_     = D_val < 1.0;  mf_ = D_val >= 1.0

summary = (
    f"Task: {LABEL}\n"
    f"Architecture: 3 WaveNet branches (dt010 AvgPool{POOL010}, dt050, dt100) + "
    f"{feat_dim}-dim features (GPU on-the-fly)\n"
    f"  channels={CHANNELS}  dilations={DILATIONS}\n"
    f"Training: {STEPS_PER_EPOCH} steps/epoch, T0={T0}, T_mult={T_MULT}, "
    f"patience={PATIENCE}, grad_clip={GRAD_CLIP}\n"
    f"  LR warmup: {WARMUP_STEPS} steps  |  AvgPool{POOL010}  |  CUDA streams  |  AMP\n"
    f"Total gradient steps: {total_steps:,}\n"
    f"Params: {n_params:,}\n"
    f"Val size: {VAL_SIZE}  (log-uniform D, n=0 simulation)\n"
    f"Val  R²={r2_score(D_val, D_pred):.4f}  "
    f"MAE={mean_absolute_error(D_val, D_pred):.4f}  "
    f"MAPE={mape(D_val, D_pred):.1f}%\n"
    f"  d <1  ({ms_.sum():5d}): R²={r2_score(D_val[ms_], D_pred[ms_]):.4f}  "
    f"MAPE={mape(D_val[ms_], D_pred[ms_]):.1f}%\n"
    f"  d>=1  ({mf_.sum():5d}): R²={r2_score(D_val[mf_], D_pred[mf_]):.4f}  "
    f"MAPE={mape(D_val[mf_], D_pred[mf_]):.1f}%\n"
    f"Runtime: {runtime:.1f}s\n"
)
print(f"\n{'=' * 60}\n{summary}{'=' * 60}")
with open(f'results_{LABEL}.txt', 'w') as f:
    f.write(summary)
log(f"Saved results_{LABEL}.txt")
log("Done.")
