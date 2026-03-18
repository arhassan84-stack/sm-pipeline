"""
Test-Time Augmentation (TTA) for all trained single-dt WaveNet models.

For each model, creates 7 circularly-shifted versions of every test trace,
averages the log(D) predictions, then reports MAPE before and after TTA.

The architecture's AdaptiveAvgPool1d(1) (global average) makes predictions
largely shift-invariant; averaging over shifts reduces variance from
Conv1d boundary-padding effects.

Models:
  wavenet_wide_aug         dt=1.0ms  feat=283  test=9502
  pt_wavenet_wide_aug_dt020  dt=0.2ms  feat=304  test=28556
  pt_wavenet_wide_aug_dt025  dt=0.25ms feat=304  test=28556
  pt_wavenet_wide_aug_dt040  dt=0.4ms  feat=304  test=28556
  pt_wavenet_wide_aug_dt050  dt=0.5ms  feat=303  test=28556
  pt_wavenet_wide_aug_dt060  dt=0.6ms  feat=303  test=28556
  pt_wavenet_wide_aug_dt080  dt=0.8ms  feat=303  test=28556
"""

import numpy as np
import torch
import torch.nn as nn
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mape(true, pred):
    m = true > 0
    return np.mean(np.abs((pred[m] - true[m]) / true[m])) * 100


# ── Architecture (identical across all dt models) ─────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn   = nn.BatchNorm1d(channels)
    def forward(self, x):
        # residual added BEFORE activation — matches original training scripts
        return torch.relu(self.bn(self.conv(x)) + x)

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


# ── Model configurations ───────────────────────────────────────────────────────

CONFIGS = [
    dict(
        label    = 'wavenet_wide_aug',
        dt_ms    = 1.0,
        n_bins   = 4096,
        model_f  = 'model_wavenet_wide_aug.pt',
        mean_f   = 'scaler_wavenet_wide_aug_mean.npy',
        scale_f  = 'scaler_wavenet_wide_aug_scale.npy',
        i_test_f = 'cache_i_test_90pct.npy',
        x_test_f = 'cache_X_test_90pct.npy',
        d_test_f = 'cache_d_test_90pct.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt020',
        dt_ms    = 0.2,
        n_bins   = 20480,
        model_f  = 'model_pt_wavenet_wide_aug_dt020.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt020_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt020_scale.npy',
        i_test_f = 'cache_i_test_dt020.npy',
        x_test_f = 'cache_X_test_dt020.npy',
        d_test_f = 'cache_d_test_dt020.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt025',
        dt_ms    = 0.25,
        n_bins   = 16384,
        model_f  = 'model_pt_wavenet_wide_aug_dt025.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt025_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt025_scale.npy',
        i_test_f = 'cache_i_test_dt025.npy',
        x_test_f = 'cache_X_test_dt025.npy',
        d_test_f = 'cache_d_test_dt025.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt040',
        dt_ms    = 0.4,
        n_bins   = 10240,
        model_f  = 'model_pt_wavenet_wide_aug_dt040.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt040_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt040_scale.npy',
        i_test_f = 'cache_i_test_dt040.npy',
        x_test_f = 'cache_X_test_dt040.npy',
        d_test_f = 'cache_d_test_dt040.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt050',
        dt_ms    = 0.5,
        n_bins   = 8192,
        model_f  = 'model_pt_wavenet_wide_aug_dt050.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt050_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt050_scale.npy',
        i_test_f = 'cache_i_test_dt050.npy',
        x_test_f = 'cache_X_test_dt050.npy',
        d_test_f = 'cache_d_test_dt050.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt060',
        dt_ms    = 0.6,
        n_bins   = 6827,
        model_f  = 'model_pt_wavenet_wide_aug_dt060.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt060_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt060_scale.npy',
        i_test_f = 'cache_i_test_dt060.npy',
        x_test_f = 'cache_X_test_dt060.npy',
        d_test_f = 'cache_d_test_dt060.npy',
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt080',
        dt_ms    = 0.8,
        n_bins   = 5120,
        model_f  = 'model_pt_wavenet_wide_aug_dt080.pt',
        mean_f   = 'scaler_pt_wavenet_wide_aug_dt080_mean.npy',
        scale_f  = 'scaler_pt_wavenet_wide_aug_dt080_scale.npy',
        i_test_f = 'cache_i_test_dt080.npy',
        x_test_f = 'cache_X_test_dt080.npy',
        d_test_f = 'cache_d_test_dt080.npy',
    ),
]

DILATIONS  = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_SIZE = 256
D_MAX      = 10.0
N_SHIFTS   = 7      # [0, ±s, ±2s, ±3s]  where s = round(N_BINS/32)
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

summary_lines = []

for cfg in CONFIGS:
    label = cfg['label']
    log(f"\n{'='*60}")
    log(f"TTA: {label}  dt={cfg['dt_ms']}ms  n_bins={cfg['n_bins']}")

    # ── Load test data ────────────────────────────────────────────────────────
    log("Loading test caches...")
    i_test_raw = np.load(cfg['i_test_f']).astype(np.float32)
    X_test_raw = np.load(cfg['x_test_f']).astype(np.float32)
    d_test_raw = np.load(cfg['d_test_f'])

    mask      = d_test_raw <= D_MAX
    i_test    = i_test_raw[mask]
    X_test    = X_test_raw[mask]
    d_test    = d_test_raw[mask].astype(np.float32)
    n         = len(d_test)
    log(f"Test samples after d<={D_MAX} mask: {n}")

    # ── Apply feature scaler ──────────────────────────────────────────────────
    mean_  = np.load(cfg['mean_f'])
    scale_ = np.load(cfg['scale_f'])
    X_te_sc = ((X_test - mean_) / scale_).astype(np.float32)
    feat_dim = X_te_sc.shape[1]
    X_t = torch.tensor(X_te_sc).to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    model = WaveNet1D(feat_dim=feat_dim, channels=256, dilations=DILATIONS)
    model.load_state_dict(torch.load(cfg['model_f'], map_location='cpu'))
    model.eval().to(device)
    log(f"Loaded {cfg['model_f']}  feat_dim={feat_dim}")

    # ── Build shift list: [0, -s, +s, -2s, +2s, -3s, +3s] ───────────────────
    s = round(cfg['n_bins'] / 32)
    shifts = [0]
    for k in range(1, (N_SHIFTS + 1) // 2):
        shifts += [-k * s, k * s]
    shifts = shifts[:N_SHIFTS]
    log(f"Shifts (s={s}): {shifts}")

    # ── Baseline prediction (shift=0) ─────────────────────────────────────────
    baseline_preds = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            chunk = i_test[i:i+BATCH_SIZE]
            mu  = chunk.mean(axis=1, keepdims=True)
            sig = chunk.std(axis=1,  keepdims=True) + 1e-8
            i_n = torch.tensor((chunk - mu) / sig).to(device)
            baseline_preds.append(model(i_n, X_t[i:i+BATCH_SIZE]).cpu().numpy())
    logd_base = np.concatenate(baseline_preds)

    # ── TTA: accumulate predictions across all shifts ─────────────────────────
    sum_logd = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for shift in shifts:
            shift_preds = []
            for i in range(0, n, BATCH_SIZE):
                chunk = i_test[i:i+BATCH_SIZE].copy()
                if shift != 0:
                    chunk = np.roll(chunk, shift, axis=1)
                mu  = chunk.mean(axis=1, keepdims=True)
                sig = chunk.std(axis=1,  keepdims=True) + 1e-8
                i_n = torch.tensor((chunk - mu) / sig).to(device)
                shift_preds.append(model(i_n, X_t[i:i+BATCH_SIZE]).cpu().numpy())
            sum_logd += np.concatenate(shift_preds)
    logd_tta = (sum_logd / N_SHIFTS).astype(np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────
    d_pred_base = np.exp(logd_base)
    d_pred_tta  = np.exp(logd_tta)

    m_slow = d_test < 1
    m_fast = d_test >= 1

    mape_base      = mape(d_test, d_pred_base)
    mape_tta       = mape(d_test, d_pred_tta)
    mape_slow_base = mape(d_test[m_slow], d_pred_base[m_slow])
    mape_slow_tta  = mape(d_test[m_slow], d_pred_tta[m_slow])
    mape_fast_base = mape(d_test[m_fast], d_pred_base[m_fast])
    mape_fast_tta  = mape(d_test[m_fast], d_pred_tta[m_fast])

    log(f"Baseline MAPE={mape_base:.2f}%  d<1={mape_slow_base:.2f}%  d>=1={mape_fast_base:.2f}%")
    log(f"TTA      MAPE={mape_tta:.2f}%  d<1={mape_slow_tta:.2f}%  d>=1={mape_fast_tta:.2f}%")
    log(f"Delta:  {mape_base - mape_tta:+.3f}%  (negative = improvement)")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f'pred_logd_{label}_test_tta.npy', logd_tta)

    res = (
        f"Task: {label}_tta\n"
        f"dt={cfg['dt_ms']}ms  N_shifts={N_SHIFTS}  shifts={shifts}\n"
        f"Test: {n}\n"
        f"Baseline MAPE={mape_base:.2f}%  d<1={mape_slow_base:.2f}%  d>=1={mape_fast_base:.2f}%\n"
        f"TTA      MAPE={mape_tta:.2f}%  d<1={mape_slow_tta:.2f}%  d>=1={mape_fast_tta:.2f}%\n"
        f"Delta MAPE: {mape_base - mape_tta:+.3f}%\n"
    )
    with open(f'results_{label}_tta.txt', 'w') as f:
        f.write(res)
    log(f"Saved results_{label}_tta.txt  pred_logd_{label}_test_tta.npy")

    summary_lines.append(
        f"dt={cfg['dt_ms']}ms  {label}:\n"
        f"  baseline {mape_base:.2f}% -> TTA {mape_tta:.2f}%  "
        f"(d<1: {mape_slow_base:.2f}->{mape_slow_tta:.2f}%  "
        f"d>=1: {mape_fast_base:.2f}->{mape_fast_tta:.2f}%)"
    )

    del i_test_raw, X_test_raw, i_test, X_test, X_te_sc, X_t, model

log(f"\n{'='*60}")
log("TTA SUMMARY")
log('='*60)
for line in summary_lines:
    log(line)
