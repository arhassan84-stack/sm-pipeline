#!/usr/bin/env python3
"""
plot_noise_v2_check.py
Visualise wavenet_noise_w512_v2 background predictions on real FCS+RAW data.

For 6 example files × 2 channels (S1/S2), each panel shows:
  • Raw 1ms intensity trace (green on black)
  • Per-window n_abs prediction  (orange step function, one value per 64ms stride)
  • Final n_abs = 20th-percentile  (red dashed horizontal line)

Output: results_01082026/noise_v2_check.png
"""

import sys
from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from fcs_io  import parse_fcs_file
from raw_io  import read_arrivals, bin_arrivals

# ── Paths ─────────────────────────────────────────────────────────────────────
FCS_DIR  = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_PATH = PROJECT / 'results_01082026' / 'noise_v2_check.png'
MODEL_PT = PROJECT / 'model_wavenet_noise_w512_v2.pt'
RS_MEAN  = PROJECT / 'raw_stat_wavenet_noise_w512_v2_mean.npy'
RS_STD   = PROJECT / 'raw_stat_wavenet_noise_w512_v2_std.npy'

FILES_TO_PLOT = ['nt_dorsal_1', 'nt_dorsal_5', 'nt_dorsal_10',
                 'nt_dorsal_11', 'nt_dorsal_15', 'nt_dorsal_18']
CHANNELS = [('S1', 0), ('S2', 1)]   # (label, FCS channel index)

RAW_OPTS = dict(bits=16, n_header=48, clock_rate_hz=15_000_000)
DT_MIN   = 0.1   # ms

# ── Window geometry ────────────────────────────────────────────────────────────
WINDOW_MS = 512
STRIDE_MS = WINDOW_MS // 8   # 64 ms
W010      = WINDOW_MS * 10   # 5120 bins at 0.1ms
W050      = WINDOW_MS *  2   # 1024 bins at 0.5ms
W100      = WINDOW_MS        #  512 bins at 1.0ms
POOL010   = WINDOW_MS // 256  # 2
N_SCALE   = 1.0

# ── Aesthetics ─────────────────────────────────────────────────────────────────
S1_COLOR   = '#00FF00'   # bright green
S2_COLOR   = '#FF4444'   # red
BG_COLOR   = 'black'
FG_COLOR   = 'white'
SPINE_LW   = 1.5
FS_SUPTITLE = 18
FS_TITLE   = 12
FS_LABEL   = 13
FS_TICK    = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ── Model definition ──────────────────────────────────────────────────────────

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
    def __init__(self, channels=256, dilations=None):
        super().__init__()
        if dilations is None: dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.branch010   = WaveNetBackbone(channels, dilations, avgpool=POOL010)
        self.branch050   = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100   = WaveNetBackbone(channels, dilations, avgpool=1)
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

    def forward(self, t010, t050, t100, feats):
        e010 = self.branch010(t010)
        e050 = self.branch050(t050)
        e100 = self.branch100(t100)
        f    = self.feat_branch(feats)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


# ── Load model ────────────────────────────────────────────────────────────────
print('Loading model...')
model = WaveNetNoise_W().to(device)
model.load_state_dict(torch.load(MODEL_PT, map_location=device))
model.eval()

raw_stat_mean = np.load(RS_MEAN).astype(np.float32)
raw_stat_std  = np.load(RS_STD ).astype(np.float32)
print(f'  raw_stat_mean (kHz): {np.round(raw_stat_mean, 2)}')


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_fcs_channel(stem, ch_idx):
    """Return 1ms FCS intensity trace (kHz) for channel ch_idx."""
    fcs = parse_fcs_file(FCS_DIR / f'{stem}.fcs')
    return fcs.intensity_trace(channel=ch_idx)   # (N,) kHz


def load_raw_channel(stem, ch_label):
    """
    Find all RAW files for this stem + channel, bin at DT_MIN=0.1ms,
    concatenate → i010 (kHz).  Derive i050 (kHz) by block-summing.
    Returns (i010, i050) or (None, None) if no RAW found.
    """
    pattern = str(FCS_DIR / f'{stem}_*_Ch{ch_label}.raw')
    raw_files = sorted(glob.glob(pattern))
    if not raw_files:
        print(f'  No RAW files for {stem} Ch{ch_label}')
        return None, None

    segments = []
    for fp in raw_files:
        arr = read_arrivals(fp, **RAW_OPTS)
        seg = bin_arrivals(arr, bin_ms=DT_MIN).astype(np.float64)
        segments.append(seg)

    # bin_arrivals returns counts per bin (not count rates).
    # To convert to kHz count rate: multiply by 1/(bin_width_s) / 1000
    #   dt=0.1ms → ×10,  dt=0.5ms → ×2,  dt=1.0ms → ×1  (kHz coincides with counts/ms)
    KHZ_FACTOR = 1.0 / (DT_MIN * 1e-3) / 1e3   # = 10 for DT_MIN=0.1ms
    i010 = np.concatenate(segments) * KHZ_FACTOR   # kHz

    B050 = round(0.5 / DT_MIN)       # 5
    trim = (len(i010) // B050) * B050
    i010 = i010[:trim]
    # mean over B050 bins preserves count rate → i050 in same kHz units as i010
    i050 = i010.reshape(-1, B050).mean(axis=1).astype(np.float64)   # kHz
    return i010, i050


# ── Sliding-window inference ───────────────────────────────────────────────────

def run_sliding_window(i010, i050, i100):
    """
    Slide a WINDOW_MS window across the trace.
    Returns:
      t_centers : (M,) float — window centre times in seconds
      n_hats    : (M,) float — predicted n_abs per window (kHz)
    """
    N = len(i100)
    t_centers = []
    n_hats    = []

    with torch.no_grad():
        for t0 in range(0, N - WINDOW_MS + 1, STRIDE_MS):
            t1 = t0 + WINDOW_MS

            w100 = i100[t0:t1].astype(np.float32)
            w050 = i050[t0 * 2 : t1 * 2].astype(np.float32)
            w010 = i010[t0 * 10 : t1 * 10].astype(np.float32)

            if len(w100) < WINDOW_MS or len(w050) < W050 or len(w010) < W010:
                continue

            # z-score normalise each channel (same as training aug without shift/noise)
            def znorm(a):
                return (a - a.mean()) / (a.std() + 1e-8)

            a100 = torch.tensor(znorm(w100)).unsqueeze(0).to(device)  # (1, W100)
            a050 = torch.tensor(znorm(w050)).unsqueeze(0).to(device)  # (1, W050)
            a010 = torch.tensor(znorm(w010)).unsqueeze(0).to(device)  # (1, W010)

            raw   = np.array([w010.mean(), w010.std(),
                               w050.mean(), w050.std(),
                               w100.mean(), w100.std()], dtype=np.float32)
            raw_sc = torch.tensor((raw - raw_stat_mean) / raw_stat_std).unsqueeze(0).to(device)

            n_hat = model(a010, a050, a100, raw_sc).item() * N_SCALE   # kHz
            t_centers.append((t0 + WINDOW_MS / 2) * 1e-3)              # s
            n_hats.append(n_hat)

    return np.array(t_centers), np.array(n_hats)


# ── Plot ──────────────────────────────────────────────────────────────────────

n_files = len(FILES_TO_PLOT)
fig, axes = plt.subplots(n_files, 2,
                         figsize=(18, n_files * 2.8),
                         facecolor=BG_COLOR, squeeze=False)

fig.suptitle(
    'wavenet_noise_w512_v2  —  raw trace (green) + per-window n̂_abs (orange) + '
    'final n_abs 20th-pct (red dashed)',
    color=FG_COLOR, fontsize=FS_SUPTITLE, y=1.01,
)

for row_i, stem in enumerate(FILES_TO_PLOT):
    print(f'\n{stem}')

    for col_i, (ch_label, ch_idx) in enumerate(CHANNELS):
        ax = axes[row_i, col_i]
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW); sp.set_edgecolor(FG_COLOR)
        ax.tick_params(colors=FG_COLOR, labelsize=FS_TICK)

        curve_color = S1_COLOR if col_i == 0 else S2_COLOR

        # Load traces
        i100 = load_fcs_channel(stem, ch_idx)
        i010, i050 = load_raw_channel(stem, ch_label)

        if i010 is None:
            ax.text(0.5, 0.5, 'No RAW data', color=FG_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{stem}  {ch_label}', color=FG_COLOR, fontsize=FS_TITLE)
            continue

        # Align lengths (RAW may be slightly longer/shorter than FCS)
        N = min(len(i100), len(i050) // 2, len(i010) // 10)
        i100 = i100[:N]
        i050 = i050[:N * 2]
        i010 = i010[:N * 10]

        t_s = np.arange(N) * 1e-3   # ms → s

        # Raw trace
        ax.plot(t_s, i100, color=curve_color, lw=0.4, alpha=0.85)

        # Sliding-window inference
        t_centers, n_hats = run_sliding_window(i010, i050, i100)
        print(f'  {ch_label}: {len(n_hats)} windows, '
              f'n_hat range [{n_hats.min():.1f}, {n_hats.max():.1f}] kHz')

        if len(n_hats) > 0:
            # Step function: draw each window estimate as a horizontal segment
            half_stride = STRIDE_MS / 2 * 1e-3   # s
            for tc, nh in zip(t_centers, n_hats):
                ax.plot([tc - half_stride, tc + half_stride], [nh, nh],
                        color='orange', lw=1.2, alpha=0.7, solid_capstyle='butt')

            # Final n_abs = 20th percentile
            n_abs = float(np.percentile(n_hats, 20))
            ax.axhline(n_abs, color='red', lw=1.4, ls='--',
                       label=f'n_abs = {n_abs:.2f} kHz  (20th pct)')
            ax.legend(fontsize=FS_TICK - 1, framealpha=0.35,
                      labelcolor=FG_COLOR, facecolor=BG_COLOR, edgecolor=FG_COLOR)

        ax.set_title(f'{stem}  {ch_label}', color=FG_COLOR, fontsize=FS_TITLE)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('kHz', color=FG_COLOR, fontsize=FS_LABEL)
        if row_i == n_files - 1:
            ax.set_xlabel('Time (s)', color=FG_COLOR, fontsize=FS_LABEL)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig)
print(f'\nSaved → {OUT_PATH}')
