#!/usr/bin/env python3
"""
extract_noise_levels.py
Run wavenet_noise_w512_v2 sliding-window inference on the 6 example files
and save per-file/channel n_abs (20th pct) to a JSON file.

Output: results_01082026/noise_levels.json
"""

import sys, json, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from fcs_io import parse_fcs_file
from raw_io  import read_arrivals, bin_arrivals

FCS_DIR  = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_JSON = PROJECT / 'results_01082026' / 'noise_levels.json'
MODEL_PT = PROJECT / 'model_wavenet_noise_w512_v2.pt'
RS_MEAN  = PROJECT / 'raw_stat_wavenet_noise_w512_v2_mean.npy'
RS_STD   = PROJECT / 'raw_stat_wavenet_noise_w512_v2_std.npy'

FILES    = ['nt_dorsal_1', 'nt_dorsal_5', 'nt_dorsal_10',
            'nt_dorsal_11', 'nt_dorsal_15', 'nt_dorsal_18']
CHANNELS = [('S1', 0), ('S2', 1)]
RAW_OPTS = dict(bits=16, n_header=48, clock_rate_hz=15_000_000)
DT_MIN   = 0.1

WINDOW_MS = 512
STRIDE_MS = WINDOW_MS // 8
W010      = WINDOW_MS * 10
W050      = WINDOW_MS * 2
POOL010   = WINDOW_MS // 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ── Model (same architecture as training) ─────────────────────────────────────

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


model = WaveNetNoise_W().to(device)
model.load_state_dict(torch.load(MODEL_PT, map_location=device))
model.eval()

raw_stat_mean = np.load(RS_MEAN).astype(np.float32)
raw_stat_std  = np.load(RS_STD ).astype(np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw_channel(stem, ch_label):
    pattern   = str(FCS_DIR / f'{stem}_*_Ch{ch_label}.raw')
    raw_files = sorted(glob.glob(pattern))
    if not raw_files:
        return None, None
    KHZ_FACTOR = 1.0 / (DT_MIN * 1e-3) / 1e3
    segments = []
    for fp in raw_files:
        arr = read_arrivals(fp, **RAW_OPTS)
        seg = bin_arrivals(arr, bin_ms=DT_MIN).astype(np.float64)
        segments.append(seg)
    i010 = np.concatenate(segments) * KHZ_FACTOR
    B050 = round(0.5 / DT_MIN)
    trim = (len(i010) // B050) * B050
    i010 = i010[:trim]
    i050 = i010.reshape(-1, B050).mean(axis=1).astype(np.float64)
    return i010, i050


def run_sliding_window(i010, i050, i100):
    N = len(i100)
    n_hats = []
    with torch.no_grad():
        for t0 in range(0, N - WINDOW_MS + 1, STRIDE_MS):
            t1 = t0 + WINDOW_MS
            w100 = i100[t0:t1].astype(np.float32)
            w050 = i050[t0 * 2 : t1 * 2].astype(np.float32)
            w010 = i010[t0 * 10 : t1 * 10].astype(np.float32)
            if len(w100) < WINDOW_MS or len(w050) < W050 or len(w010) < W010:
                continue
            def znorm(a): return (a - a.mean()) / (a.std() + 1e-8)
            a100 = torch.tensor(znorm(w100)).unsqueeze(0).to(device)
            a050 = torch.tensor(znorm(w050)).unsqueeze(0).to(device)
            a010 = torch.tensor(znorm(w010)).unsqueeze(0).to(device)
            raw   = np.array([w010.mean(), w010.std(),
                               w050.mean(), w050.std(),
                               w100.mean(), w100.std()], dtype=np.float32)
            raw_sc = torch.tensor((raw - raw_stat_mean) / raw_stat_std).unsqueeze(0).to(device)
            n_hats.append(model(a010, a050, a100, raw_sc).item())
    return np.array(n_hats)


# ── Main ──────────────────────────────────────────────────────────────────────

results = {}

for stem in FILES:
    print(f'\n{stem}')
    fcs   = parse_fcs_file(FCS_DIR / f'{stem}.fcs')
    entry = {}

    for ch_label, ch_idx in CHANNELS:
        i100         = fcs.intensity_trace(channel=ch_idx)
        i010, i050   = load_raw_channel(stem, ch_label)

        if i010 is None:
            print(f'  {ch_label}: no RAW')
            entry[ch_label] = None
            continue

        N    = min(len(i100), len(i050) // 2, len(i010) // 10)
        i100 = i100[:N];  i050 = i050[:N * 2];  i010 = i010[:N * 10]

        n_hats = run_sliding_window(i010, i050, i100)
        n_abs  = float(np.percentile(n_hats, 20)) if len(n_hats) > 0 else None

        print(f'  {ch_label}: {len(n_hats)} windows, n_abs = {n_abs:.2f} kHz')
        entry[ch_label] = n_abs

    results[stem] = entry

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved → {OUT_JSON}')
