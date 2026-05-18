"""
model_classes.py — PyTorch model architecture definitions
══════════════════════════════════════════════════════════
All nn.Module classes used by the inference pipeline, extracted from their
training scripts. CUDA streams are made conditional so models run on CPU too.

Classes
-------
  DilatedResBlock      — shared residual block
  WaveNetBackbone      — shared backbone (trace → embedding)
  WaveNetMultiDT_B     — 3-branch model (dt=0.1, 0.5, 1.0 ms)
  WaveNetMultiDT_E     — 4-branch model (dt=0.05, 0.1, 0.5, 1.0 ms)
  WaveNet1D            — single-dt model (dt=0.5 ms, wide_aug)
  WaveNetNoise_W       — windowed noise model (3 branches, Softplus output)

Factory
-------
  build_model(class_name, **kwargs)  → nn.Module  (uninitialised weights)
"""

from __future__ import annotations

import torch
import torch.nn as nn

_CUDA = torch.cuda.is_available()


# ── Shared building blocks ────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


class WaveNetBackbone(nn.Module):
    """Trace → global-average-pooled embedding.

    Parameters
    ----------
    channels : int
    dilations : list[int]
    avgpool   : int — AvgPool1d factor applied before Conv1d (1 = no pooling)
    """
    def __init__(self, channels: int = 256, dilations=None, avgpool: int = 1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        proj = []
        if avgpool > 1:
            proj.append(nn.AvgPool1d(kernel_size=avgpool, stride=avgpool))
        proj += [
            nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(channels), nn.ReLU(),
        ]
        self.input_proj = nn.Sequential(*proj)
        self.blocks     = nn.Sequential(*[DilatedResBlock(channels, d) for d in dilations])
        self.gap        = nn.AdaptiveAvgPool1d(1)

    def forward(self, trace):   # trace: (B, N)
        x = self.input_proj(trace.unsqueeze(1))
        return self.gap(self.blocks(x)).squeeze(2)   # (B, channels)


# ── Diffusion models ──────────────────────────────────────────────────────────

class WaveNetMultiDT_B(nn.Module):
    """3-branch multi-dt model: dt=0.1, 0.5, 1.0 ms.

    Parameters
    ----------
    feat_dim  : int  — feature vector length (910 for standard multidt_B)
    channels  : int  — WaveNet channel width (256)
    dilations : list[int]
    pool010   : int  — AvgPool factor for dt010 branch (16: 40960→2560 bins)
    """
    def __init__(self, feat_dim: int, channels: int = 256,
                 dilations=None, pool010: int = 16):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16, 32, 64, 128]
        self.branch010  = WaveNetBackbone(channels, dilations, avgpool=pool010)
        self.branch050  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        if _CUDA:
            self.s010 = torch.cuda.Stream()
            self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        if _CUDA:
            cur = torch.cuda.current_stream()
            self.s010.wait_stream(cur); self.s050.wait_stream(cur)
            with torch.cuda.stream(self.s010): e010 = self.branch010(t010)
            with torch.cuda.stream(self.s050): e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
            cur.wait_stream(self.s010); cur.wait_stream(self.s050)
        else:
            e010 = self.branch010(t010)
            e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


class WaveNetMultiDT_E(nn.Module):
    """4-branch multi-dt model: dt=0.05, 0.1, 0.5, 1.0 ms.

    Parameters
    ----------
    feat_dim  : int  — feature vector length (1216)
    channels  : int  — WaveNet channel width (256)
    dilations : list[int]
    pool005   : int  — AvgPool factor for dt005 branch (32: 81920→2560 bins)
    pool010   : int  — AvgPool factor for dt010 branch (16: 40960→2560 bins)
    """
    def __init__(self, feat_dim: int, channels: int = 256,
                 dilations=None, pool005: int = 32, pool010: int = 16):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16, 32, 64, 128]
        self.branch005  = WaveNetBackbone(channels, dilations, avgpool=pool005)
        self.branch010  = WaveNetBackbone(channels, dilations, avgpool=pool010)
        self.branch050  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 4 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        if _CUDA:
            self.s005 = torch.cuda.Stream()
            self.s010 = torch.cuda.Stream()

    def forward(self, t005, t010, t050, t100, feats):
        if _CUDA:
            cur = torch.cuda.current_stream()
            self.s005.wait_stream(cur); self.s010.wait_stream(cur)
            with torch.cuda.stream(self.s005): e005 = self.branch005(t005)
            with torch.cuda.stream(self.s010): e010 = self.branch010(t010)
            e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
            cur.wait_stream(self.s005); cur.wait_stream(self.s010)
        else:
            e005 = self.branch005(t005)
            e010 = self.branch010(t010)
            e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
        return self.head(torch.cat([e005, e010, e050, e100, f], dim=1)).squeeze(1)


class WaveNet1D(nn.Module):
    """Single-dt WaveNet (wide_aug) — AdaptiveAvgPool handles any input length.

    Parameters
    ----------
    feat_dim  : int  — feature vector length (303)
    channels  : int  — WaveNet channel width (256)
    dilations : list[int]
    """
    def __init__(self, feat_dim: int, channels: int = 256, dilations=None):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16, 32, 64, 128]
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
        x = self.gap(self.blocks(x)).squeeze(2)
        f = self.feat_branch(feats)
        return self.head(torch.cat([x, f], dim=1)).squeeze(1)


# ── Noise model ───────────────────────────────────────────────────────────────

class WaveNetNoise_W(nn.Module):
    """3-branch WaveNet for windowed noise estimation.

    Returns n_abs in kHz (Softplus ensures non-negative output).
    The model is trained with target = n_abs_hz / N_SCALE where N_SCALE=1e3,
    so raw_output * N_SCALE / 1000 = raw_output gives n_abs in kHz directly.

    Parameters
    ----------
    channels  : int
    dilations : list[int]
    pool010   : int  — window-size-dependent AvgPool (= max(1, window_ms//256))
    """
    def __init__(self, channels: int = 256, dilations=None, pool010: int = 1):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16, 32, 64, 128]
        self.branch010  = WaveNetBackbone(channels, dilations, avgpool=pool010)
        self.branch050  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.branch100  = WaveNetBackbone(channels, dilations, avgpool=1)
        self.feat_branch = nn.Sequential(
            nn.Linear(6, 32),  nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3 + 32, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        if _CUDA:
            self.s010 = torch.cuda.Stream()
            self.s050 = torch.cuda.Stream()

    def forward(self, t010, t050, t100, feats):
        if _CUDA:
            cur = torch.cuda.current_stream()
            self.s010.wait_stream(cur); self.s050.wait_stream(cur)
            with torch.cuda.stream(self.s010): e010 = self.branch010(t010)
            with torch.cuda.stream(self.s050): e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
            cur.wait_stream(self.s010); cur.wait_stream(self.s050)
        else:
            e010 = self.branch010(t010)
            e050 = self.branch050(t050)
            e100 = self.branch100(t100)
            f    = self.feat_branch(feats)
        return self.head(torch.cat([e010, e050, e100, f], dim=1)).squeeze(1)


# ── Factory ───────────────────────────────────────────────────────────────────

_CLASS_REGISTRY = {
    'WaveNetMultiDT_B':  WaveNetMultiDT_B,
    'WaveNetMultiDT_S2': WaveNetMultiDT_B,   # S2/mCherry2 model — identical architecture
    'WaveNetMultiDT_E':  WaveNetMultiDT_E,
    'WaveNet1D':         WaveNet1D,
    'WaveNetNoise_W':    WaveNetNoise_W,
}


def build_model(class_name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by class name with keyword constructor arguments.

    Example
    -------
        model = build_model('WaveNetMultiDT_B', feat_dim=910, channels=256)
        model.load_state_dict(torch.load('model_wavenet_multidt_B.pt',
                                          map_location='cpu'))
    """
    if class_name not in _CLASS_REGISTRY:
        raise ValueError(
            f'Unknown model class: {class_name!r}. '
            f'Available: {list(_CLASS_REGISTRY)}'
        )
    return _CLASS_REGISTRY[class_name](**kwargs)
