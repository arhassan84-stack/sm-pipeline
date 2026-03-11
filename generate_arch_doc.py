"""
Generate Model_Architecture_Guide.docx / .pdf
One section per model (or model family) with text-based architecture diagrams.
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import reportlab.lib.pagesizes as pagesizes
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 PageBreak, KeepTogether, HRFlowable)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Architecture diagrams (ASCII art, monospace)
# ─────────────────────────────────────────────────────────────────────────────

DIAG_GBR = """\
  Feature Vector (39–207 dims)
           │
           ▼
  ┌─────────────────────┐
  │  Decision Tree      │  ×N estimators
  │  (split on feature) │  (GBR: 100–300 trees;
  │  leaf → partial fit │   HistGBR: histogram bins)
  └─────────────────────┘
           │
           ▼
  Σ (weighted sum of trees)
           │
           ▼
  exp( prediction ) → D̂"""

DIAG_SKLEARN_MLP = """\
  Feature Vector (207–283 dims)
           │
           ▼
  ┌──────────────────┐
  │  Linear + ReLU   │  layer 1 (e.g. 512)
  └──────────────────┘
           │
  ┌──────────────────┐
  │  Linear + ReLU   │  layer 2 (e.g. 256)
  └──────────────────┘
           │
         ...
           │
  ┌──────────────────┐
  │  Linear (→ 1)    │  output
  └──────────────────┘
           │
           ▼
  exp( log D̂ ) → D̂"""

DIAG_PT_MLP = """\
  Feature Vector (302–283 dims)
           │
           ▼
  ┌─────────────────────────────┐
  │  Linear → BatchNorm → ReLU  │  2048 / 1024
  │  → Dropout(0.15)            │
  └─────────────────────────────┘
           │   (×4 blocks, dims: 2048→1024→512→256→128)
           ▼
  ┌─────────────────────────────┐
  │  Linear (128 → 1)           │  output
  └─────────────────────────────┘
           │
           ▼
  exp( log D̂ ) → D̂"""

DIAG_RESNET = """\
  Feature Vector (302–283 dims)
           │
  ┌─────────────────────────────┐
  │  Input projection → 512     │
  └──────────┬──────────────────┘
             │
  ┌──────────▼──────────┐
  │  Linear 512→512     │   ┐
  │  BatchNorm → ReLU   │   │ residual
  │  + skip connection  │   │ block ×6
  └──────────┬──────────┘   ┘
             │
  ┌──────────▼──────────┐
  │  Linear 512 → 1     │
  └─────────────────────┘
           │
           ▼
  exp( log D̂ ) → D̂"""

DIAG_CNN_FUSION = """\
  Raw Trace (4096 pts)        Feature Vector (207–302 dims)
       │                               │
       ▼                               ▼
  ┌──────────────────┐    ┌──────────────────────────┐
  │ Conv1d(1→32,k=8) │    │ Linear(F→256) → ReLU     │
  │ → ReLU → Pool    │    │ Linear(256→128) → ReLU   │
  │ Conv1d(32→64,k=4)│    └──────────┬───────────────┘
  │ → ReLU → Pool    │               │
  │ Conv1d(64→128)   │               │
  │ → ReLU → GAP     │               │
  └──────────┬───────┘               │
             │       128-dim          │  128-dim
             └────────────┬──────────┘
                          │
                   cat → 256-dim
                          │
                   ┌──────▼───────┐
                   │ Linear → 64  │
                   │ → ReLU → 1   │
                   └──────────────┘
                          │
                          ▼
                   exp( log D̂ ) → D̂"""

DIAG_MULTISCALE_CNN = """\
  Raw Trace (4096 pts)
       │
       ├─────────────────────┬──────────────────────┐
       ▼                     ▼                      ▼
  Conv1d k=8            Conv1d k=32            Conv1d k=128
  → Pool → flatten      → Pool → flatten       → Pool → flatten
  (short bursts)        (medium events)        (long events)
       │                     │                      │
       └───────────┬──────────┘                     │
                   └──────────────┬─────────────────┘
                                  │
                        cat + Feature branch (207f → 128)
                                  │
                        ┌─────────▼──────────┐
                        │ Linear → 64 → 1    │
                        └────────────────────┘
                                  │
                                  ▼
                          exp( log D̂ ) → D̂"""

DIAG_PATCH_TRANSFORMER = """\
  Raw Trace (4096 pts) → split into 64 patches × 64 pts
       │
       ▼
  Linear patch embedding → d=128 per patch
       │
       ▼  + positional encoding
  ┌──────────────────────────────┐
  │  Multi-head Self-Attention   │  4 heads
  │  (64+1 tokens × 128-dim)    │  4 layers
  │  + Feed-Forward(128→256→128)│
  └──────────────────┬───────────┘
                     │ [CLS] token → 128-dim
                     │
  Feature Vector (207f) ──→ Linear → 128-dim
                     │               │
                     └───────┬───────┘
                             │
                     ┌───────▼──────┐
                     │ Linear → 1   │
                     └──────────────┘
                             │
                             ▼
                     exp( log D̂ ) → D̂"""

DIAG_FTT = """\
  Feature Vector (283 features)
       │
       ▼  feature tokenization
  Each feature xᵢ → embedding wᵢ·xᵢ + bᵢ ∈ ℝᵈ   (d=64 or 128)
       │
  ┌────▼──────────────────────────────────────┐
  │  [CLS] token  |  f₁  |  f₂  | … | f₂₈₃  │  (284 tokens)
  └────┬──────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────┐
  │  Multi-head Self-Attention (8H)          │
  │  + LayerNorm + FFN (d×4)                 │
  └────┬─────────────────────────────────────┘
       │  × 4 or 6 layers
       ▼
  [CLS] representation ∈ ℝᵈ
       │
  ┌────▼──────────────────┐
  │  Linear(d → 1)        │
  └───────────────────────┘
       │
       ▼
  exp( log D̂ ) → D̂"""

DIAG_WAVENET = """\
  Raw Trace (4096 pts)         Feature Vector (302 dims)
       │                               │
       ▼                               ▼
  ┌──────────────────────────┐  ┌────────────────────────────┐
  │ Conv1d(1→128, k=16, s=4) │  │ Linear(302→256) + BN+ReLU  │
  │ BatchNorm → ReLU         │  │ Dropout(0.1)               │
  └──────────────────────────┘  │ Linear(256→128) + BN+ReLU  │
       │                        └──────────────┬─────────────┘
       │  (1024-pt sequence, 128 channels)      │
       ▼                                        │
  ┌──────────────────────────┐                  │
  │ DilatedResBlock(dil=1)   │                  │
  │ DilatedResBlock(dil=2)   │  dilations       │
  │ DilatedResBlock(dil=4)   │  [1,2,4,8,       │
  │ DilatedResBlock(dil=8)   │   16,32,64,128]  │
  │ DilatedResBlock(dil=16)  │                  │
  │ DilatedResBlock(dil=32)  │  each block:     │
  │ DilatedResBlock(dil=64)  │  Conv1d(ch,ch,   │
  │ DilatedResBlock(dil=128) │  k=3,pad=dil)    │
  └──────────────┬───────────┘  + BN + ReLU     │
                 │                + skip         │
       AdaptiveAvgPool1d(1)                      │
                 │  128-dim                      │  128-dim
                 └──────────────┬────────────────┘
                                │
                        cat → 256-dim
                                │
                   ┌────────────▼────────────┐
                   │ Linear(256→256)+BN+ReLU │
                   │ Dropout(0.1)            │
                   │ Linear(256→64) + ReLU   │
                   │ Linear(64→1)            │
                   └─────────────────────────┘
                                │
                                ▼
                        exp( log D̂ ) → D̂

  DilatedResBlock detail:
    x ──→ Conv1d(ch, ch, k=3, dilation=d, padding=d) → BN → ReLU → + x → output"""

DIAG_WAVENET_WIDE = """\
  Same as WaveNet (above) but:
  ─────────────────────────────
  channels = 256  (vs 128 for base)
  → larger trace embedding (256-dim instead of 128)
  → head input: 256+128 = 384-dim

  Raw Trace (4096 pts)         Feature Vector (302 dims)
       │                               │
  Conv1d(1→256, k=16, s=4)    Linear(302→256)→Linear(256→128)
       │                               │
  8×DilatedResBlock(ch=256)            │
       │                               │
  AdaptiveAvgPool1d(1) → 256-dim  128-dim
       └──────────────────┬────────────┘
                          │
                   cat → 384-dim
                          │
              Linear(384→256)+BN+ReLU+Dropout
              Linear(256→64)+ReLU
              Linear(64→1)
                          │
                          ▼
                  exp( log D̂ ) → D̂"""

DIAG_WAVENET_AUG = """\
  Training-time augmentation applied to raw traces before z-score normalization:
  ─────────────────────────────────────────────────────────────────────────────
   1. Amplitude scale: × Uniform[0.85, 1.15]  per trace
   2. Additive noise:  + Normal(0, 0.03)       per bin
   3. Cyclic shift:    roll ±128 bins          per trace (dt=1ms)
                                               roll ±256 bins (dt=0.5ms)

  Augmented trace → z-score normalize → feed to WaveNet wide (ch=256)

""" + DIAG_WAVENET_WIDE

DIAG_WAVENET_FTT = """\
  Raw Trace (4096 pts)         Feature Vector (283 dims)
       │                               │
       ▼                               ▼
  WaveNet trace branch         FT-Transformer feature branch
  (Conv1d → 8 ResBlocks        (feature tokenization → 4-layer
   → GAP → 128-dim)             self-attention → CLS → 64-dim)
       │                               │
       └──────────────┬────────────────┘
                      │
               cat → 192-dim
                      │
           ┌──────────▼───────────┐
           │ Linear(192→128)+ReLU │
           │ Linear(128→64)+ReLU  │
           │ Linear(64→1)         │
           └──────────────────────┘
                      │
                      ▼
              exp( log D̂ ) → D̂"""

DIAG_SMOOTH = """\
  Smoothing pre-processing:
  ─────────────────────────────────────────────────────────
  Smooth trace = moving average of raw trace with window W

  raw trace [4096 pts]  ──→  smooth trace [4096 pts]
                         │
                         └──→  recompute all 302 features
                               on smoothed trace

  Both smoothed trace + smoothed features fed to WaveNet wide (ch=256):

  Smooth Trace (4096-pt)        Smooth Features (302-dim)
       │                               │
  Conv1d(1→256, k=16, s=4)     Linear(302→256)→Linear(256→128)
       │                               │
  8×DilatedResBlock(ch=256)            │
       │                               │
  GAP → 256-dim               128-dim
       └──────────────┬────────────────┘
                      │  cat → 384-dim
              Linear(384→256)+BN+ReLU
              Linear(256→64)+ReLU → Linear(64→1)
                      │
                      ▼
              exp( log D̂ ) → D̂"""

DIAG_FUSED = """\
  Dual-representation input: raw + smoothed (W bins moving average)
  ─────────────────────────────────────────────────────────────────
  Raw trace  [4096 pts] ─┐
                          ├── concatenate → 8192-pt trace
  Smooth [W] [4096 pts] ─┘

  Raw features  [302-dim] ─┐
                            ├── concatenate → 604-dim feature vector
  Smooth features [302-dim]─┘

  AMP (Automatic Mixed Precision) training enabled.

  Fused Trace (8192-pt)          Fused Features (604-dim)
       │                               │
  Conv1d(1→256, k=16, s=4)     Linear(604→256)→Linear(256→128)
       │                               │
  8×DilatedResBlock(ch=256)            │
       │                               │
  AdaptiveAvgPool1d(1) → 256-dim  128-dim
       └──────────────┬────────────────┘
                      │  cat → 384-dim
              Linear(384→256)+BN+ReLU+Dropout
              Linear(256→64)+ReLU → Linear(64→1)
                      │
                      ▼
              exp( log D̂ ) → D̂

  Note: AdaptiveAvgPool1d(1) makes the model length-agnostic —
        8192-pt input handled without architectural changes."""

DIAG_ONLINE_BG = """\
  Online Poisson background augmentation per training batch:
  ──────────────────────────────────────────────────────────
  b_pct ~ Uniform[0%, 20%]   (drawn fresh each batch)
  b_bin  = b_pct × MAX_RATE × DT   (0–10 photons/bin)
  bg     ~ Poisson(b_bin)  per bin

  Raw trace (clean) + bg / DT  → noisy trace → z-score → WaveNet

  Clean features (pre-computed, no background)  → feature branch

  Raw Trace (4096-pt) + Poisson bg    Clean Features (302-dim)
               │                               │
  z-score normalize                    Linear(302→256)→(256→128)
               │                               │
  WaveNet wide (ch=256)                        │
  8×DilatedResBlock → GAP → 256-dim   128-dim
               └──────────────┬────────────────┘
                              │  cat → 384-dim
                      Linear(384→256)+BN+ReLU
                      Linear(256→64)+ReLU → Linear(64→1)
                              │
                              ▼
                      exp( log D̂ ) → D̂

  Validation always at b=0 (clean) for comparable val loss."""

DIAG_DT050 = """\
  dt=0.5ms: tMax=4096ms → 8192 bins per trace (2× longer than dt=1ms)

  Augmentation (online, scaled proportionally):
  ─────────────────────────────────────────────
  Scale:  × Uniform[0.85, 1.15]
  Noise:  + Normal(0, 0.03)
  Shift:  roll ±256 bins  (proportional: 128/4096 × 8192)

  Raw Trace dt050 (8192-pt)      Features dt050 (302-dim)
               │                         │
               │    (make_lag_constants(8192) → all lag arrays)
               │                         │
  Conv1d(1→256, k=16, s=4)       Linear(302→256)→(256→128)
               │  → 2048-pt × 256 ch            │
  8×DilatedResBlock(ch=256)               │
               │                         │
  AdaptiveAvgPool1d(1) → 256-dim   128-dim
               └──────────────┬──────────┘
                              │  cat → 384-dim
                      Linear(384→256)+BN+ReLU
                      Linear(256→64)+ReLU → Linear(64→1)
                              │
                              ▼
                      exp( log D̂ ) → D̂

  AdaptiveAvgPool1d(1) is fully length-agnostic:
    dt=1ms  → 4096-pt input → 1024-pt after stride-4 → pooled to 1
    dt=0.5ms → 8192-pt input → 2048-pt after stride-4 → pooled to 1
  Identical architecture, different input resolution."""

DIAG_ENSEMBLE = """\
  Model 1 predictions        Model 2 predictions   ...   Model N predictions
  pred_logd_1 [n_test]       pred_logd_2 [n_test]        pred_logd_N [n_test]
        │                          │                            │
        └──────────────────────────┴────────────────────────────┘
                                   │
                     Geometric mean in log space:
                     log D̂_ens = (1/N) Σᵢ log D̂ᵢ
                                   │
                                   ▼
                           exp( log D̂_ens ) → D̂

  OR (NNLS meta-learner):
  Non-negative least squares: log D̂_ens = Σᵢ wᵢ · log D̂ᵢ
  weights wᵢ ≥ 0, Σwᵢ = 1, fit by 5-fold cross-validation on training preds."""

DIAG_SPECIALIST = """\
  Training: filter training set to d ≤ threshold (slow) or d ≥ threshold (fast)
  Architecture: same as MLP+BN+Dropout or WaveNet wide

  At test time — standalone (NO routing):
  All test samples → specialist → predictions
  (specialist extrapolates badly to out-of-distribution d values)

  At test time — with router:
  Predicted D̂_router ──→ if D̂ < threshold: use general model
                          if D̂ ≥ threshold: use specialist"""

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

ARCH_MODELS = [
    # ── Phase 1 ──────────────────────────────────────────────────────────
    {
        'name': 'Gradient Boosting Regressor (GBR) Baseline — 39 Features',
        'phase': 'Phase 1', 'status': 'Complete',
        'label': 'GradientBoostingRegressor (sklearn)',
        'trace_input': 'None',
        'feature_input': '39 features: intensity stats + ACF@29 lags',
        'feat_dim': '39',
        'train_samples': '10,000 (50% split)',
        'mape': '54.7%  d<1: —  d≥1: —',
        'hyperparams': 'n_estimators=100, learning_rate=0.1, max_depth=3. Predicts raw d (not log-transformed).',
        'diagram': DIAG_GBR,
        'notes': 'First end-to-end run. High MAPE due to raw d target (not log-transformed) and small feature set.',
    },
    {
        'name': 'Histogram Gradient Boosting Regressor (HistGBR) — 175 Features, 50k Samples, log(D) Target',
        'phase': 'Phase 1', 'status': 'Complete',
        'label': 'HistGradientBoostingRegressor (sklearn)',
        'trace_input': 'None',
        'feature_input': '175 features (+scattering J=8,Q=2)',
        'feat_dim': '175',
        'train_samples': '50,000 (50% split)',
        'mape': '17.5%  d<1: —  d≥1: —',
        'hyperparams': 'Default params. log(d) target. 3.6 s training (vs 3 min for GBR).',
        'diagram': DIAG_GBR,
        'notes': 'First model with log(d) target — large MAPE reduction from 38.5% (raw d) to 17.5%.',
    },
    {
        'name': 'Histogram Gradient Boosting Regressor (HistGBR) Tuned — 207 Features, 50k Samples',
        'phase': 'Phase 1', 'status': 'Complete',
        'label': 'HistGradientBoostingRegressor (tuned, sklearn)',
        'trace_input': 'None',
        'feature_input': '207 features (+PSD at 32 log-spaced bins)',
        'feat_dim': '207',
        'train_samples': '50,000 (50% split)',
        'mape': '16.8%  d<1: —  d≥1: —',
        'hyperparams': (
            'max_iter=800, max_depth=5, learning_rate=0.08,\n'
            'min_samples_leaf=80, max_leaf_nodes=127, l2_regularization=0.0\n'
            '(found by grid search)'),
        'diagram': DIAG_GBR,
        'notes': 'PSD features add modest gain (+0.7% vs 175f). Tuned HistGBR is the sklearn ceiling.',
    },
    {
        'name': 'Multi-Layer Perceptron (MLP) 512-256-128 — 207 Features, 50k Samples (sklearn)',
        'phase': 'Phase 1', 'status': 'Complete',
        'label': 'MLPRegressor (sklearn)',
        'trace_input': 'None',
        'feature_input': '207 features',
        'feat_dim': '207',
        'train_samples': '50,000 (50% split)',
        'mape': '16.4%  d<1: —  d≥1: —',
        'hyperparams': 'hidden_layer_sizes=(512,256,128), early_stopping=True (29 iters).',
        'diagram': DIAG_SKLEARN_MLP,
        'notes': 'Comparable to tuned HistGBR. CPU-only training limits capacity.',
    },
    # ── Phase 2 ──────────────────────────────────────────────────────────
    {
        'name': 'Histogram Gradient Boosting Regressor (HistGBR) — 283 Features, 90/10 Split',
        'phase': 'Phase 2', 'status': 'Complete',
        'label': 'HistGradientBoostingRegressor (sklearn, full feature set)',
        'trace_input': 'None',
        'feature_input': '283 features (212 base + 71 expanded)',
        'feat_dim': '283',
        'train_samples': '85,698',
        'mape': '16.1%  d<1: 13.7%  d≥1: 21.0%',
        'hyperparams': 'Same tuned params as Phase 1. 90/10 split. R²=0.9115.',
        'diagram': DIAG_GBR,
        'notes': 'Feature baseline. 71 expanded features include ACF fits, segment stats, cumulants.',
    },
    {
        'name': 'Multi-Layer Perceptron (MLP) Baseline 512-256-128 — sklearn, 90% Split',
        'phase': 'Phase 2', 'status': 'Complete',
        'label': 'MLPRegressor (sklearn)',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '15.9%  d<1: —  d≥1: —',
        'hyperparams': 'hidden_layer_sizes=(512,256,128), early_stopping=True.',
        'diagram': DIAG_SKLEARN_MLP,
        'notes': 'Better than HistGBR due to larger training set and 302-feature pipeline.',
    },
    {
        'name': 'Multi-Layer Perceptron (MLP) Wider Variants — sklearn, 90% Split',
        'phase': 'Phase 2', 'status': 'Complete',
        'label': 'MLPRegressor — 3 width variants',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '15.8–16.3%  d<1: —  d≥1: —',
        'hyperparams': (
            'Variants: (1024,512,256,128), (2048,1024,512,256), (2048,1024,512,256,128).\n'
            'Marginal differences — CPU training limits benefit of width.'),
        'diagram': DIAG_SKLEARN_MLP,
        'notes': 'Plateau reached with sklearn MLPs. GPU PyTorch models improve substantially.',
    },
    # ── Phase 3 ──────────────────────────────────────────────────────────
    {
        'name': 'PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization (BN) and Dropout',
        'phase': 'Phase 3', 'status': 'Complete',
        'label': 'PyTorch MLP with BatchNorm and Dropout',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '15.2%  d<1: —  d≥1: —',
        'hyperparams': 'dims: 2048→1024→512→256→128→1, Dropout(0.15), AdamW LR=1e-3, ReduceLROnPlateau.',
        'diagram': DIAG_PT_MLP,
        'notes': 'First GPU model. 3.2M params. 99 s training. 27-42× faster than sklearn equivalent.',
    },
    {
        'name': 'PyTorch Residual Network (ResNet) with Multi-Layer Perceptron (MLP) Head',
        'phase': 'Phase 3', 'status': 'Complete',
        'label': 'PyTorch ResNet-style MLP, 6 residual blocks',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '15.6%  d<1: —  d≥1: —',
        'hyperparams': '3.3M params. Input proj → 512-dim. 6 residual blocks (Linear+BN+ReLU+skip). 93 s.',
        'diagram': DIAG_RESNET,
        'notes': 'Residual connections help gradient flow but skip connections add no gain over MLP+BN.',
    },
    {
        'name': 'Convolutional Neural Network Fusion (CNN-Fusion) v1 — Raw Trace + Feature Branch',
        'phase': 'Phase 3', 'status': 'Complete',
        'label': '1D CNN on raw trace + FC feature branch',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '207 features',
        'feat_dim': '207',
        'train_samples': '47,600 (50% split)',
        'mape': '14.9%  d<1: —  d≥1: —',
        'hyperparams': '0.57M params. 3 Conv1d layers (32→64→128 channels). GAP + 207f branch → head. 61 s.',
        'diagram': DIAG_CNN_FUSION,
        'notes': 'First model using raw traces. GAP makes it trace-length-agnostic. Lower MAPE than MLP alone.',
    },
    {
        'name': 'Patch Transformer — 64 Patches × 64 Points per Patch',
        'phase': 'Phase 3', 'status': 'Complete',
        'label': 'Patch Transformer (64×64-pt patches) + feature branch',
        'trace_input': 'Raw 4096-pt trace (as 64 patches of 64 pts)',
        'feature_input': '207 features',
        'feat_dim': '207',
        'train_samples': '47,600',
        'mape': '15.9%  d<1: —  d≥1: —',
        'hyperparams': '0.66M params. d=128, 4 heads, 4 layers. Positional encoding. 142 s.',
        'diagram': DIAG_PATCH_TRANSFORMER,
        'notes': 'Patches too short (64 pts ≈ 64 ms) to capture slow diffusion ACF. Worse than CNN-Fusion.',
    },
    {
        'name': 'Multi-Scale Convolutional Neural Network (CNN) v1 — Three Parallel Branches',
        'phase': 'Phase 3', 'status': 'Complete',
        'label': '3-branch CNN (k=8,32,128) + feature branch',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '207 features',
        'feat_dim': '207',
        'train_samples': '47,600',
        'mape': '14.7%  d<1: —  d≥1: —',
        'hyperparams': '0.46M params. Three parallel Conv1d branches: kernels 8, 32, 128. Concat → head. 175 s.',
        'diagram': DIAG_MULTISCALE_CNN,
        'notes': 'Multi-scale kernels capture burst structure at different timescales. Best Phase 3 model.',
    },
    # ── Phase 4 ──────────────────────────────────────────────────────────
    {
        'name': 'Feature Tokenization Transformer (FT-Transformer) — Embedding=64, 8 Heads, 4 Layers',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'Feature Tokenization Transformer — base variant',
        'trace_input': 'None',
        'feature_input': '283 features',
        'feat_dim': '283',
        'train_samples': '85,698',
        'mape': '14.7%  d<1: 12.1%  d≥1: 20.1%',
        'hyperparams': '236k params. E=64, 8 heads, 4 layers, FFN dim=256, Dropout=0.1. LR=5e-4. 48 min.',
        'diagram': DIAG_FTT,
        'notes': 'Best individual R²=0.9196 (feature-only model). Attention learns feature interactions. FTTs zeroed by NNLS when CNNs present.',
    },
    {
        'name': 'Feature Tokenization Transformer (FT-Transformer) Large — Embedding=128, 8 Heads, 6 Layers',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'Feature Tokenization Transformer — large variant',
        'trace_input': 'None',
        'feature_input': '283 features',
        'feat_dim': '283',
        'train_samples': '85,698',
        'mape': '15.0%  d<1: 12.3%  d≥1: 20.5%',
        'hyperparams': '1.26M params. E=128, 8 heads, 6 layers, FFN dim=512, Dropout=0.15. 100 min.',
        'diagram': DIAG_FTT,
        'notes': 'Larger FTT is WORSE standalone — dataset too small for 1.26M params. Valuable in ensemble for diversity.',
    },
    {
        'name': 'Feature Tokenization Transformer (FT-Transformer) v2 — Embedding=64, 4 Heads, 6 Layers',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'Feature Tokenization Transformer — v2 variant',
        'trace_input': 'None',
        'feature_input': '283 features',
        'feat_dim': '283',
        'train_samples': '85,698',
        'mape': '14.8%  d<1: 12.2%  d≥1: 20.2%',
        'hyperparams': '336k params. E=64, 4 heads, 6 layers, Dropout=0.20. LR=1e-3. 42 min.',
        'diagram': DIAG_FTT,
        'notes': 'Different head/layer tradeoff vs base FTT. Ensemble diversity more important than standalone MAPE.',
    },
    {
        'name': 'Multi-Layer Perceptron (MLP) with Cosine Annealing Warm Restarts Scheduler — 302 Features',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'PyTorch MLP+BN+Dropout + CosineAnnealingWarmRestarts',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '14.6%  d<1: —  d≥1: —',
        'hyperparams': '3.2M params. CosineAnnealingWarmRestarts(T_0=30). AdamW. 98 s.',
        'diagram': DIAG_PT_MLP,
        'notes': 'CosineAnnealingWarmRestarts became the standard scheduler for all subsequent models.',
    },
    {
        'name': 'Residual Network (ResNet) with Cosine Annealing Warm Restarts Scheduler — 302 Features',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'PyTorch ResNet-MLP + CosineAnnealingWarmRestarts',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '14.7%  d<1: —  d≥1: —',
        'hyperparams': '3.3M params. CosineAnnealingWarmRestarts(T_0=30). AdamW. 112 s.',
        'diagram': DIAG_RESNET,
        'notes': 'Paired with MLP+Cosine as ensemble partner throughout later phases.',
    },
    {
        'name': 'Convolutional Neural Network (CNN) on 90k Traces — v2, Full Training Set',
        'phase': 'Phase 4', 'status': 'Complete',
        'label': 'CNN-Fusion on 90k traces + 302-dim features',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features (incl. 5 transit-time)',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '14.2%  d<1: —  d≥1: —',
        'hyperparams': '0.57M params. Transit-time features added: 14.6%→14.0%. CosineAnnealing. 161 s.',
        'diagram': DIAG_CNN_FUSION,
        'notes': 'Transit-time features (fraction above 10–80% of peak) reduce MAPE by 0.6%. Core CNN anchor for all ensembles.',
    },
    # ── Phase 5 ──────────────────────────────────────────────────────────
    {
        'name': '3-Model Ensemble — Geometric Mean in Log Space',
        'phase': 'Phase 5', 'status': 'Complete',
        'label': 'Equal-weight geometric mean ensemble',
        'trace_input': 'Ensemble of above',
        'feature_input': 'Ensemble of above',
        'feat_dim': 'N/A',
        'train_samples': '85,698',
        'mape': '13.9%  d<1: 11.2%  d≥1: 19.9%',
        'hyperparams': 'Geometric mean: log D̂_ens = (1/3) Σ log D̂ᵢ. MLP+Cosine + ResNet+Cosine + CNN 90k.',
        'diagram': DIAG_ENSEMBLE,
        'notes': 'Geometric mean in log space = arithmetic mean of log predictions. No extra training needed.',
    },
    {
        'name': 'Non-Negative Least Squares (NNLS) Meta-Learner — 13 Models, 5-Fold Cross-Validation (CV)',
        'phase': 'Phase 5', 'status': 'Complete',
        'label': 'Non-negative least squares meta-learner',
        'trace_input': 'Predictions from all 13 base models',
        'feature_input': 'Predictions from all 13 base models',
        'feat_dim': 'N/A',
        'train_samples': '85,698 (5-fold CV)',
        'mape': '13.07%  d<1: 10.4%  d≥1: 18.5%',
        'hyperparams': (
            'NNLS with Ridge regularization (alpha=0.10). 5-fold stratified CV on training predictions.\n'
            'Top weights: cnn_s123=0.243, wavenet_ftt=0.204, wavenet_s2=0.190. FTTs zeroed.'),
        'diagram': DIAG_ENSEMBLE,
        'notes': 'NNLS consistently zeroes FTT weights when CNNs/WaveNets present. Equal-weight geometric mean often competitive.',
    },
    {
        'name': 'Balanced Multi-Layer Perceptron (MLP) + Balanced Residual Network (ResNet) — Class-Balanced Training',
        'phase': 'Phase 5', 'status': 'Complete',
        'label': 'Class-balanced training (d<1 downsampled)',
        'trace_input': 'None',
        'feature_input': '283 features',
        'feat_dim': '283',
        'train_samples': '57,190 (balanced: d<1 downsampled to match d≥1 count)',
        'mape': 'MLP: 15.4%  ResNet: 15.2%  d<1: ~12.5%  d≥1: ~20.5%',
        'hyperparams': 'Same architecture as MLP+Cosine / ResNet+Cosine. 27% less training data.',
        'diagram': DIAG_PT_MLP,
        'notes': 'Data balancing does NOT help d≥1 — the gap is physics-limited (faster diffusion → shorter ACF decay → harder to predict), not data-imbalance-limited.',
    },
    # ── Phase 6 ──────────────────────────────────────────────────────────
    {
        'name': 'Specialist Models — Slow Diffuser (d≤1.5) and Fast Diffuser (d≥0.5)',
        'phase': 'Phase 6', 'status': 'Abandoned',
        'label': 'Specialist MLP models + routing',
        'trace_input': 'None',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': 'Slow: 62,113 (d≤1.5 filter)  Fast: 37,211 (d≥0.5 filter)',
        'mape': (
            'Slow standalone: 23.1% (d<1=11.9%, d≥1=46.2%) — fails on fast diffusers\n'
            'Fast standalone: 125.7% (d<1=176.6%) — catastrophically fails on slow\n'
            'Router (soft gating): 14.9%'),
        'hyperparams': (
            'Same MLP+BN+Dropout arch. Router: sigmoid gating between specialists.\n'
            'Routing does not recover OOD performance — specialist approach abandoned.'),
        'diagram': DIAG_SPECIALIST,
        'notes': 'Key lesson: specialists trained on filtered data fail catastrophically on out-of-distribution samples. Routing cannot fix fundamentally poor extrapolation.',
    },
    # ── Phase 7 ──────────────────────────────────────────────────────────
    {
        'name': 'WaveNet1D — Dilated Residual Convolutional Network, 128 Channels, 8 Blocks',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet1D — base variant, channels=128',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '13.4%  d<1: 10.8%  d≥1: 18.8%',
        'hyperparams': (
            'ch=128, dilations=[1,2,4,8,16,32,64,128], RF=2040 orig lags.\n'
            'AdamW LR=5e-4, CosineAnnealingWarmRestarts T0=30, patience=25. 1089 s.'),
        'diagram': DIAG_WAVENET,
        'notes': 'BREAKTHROUGH — first model to beat 14% MAPE. Dilated receptive field covers full ACF decay. Best standalone at time of training.',
    },
    {
        'name': 'WaveNet1D Seed Variants (Seeds s2–s5) — Ensemble Diversity via Re-initialization',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet1D reseeded for ensemble diversity',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '13.2–13.6%  d<1: 10.6–10.9%  d≥1: 18.7–19.2%',
        'hyperparams': 'Same as WaveNet base. torch.manual_seed(2/3/4/5). Different random init → different error structure.',
        'diagram': DIAG_WAVENET,
        'notes': 'Seed diversity is highly valuable for ensembling. s2 achieves 13.3% — best standalone model among base WaveNets.',
    },
    {
        'name': 'WaveNet1D Alternative — Full Receptive Field (RF=8184 Lags), 10 Dilated Blocks',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet1D — 10 blocks, dilations=[1..512], full-trace RF',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '13.5%  d<1: 10.9%  d≥1: 18.8%',
        'hyperparams': 'dilations=[1,2,4,8,16,32,64,128,256,512], RF=8184 lags = full 4096-pt trace. 1454 s.',
        'diagram': DIAG_WAVENET,
        'notes': 'Full receptive field does not substantially improve over RF=2040. Ensemble value via different timescale weighting.',
    },
    {
        'name': 'WaveNet1D with Initial Stride-2 Downsampling — Receptive Field (RF) = 4092 Lags',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet1D — initial stride=2, 10 blocks',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '12.8%  d<1: 10.1%  d≥1: 18.2%',
        'hyperparams': 'Initial Conv1d stride=2 (2048-pt after first layer). 10 blocks, RF=4092. Best Phase 7 single model.',
        'diagram': DIAG_WAVENET,
        'notes': 'Stride-2 downsampling before dilated blocks provides a coarser temporal view. Best single-model result at end of Phase 7.',
    },
    {
        'name': 'WaveNet1D Wide — 256 Channels, 8 Dilated Residual Blocks',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet1D — wider channels, 8 blocks',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '13.6%  d<1: 10.7%  d≥1: 19.5%',
        'hyperparams': 'ch=256, dilations=[1..128], RF=2040. Head input: 384-dim. Larger model than ch=128.',
        'diagram': DIAG_WAVENET_WIDE,
        'notes': 'More channels alone does not substantially improve over ch=128 without augmentation.',
    },
    {
        'name': 'WaveNet1D + Feature Tokenization Transformer (FTT) Hybrid — Dual Trace/Feature Branches',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'WaveNet trace branch + FT-Transformer feature branch',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '14.5%  d<1: 12.2%  d≥1: 19.2%',
        'hyperparams': '668k params. Single LR=5e-4 (suboptimal for FTT branch). 3526 s (~1h). R²=0.9034 (standalone).',
        'diagram': DIAG_WAVENET_FTT,
        'notes': 'Weak standalone due to single shared LR. Despite 14.5% MAPE, NNLS assigns it the HIGHEST weight (0.200) of 14 models — unique diversity for d≥1.',
    },
    {
        'name': 'Convolutional Neural Network (CNN) with Trace Augmentation — Amplitude Scale + Noise + Cyclic Shift',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'CNN-Fusion + pre-computed augmented trace cache',
        'trace_input': 'Augmented 4096-pt trace (scale+noise+shift)',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '16.5%  d<1: 13.2%  d≥1: 23.3%',
        'hyperparams': 'Augmentation: amplitude scale U[0.85,1.15] + noise σ=0.03 + cyclic shift ±128 bins.',
        'diagram': DIAG_CNN_FUSION,
        'notes': 'Augmentation hurts CNN — ACF structure disrupted by shifts. WaveNet handles aug much better due to dilated receptive field.',
    },
    {
        'name': 'Convolutional Neural Network (CNN) with Sample-Weighted Mean Squared Error (MSE) Loss',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'CNN-Fusion with sample-weighted MSE loss',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '14.8%  d<1: 12.0%  d≥1: 20.7%',
        'hyperparams': 'Weight w = max(1, sqrt(d)) per sample in MSE loss. No improvement vs unweighted.',
        'diagram': DIAG_CNN_FUSION,
        'notes': 'd≥1 regime is physics-limited, not training-distribution-limited. Reweighting does not close the gap.',
    },
    {
        'name': 'Convolutional Neural Network (CNN) on Peak-Cropped Window (W=512 Points)',
        'phase': 'Phase 7', 'status': 'Complete',
        'label': 'CNN on 512-pt peak-centered window',
        'trace_input': '512-pt peak crop (centered on burst maximum)',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '85,698',
        'mape': '15.7%  d<1: 13.0%  d≥1: 21.3%',
        'hyperparams': 'Crop window: 512 pts around trace maximum. Discards tail (long ACF) information.',
        'diagram': DIAG_CNN_FUSION,
        'notes': 'Tails of trace contain ACF decay info critical for d prediction. Short window loses this signal.',
    },
    # ── Phase 8 ──────────────────────────────────────────────────────────
    {
        'name': 'WaveNet1D Wide + Trace Augmentation — 390k Simulated Traces  ★ PROJECT BEST ★',
        'phase': 'Phase 8', 'status': 'Complete',
        'label': 'WaveNet1D ch=256 + online trace augmentation (390k simulated traces)',
        'trace_input': 'Raw 4096-pt trace (augmented during training)',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '390,000 simulated (dt=1ms, tMax=4096ms)',
        'mape': '12.34%  d<1: 9.79%  d≥1: 17.60%  ← PROJECT BEST',
        'hyperparams': (
            'ch=256, 8 dilated blocks, dilations=[1..128].\n'
            'Aug: scale U[0.85,1.15] + noise N(0,0.03) + shift ±128 bins.\n'
            'AdamW LR=5e-4, WD=1e-4, CosineAnnealingWarmRestarts T0=30, patience=25.'),
        'diagram': DIAG_WAVENET_AUG,
        'notes': 'Key insight: 390k simulated traces + augmentation significantly outperforms 85k real traces. Augmentation improves d≥1 by ~2pp vs non-aug WaveNet wide.',
    },
    {
        'name': 'WaveNet1D Wide Specialist — Trained on Fast Diffusers Only (d≥0.8)',
        'phase': 'Phase 8', 'status': 'Complete',
        'label': 'WaveNet wide trained only on fast diffusers (d≥0.8)',
        'trace_input': 'Raw 4096-pt trace',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '~230,000 (d≥0.8 filter from 390k set)',
        'mape': '~17.5% (full test)  d<1: —  d≥1: 17.5%',
        'hyperparams': 'Same WaveNet wide arch. Filter: d≥0.8 provides boundary overlap.',
        'diagram': DIAG_WAVENET_WIDE,
        'notes': 'Used as specialist in router ensemble. Router: if predicted D ≥ threshold, use specialist. No significant improvement over single wavenet_wide_aug model.',
    },
    {
        'name': 'WaveNet1D Wide with Augmentation v2 — Fine-Tuned from Checkpoint (Learning Rate = 5e-5)',
        'phase': 'Phase 8', 'status': 'Complete',
        'label': 'Fine-tuned from wavenet_wide_aug checkpoint',
        'trace_input': 'Raw 4096-pt trace (augmented)',
        'feature_input': '302 features',
        'feat_dim': '302',
        'train_samples': '390,000 simulated',
        'mape': '13.1%  d<1: —  d≥1: —',
        'hyperparams': 'Loads model_wavenet_wide_aug.pt. LR=5e-5 (10× lower). CosineAnnealing(T_max=100). Patience=20.',
        'diagram': DIAG_WAVENET_AUG,
        'notes': 'Fine-tuning from checkpoint results in slight regression (13.1% vs 12.34%). Model already well-converged.',
    },
    # ── Phase 9: smooth ───────────────────────────────────────────────────
    {
        'name': 'WaveNet1D on Pre-Smoothed Traces — Moving Average Windows W=3, 5, 10, 20, 50',
        'phase': 'Phase 9', 'status': 'Complete',
        'label': 'WaveNet wide on moving-average smoothed trace + smoothed features',
        'trace_input': 'Smoothed 4096-pt trace (moving avg W bins)',
        'feature_input': '302 features recomputed on smoothed trace',
        'feat_dim': '302',
        'train_samples': '390,000 simulated',
        'mape': (
            'W=3:  13.0%  d<1=10.8%  d≥1=17.7%\n'
            'W=5:  13.5%  d<1=11.1%  d≥1=18.6%\n'
            'W=10: 13.8%  d<1=11.6%  d≥1=18.3%\n'
            'W=20: 14.3%  d<1=12.3%  d≥1=18.6%\n'
            'W=50: 14.6%  d<1=12.5%  d≥1=18.8%'),
        'hyperparams': 'Same WaveNet wide arch. Smoothing applied to both trace and feature computation. No augmentation.',
        'diagram': DIAG_SMOOTH,
        'notes': 'All 5 variants worse than wavenet_wide_aug (12.34%). Monotonic degradation with increasing W. Raw traces + augmentation outperform pre-smoothed traces.',
    },
    # ── Phase 10: fused ──────────────────────────────────────────────────
    {
        'name': 'WaveNet1D Fused Raw + Smoothed Traces — Concatenated Input, Windows W=3, 5, 10, 20, 50',
        'phase': 'Phase 10', 'status': 'Complete',
        'label': 'WaveNet wide: raw+smooth concatenated trace and features',
        'trace_input': 'Raw (4096-pt) + Smooth (4096-pt) → 8192-pt concat',
        'feature_input': 'Raw 302 + Smooth 302 → 604-dim',
        'feat_dim': '604',
        'train_samples': '390,000 simulated',
        'mape': (
            'W=3:  14.0%  d<1=11.1%  d≥1=20.0%\n'
            'W=5:  12.6%  d<1=10.1%  d≥1=17.8%  ← best fused\n'
            'W=10: 13.5%  d<1=11.2%  d≥1=18.2%\n'
            'W=20: 13.8%  d<1=11.9%  d≥1=17.7%\n'
            'W=50: 13.0%  d<1=10.6%  d≥1=17.9%'),
        'hyperparams': 'AMP (mixed precision) training. AdaptiveAvgPool1d(1) — length-agnostic for 8192-pt input.',
        'diagram': DIAG_FUSED,
        'notes': 'Fused W=5 is closest competitor to wavenet_wide_aug (12.6% vs 12.34%). AdaptiveAvgPool1d allows the same architecture to handle 4096 or 8192-pt inputs.',
    },
    # ── Phase 11: online background ───────────────────────────────────────
    {
        'name': 'WaveNet1D Wide with Online Poisson Background Augmentation — Background Rate b=0–20%',
        'phase': 'Phase 11', 'status': 'Complete',
        'label': 'WaveNet wide + online Poisson background augmentation',
        'trace_input': 'Raw 4096-pt trace + Poisson background (during training)',
        'feature_input': '302 clean features (pre-computed, no background)',
        'feat_dim': '302',
        'train_samples': '390,000 simulated',
        'mape': (
            'b=0%  (clean):  19.5%  d<1=18.3%  d≥1=21.9%\n'
            'b=5%  (2.5 ph/bin): 19.4%  d<1=17.8%  d≥1=22.6%\n'
            'b=10% (5.0 ph/bin): 20.0%  d<1=18.3%  d≥1=23.6%\n'
            'b=20% (10.0 ph/bin): 21.5%  d<1=19.6%  d≥1=25.6%'),
        'hyperparams': (
            'MAX_RATE=50,000/s, DT=1e-3 s. b_pct~Uniform[0,20%] per batch.\n'
            'b_bin = b_pct × MAX_RATE × DT = 0–10 photons/bin.\n'
            'bg ~ Poisson(b_bin) per bin. Added before z-score normalization.'),
        'diagram': DIAG_ONLINE_BG,
        'notes': (
            'b=0–20% range (0–10 ph/bin) too aggressive. Even at b=0 test: 19.5% vs 12.34% baseline.\n'
            'Root cause: heavy background corrupts trace structure; feature branch uses clean features creating mismatch.\n'
            'Needs revisiting with narrower range (e.g. b=0–5%).'),
    },
    # ── Phase 12: dt050 ──────────────────────────────────────────────────
    {
        'name': 'WaveNet1D Wide + Augmentation on dt=0.5ms Simulation Data — 8192 Bins per Trace',
        'phase': 'Phase 12', 'status': 'Running',
        'label': 'WaveNet wide + aug on dt=0.5ms simulation data (8192 bins/trace)',
        'trace_input': 'Raw 8192-pt trace (dt=0.5ms, tMax=4096ms)',
        'feature_input': '302 features (make_lag_constants(8192))',
        'feat_dim': '302',
        'train_samples': '~194k simulated (dt050, d≤10)',
        'mape': 'Pending (job running)',
        'hyperparams': (
            'Identical WaveNet wide arch (ch=256, 8 blocks, dilations=[1..128]).\n'
            'AUG_SHIFT=256 bins (proportional: 128/4096 × 8192).\n'
            'AdaptiveAvgPool1d(1) — same arch, 8192-pt input handled automatically.'),
        'diagram': DIAG_DT050,
        'notes': (
            'dt=0.5ms provides 2× temporal resolution. tMax=4096ms fixed → 8192 bins.\n'
            'Feature pipeline generalised via make_lag_constants(N) — all lag arrays scaled to N.\n'
            'SLURM chain: cache_build (job 55614563) → training (job 55614564, afterok).'),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DOCX builder
# ─────────────────────────────────────────────────────────────────────────────

STATUS_HEX = {
    'Complete':  (0x2E, 0x7D, 0x32),
    'Running':   (0xE6, 0x51, 0x00),
    'Abandoned': (0xB7, 0x1C, 0x1C),
}


def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def _add_kv(doc, key, val, mono=False):
    p = doc.add_paragraph()
    r1 = p.add_run(f'{key}: ')
    r1.bold = True
    r1.font.size = Pt(9)
    r2 = p.add_run(str(val))
    r2.font.size = Pt(9)
    if mono:
        r2.font.name = 'Courier New'
    p.paragraph_format.space_after = Pt(2)


def build_arch_docx(out_path):
    doc = Document()
    for sec in doc.sections:
        sec.left_margin   = Inches(1.0)
        sec.right_margin  = Inches(1.0)
        sec.top_margin    = Inches(0.8)
        sec.bottom_margin = Inches(0.8)

    # Title page
    title = doc.add_heading('Model Architecture Guide', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
        run.font.size = Pt(22)

    doc.add_paragraph(
        'Diffusion Coefficient Prediction — Architecture Diagrams, '
        'Inputs, and Hyperparameters for All Models'
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph('')
    intro = doc.add_paragraph(
        'This document contains one section per model (or closely related model family) '
        'covering: architecture flowchart, inputs, hyperparameters, and performance. '
        'Diagrams use monospace ASCII art. All models predict log(D); exp() is applied '
        'to obtain D̂ in final output.'
    )
    intro.runs[0].font.size = Pt(10)
    doc.add_page_break()

    current_phase = None
    for mdl in ARCH_MODELS:
        # Phase divider
        if mdl['phase'] != current_phase:
            current_phase = mdl['phase']
            h = doc.add_heading(current_phase, level=1)
            for run in h.runs:
                run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

        rgb = STATUS_HEX.get(mdl['status'], (0, 0, 0))

        # Model heading
        h2 = doc.add_heading(mdl['name'], level=2)
        for run in h2.runs:
            run.font.color.rgb = RGBColor(*rgb)

        # Status badge
        sp = doc.add_paragraph()
        badge = sp.add_run(f"Status: {mdl['status'].upper()}   |   {mdl['label']}")
        badge.bold = True
        badge.font.color.rgb = RGBColor(*rgb)
        badge.font.size = Pt(9)

        # Info table
        tbl = doc.add_table(rows=5, cols=2)
        tbl.style = 'Table Grid'
        rows_data = [
            ('Trace input',   mdl['trace_input']),
            ('Feature input', mdl['feature_input']),
            ('Feature dim',   mdl['feat_dim']),
            ('Train samples', mdl['train_samples']),
            ('MAPE / metrics', mdl['mape']),
        ]
        for r_idx, (k, v) in enumerate(rows_data):
            row = tbl.rows[r_idx]
            row.cells[0].text = k
            row.cells[1].text = v
            for par in row.cells[0].paragraphs:
                for run in par.runs:
                    run.bold = True
                    run.font.size = Pt(8)
            for par in row.cells[1].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(8)
            set_cell_bg(row.cells[0], 'E8EAF6')
        # Col widths
        for cell in tbl.columns[0].cells:
            cell.width = Inches(1.4)
        for cell in tbl.columns[1].cells:
            cell.width = Inches(4.6)

        doc.add_paragraph('')

        # Hyperparameters
        hp = doc.add_paragraph()
        hp_r = hp.add_run('Hyperparameters: ')
        hp_r.bold = True
        hp_r.font.size = Pt(9)
        hp_val = hp.add_run(mdl['hyperparams'])
        hp_val.font.size = Pt(9)
        doc.add_paragraph('')

        # Architecture diagram
        arch_hdr = doc.add_paragraph()
        ah = arch_hdr.add_run('Architecture Diagram:')
        ah.bold = True
        ah.font.size = Pt(9)

        diag_para = doc.add_paragraph(mdl['diagram'])
        for run in diag_para.runs:
            run.font.name = 'Courier New'
            run.font.size = Pt(7)

        doc.add_paragraph('')

        # Notes
        notes_p = doc.add_paragraph()
        nr = notes_p.add_run('Notes: ')
        nr.bold = True
        nr.font.size = Pt(9)
        nv = notes_p.add_run(mdl['notes'])
        nv.font.size = Pt(9)

        doc.add_page_break()

    doc.save(out_path)
    print(f'Saved arch docx → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# PDF builder
# ─────────────────────────────────────────────────────────────────────────────

def build_arch_pdf(out_path):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesizes.letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.8*inch, bottomMargin=0.8*inch,
    )
    styles = getSampleStyleSheet()

    title_s   = ParagraphStyle('T',  parent=styles['Title'],
                                fontSize=20, textColor=colors.HexColor('#1A237E'),
                                spaceAfter=6, alignment=TA_CENTER)
    sub_s     = ParagraphStyle('S',  parent=styles['Normal'],
                                fontSize=11, spaceAfter=8, alignment=TA_CENTER)
    intro_s   = ParagraphStyle('I',  parent=styles['Normal'],
                                fontSize=9, leading=13, spaceAfter=12)
    phase_s   = ParagraphStyle('PH', parent=styles['Heading1'],
                                fontSize=14, textColor=colors.HexColor('#1A237E'),
                                spaceBefore=6, spaceAfter=4)
    h2_s      = ParagraphStyle('H2', parent=styles['Heading2'],
                                fontSize=12, spaceBefore=10, spaceAfter=2)
    badge_s   = ParagraphStyle('B',  parent=styles['Normal'],
                                fontSize=9, spaceAfter=6)
    key_s     = ParagraphStyle('K',  parent=styles['Normal'],
                                fontSize=8, leading=11, spaceAfter=2)
    diag_s    = ParagraphStyle('D',  parent=styles['Code'],
                                fontName='Courier', fontSize=6.5, leading=9,
                                spaceAfter=8, leftIndent=12)
    note_s    = ParagraphStyle('N',  parent=styles['Normal'],
                                fontSize=8, leading=11, spaceAfter=4,
                                textColor=colors.HexColor('#444444'))
    hr_style  = {'width': '100%', 'thickness': 0.5, 'color': colors.HexColor('#BBBBBB'),
                 'spaceAfter': 4}

    STATUS_COLOR_PDF = {
        'Complete':  colors.HexColor('#2E7D32'),
        'Running':   colors.HexColor('#E65100'),
        'Abandoned': colors.HexColor('#B71C1C'),
    }

    story = []

    story.append(Paragraph('Model Architecture Guide', title_s))
    story.append(Paragraph(
        'Diffusion Coefficient Prediction — Architecture Diagrams, Inputs, and Hyperparameters',
        sub_s))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'This document contains one section per model (or closely related family) covering: '
        'architecture flowchart, inputs, hyperparameters, and performance metrics. '
        'All models predict log(D); exp() is applied to obtain D̂.',
        intro_s))
    story.append(PageBreak())

    current_phase = None
    for mdl in ARCH_MODELS:
        if mdl['phase'] != current_phase:
            current_phase = mdl['phase']
            story.append(Paragraph(current_phase, phase_s))
            story.append(HRFlowable(**hr_style))

        sc = STATUS_COLOR_PDF.get(mdl['status'], colors.black)
        h2_colored = ParagraphStyle(f'h2_{mdl["name"][:10]}', parent=h2_s, textColor=sc)
        badge_colored = ParagraphStyle(f'b_{mdl["name"][:10]}', parent=badge_s, textColor=sc)

        blocks = []
        blocks.append(Paragraph(mdl['name'], h2_colored))
        blocks.append(Paragraph(
            f"<b>Status: {mdl['status'].upper()}</b>   |   {mdl['label']}",
            badge_colored))

        # Key-value table
        from reportlab.platypus import Table, TableStyle
        kv_data = [
            ['Trace input',    mdl['trace_input']],
            ['Feature input',  mdl['feature_input']],
            ['Feature dim',    mdl['feat_dim']],
            ['Train samples',  mdl['train_samples']],
            ['MAPE / metrics', mdl['mape']],
        ]
        kv_table = Table(
            [[Paragraph(f'<b>{k}</b>', key_s), Paragraph(v, key_s)] for k, v in kv_data],
            colWidths=[1.3*inch, 4.7*inch],
        )
        kv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8EAF6')),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        blocks.append(kv_table)
        blocks.append(Spacer(1, 4))

        # Hyperparameters
        blocks.append(Paragraph(
            f'<b>Hyperparameters:</b> {mdl["hyperparams"]}', key_s))
        blocks.append(Spacer(1, 4))

        # Diagram
        blocks.append(Paragraph('<b>Architecture Diagram:</b>', key_s))
        blocks.append(Paragraph(mdl['diagram'].replace('\n', '<br/>'), diag_s))

        # Notes
        blocks.append(Paragraph(f'<b>Notes:</b> {mdl["notes"]}', note_s))
        blocks.append(HRFlowable(**hr_style))
        blocks.append(PageBreak())

        story.append(KeepTogether(blocks[:6]))  # keep header + table together
        for b in blocks[6:]:
            story.append(b)

    doc.build(story)
    print(f'Saved arch pdf  → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    build_arch_docx(os.path.join(BASE, 'Model_Architecture_Guide.docx'))
    build_arch_pdf(os.path.join(BASE,  'Model_Architecture_Guide.pdf'))
    print('Done.')
