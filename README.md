# Single-Molecule FCS Inference Pipeline

End-to-end pipeline for detecting single-molecule fluorescence correlation spectroscopy (FCS) transit events and estimating diffusion coefficients in live zebrafish embryos, combining physics-based simulation with deep learning.

## Pipeline Overview

```
Raw FCS photon trace (.fcs / .raw)
        │
        ▼  Stage 1-2: Load + baseline normalization
        │
        ▼  Stage 3: Matched-filter burst detector (adaptive NMS)     [80.4% detection rate]
        │
        ▼  Stage 4: Feature extraction (212 ACF / PSD / morphology features)
        │
        ▼  Stage 5: WaveNet multi-dt diffusion estimator             [MAPE 9.9%]
                    Noise window classifier (SNR gating)
        │
        ▼  Events CSV: t_peak, duration, D_hat, confidence interval
```

## Key Results

| Model | Architecture | MAPE |
|-------|-------------|------|
| `multidt_B_large` | 3-branch WaveNet (dt = 0.1 / 0.5 / 1.0 ms) | **9.5%** |
| `multidt_B` | 3-branch WaveNet (dt = 0.1 / 0.5 / 1.0 ms) | 9.9% |
| `wavenet_wide_aug` | Single-dt WaveNet | 12.3% |
| MLP (212 features) | Feature-based baseline | ~18% |

Event detection: 80.4% exact-match recovery across 5 diffusion-coefficient bins (D = 0.01–10 μm²/s).

## Diffusion Estimator Architecture

Each event is processed at three time resolutions simultaneously:

```
photon trace
  ├─ dt = 0.1 ms (40 960 bins) → AvgPool(16) → WaveNetBackbone(256) ──┐
  ├─ dt = 0.5 ms  (8 192 bins) →    native    → WaveNetBackbone(256) ──┤ → concat → MLP → log₁₀(D)
  ├─ dt = 1.0 ms  (4 096 bins) →    native    → WaveNetBackbone(256) ──┤
  └─ 906 ACF / PSD features                   → feat_branch(512→128) ──┘
```

## Repository Structure

| File / Group | Description |
|---|---|
| `run_full_pipeline_v*.py` | Versioned end-to-end pipeline (detection → estimation → plots) |
| `stage3_v*.py` | Single-molecule burst detection modules (v2 → v3.95 → v4.4) |
| `pt_wavenet_multidt_*.py` | WaveNet training scripts (GPU, multi-resolution) |
| `pipeline.py`, `measurement.py` | Core pipeline logic and measurement abstraction |
| `model_registry.py`, `model_classes.py` | Model catalogue and inference wrappers |
| `compute_features*.py` | 212-feature ACF / PSD / morphology extraction |
| `generate_simulations*.py` | Physics-based Brownian diffusion simulation |
| `build_cache_multidt*.py` | Training cache construction for multi-dt WaveNet |
| `fcs_io.py`, `raw_io.py` | I/O for FCS and raw photon-count formats |
| `bookkeeper.py`, `orchestrator.py` | Experiment registry and SLURM job orchestration |
| `gui.py` | Tkinter GUI for local pipeline execution with real-time SLURM monitoring |
| `aggregate_events.py` | Postprocessor: aggregate events across experiments into a flat CSV |
| `job_*.sh` | SLURM job scripts for HPC execution |
| `diffusion_models.json`, `noise_models.json` | Model catalogues |
| `algorithm_sm_inference.md` | Full pipeline algorithm specification |

## Requirements

```
Python 3.10+, PyTorch ≥ 2.0, NumPy, SciPy, scikit-learn
```

HPC: SLURM cluster with NVIDIA GPU (tested on RTX 5000 Ada).

## Note on Data and Model Weights

Raw experimental data (`.fcs`, `.raw`) and trained model weights (`.pt`) are not included; they are large files stored on the HPC cluster.
