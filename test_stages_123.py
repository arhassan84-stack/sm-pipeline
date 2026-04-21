# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
test_stages_123.py
Run Stages 0-3 locally (no torch required) using a mock D model that
returns a fixed D value.  Verifies that:
  - RAW files are loaded correctly with the new bits=32 / n_header=32 params
  - measurement.py produces valid i100_full traces
  - Stages 1-3 (baseline, background subtraction, event detection) run cleanly

Usage:
    python3 test_stages_123.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s',
                    stream=sys.stdout)

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_DIR  = PROJECT / 'results_01082026' / 'pipeline_test'

sys.path.insert(0, str(PROJECT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from measurement import load_measurement
from stage3_v2   import run_stage3_v2, TraceBundle
from pipeline    import _stage1_baseline, _stage2_subtract

# ── Aesthetics (MEMORY.md) ────────────────────────────────────────────────────
CURVE_COLOR = '#00FF00'
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12
N_COLS      = 3
N_ROWS      = 6
WIN_MS      = 2000
FOLD_MIN_S1 = 3.0    # S1 (GFP):    peak must satisfy (peak-bl)/bl >= 3  →  ≥ 4× baseline
FOLD_MIN_S2 = 2.0    # S2 (mCherry): peak must satisfy (peak-bl)/bl >= 2  →  ≥ 3× baseline


def make_plot(stem, ch_id, i100, baseline, events, out_path):
    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'
    N    = len(i100)
    t_ms = np.arange(N, dtype=float)

    win_ms   = 10000   # 10 seconds per panel
    n_panels = int(np.ceil(N / win_ms))
    n_rows   = int(np.ceil(n_panels / N_COLS))

    cmap = cm.RdYlBu_r
    norm = Normalize(vmin=-2.5, vmax=-0.5)

    n_panels = N_ROWS * N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS,
                              figsize=(N_COLS * 5, n_rows * 2),
                              facecolor='white')
    fig.suptitle(f'{stem} / {ch_id}   {len(events)} events',
                 fontsize=FS_SUPTITLE, color='black')

    for idx, ax in enumerate(axes.flat):
        t0 = idx * win_ms
        t1 = t0 + win_ms
        if t0 >= N:
            ax.set_visible(False)
            continue

        seg  = i100[t0 : min(t1, N)]
        t_seg = t_ms[t0 : t0 + len(seg)]
        bl   = baseline[t0 : t0 + len(seg)]

        ax.set_facecolor('white')

        # Event boundary lines
        for ev in events:
            if ev['t_right'] < t0 or ev['t_left'] > t1:
                continue
            if ev['t_left'] >= t0:
                ax.axvline(ev['t_left']  * 1e-3, color='blue', lw=2.0, ls='-')
            if ev['t_right'] <= t1:
                ax.axvline(ev['t_right'] * 1e-3, color='red',  lw=2.0, ls='-')

        # Baseline
        ax.plot(t_seg * 1e-3, bl, color='gray', lw=0.8, alpha=0.7, ls='--')
        # Intensity trace on top
        ax.plot(t_seg * 1e-3, seg, color=trace_color, lw=0.7)
        ax.set_ylim(bottom=0, top=120)

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_color('black')
        ax.tick_params(colors='black', labelsize=FS_TICK)
        ax.set_xlim(t0 * 1e-3, t1 * 1e-3)
        ax.set_title(f'{t0/1000:.0f}–{t1/1000:.0f} s',
                     fontsize=FS_TITLE, color='black', pad=2)
        if idx % N_COLS == 0:
            ax.set_ylabel('kHz', fontsize=FS_TICK, color='black')
        if idx // N_COLS == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path.name}')


# ── Mock D model ──────────────────────────────────────────────────────────────

class MockDModel:
    """
    Minimal stand-in for ModelWrapper.  Returns a fixed D value so that
    the expansion loop runs without torch.  D=0.3 um2/s -> tau_max=512ms.
    """
    label                = 'mock_D0.3'
    dt_channels          = {1.0}          # only 1ms channel needed
    requires_acf_features = False
    min_trace_length_ms  = 0
    max_trace_length_ms  = 1e9
    requires_raw         = False

    def predict(self, traces, features=None):
        return 0.3   # um2/s

    def predict_batch(self, traces_list, features_list=None, batch_size=128):
        return [0.3] * len(traces_list)


class MockRegistry:
    """Minimal registry that returns the mock model."""
    class _DiffModels:
        def best_for(self, trace_length_ms=None, has_raw=None):
            return MockDModel()
    class _NoiseModels:
        _wrappers = []
        def best_for(self, *a, **kw):
            return None
    diffusion_models = _DiffModels()
    noise_models     = _NoiseModels()
    dt_min           = 0.1   # needed by measurement.py


# ── Files ─────────────────────────────────────────────────────────────────────

FILES    = ['nt_dorsal_1', 'nt_dorsal_10', 'nt_dorsal_15']
RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)

OUT_DIR.mkdir(parents=True, exist_ok=True)
registry = MockRegistry()

# ── Run ───────────────────────────────────────────────────────────────────────

for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    print('\n' + '='*60)
    print(f'  {stem}')
    print('='*60)

    try:
        # Stage 0: load measurement
        meas = load_measurement(
            fcs_path = fcs_path,
            data_dir = DATA_DIR,
            registry = registry,
            raw_opts = RAW_OPTS,
        )
        print(f'  has_raw={meas.has_raw}  dur={meas.total_duration_ms/1000:.0f}s  '
              f'channels={meas.channel_ids}')

        for ch_id in meas.channel_ids:
            ch_data  = meas.channels[ch_id]
            i100     = ch_data.i100_full
            print(f'\n  [{ch_id}]  len(i100)={len(i100)}  '
                  f'mean={i100.mean():.2f} kHz  max={i100.max():.1f} kHz')

            # Stage 1: baseline
            baseline = _stage1_baseline(i100)
            print(f'         baseline: mean={baseline.mean():.2f} kHz  '
                  f'min={baseline.min():.2f}  max={baseline.max():.2f}')

            # Stage 2: background subtraction
            i_min_c, i050_c, i100_c = _stage2_subtract(ch_data, baseline, meas)

            # Stage 3: event detection with mock D model
            d_model = MockDModel()
            bundle  = TraceBundle(
                i1ms     = i100,
                baseline = baseline,
                i_dt050  = i050_c,
                i_dt010  = i_min_c,
            )
            amp_thresh = 50.0 if ch_id == 'S1' else 30.0
            fold_min   = FOLD_MIN_S1 if ch_id == 'S1' else FOLD_MIN_S2
            events, z_score = run_stage3_v2(bundle, d_model,
                                            amplitude_min_khz=amp_thresh,
                                            fold_min=fold_min)

            print(f'         events detected: {len(events)}')
            if events:
                amps = [float(i100[e['t_peak']]) for e in events]
                ws   = [e['W_seg'] for e in events]
                print(f'         amplitude: min={min(amps):.1f}  '
                      f'mean={np.mean(amps):.1f}  max={max(amps):.1f} kHz')
                print(f'         W_seg:     min={min(ws)}  '
                      f'mean={np.mean(ws):.0f}  max={max(ws)} ms')
                n_cat1 = sum(1 for e in events if e.get('category') == 1)
                n_cat2 = sum(1 for e in events if e.get('category') == 2)
                print(f'         cat1={n_cat1}  cat2={n_cat2}  '
                      f'fast={sum(1 for e in events if e.get("fast_group"))}')

            out = OUT_DIR / f'{stem}_{ch_id}_stages123.png'
            make_plot(stem, ch_id, i100, baseline, events, out)

    except Exception as e:
        import traceback
        print(f'  ERROR: {e}')
        traceback.print_exc()

print('\nDone.')
