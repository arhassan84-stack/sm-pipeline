#!/usr/bin/env python3
"""
test_pipeline.py
Run the updated pipeline (sliding-window KDE baseline) on a small selection
of real FCS files from the 01082026 dataset.

For each file × channel, prints stage-by-stage diagnostics and saves:
  results_01082026/{stem}_{channel}_events.csv
  results_01082026/{stem}_{channel}_trace.png
  results_01082026/{stem}_{channel}_d_scatter.png
  results_01082026/{stem}_{channel}_d_histogram.png
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(levelname)s  %(message)s',
    stream  = sys.stdout,
)

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_DIR  = PROJECT / 'results_01082026' / 'pipeline_test'

sys.path.insert(0, str(PROJECT))

from model_registry import ModelRegistry
from measurement    import load_measurement
from pipeline       import run_pipeline

# ── Model registry ────────────────────────────────────────────────────────────
registry = ModelRegistry(
    noise_json     = str(PROJECT / 'noise_models.json'),
    diffusion_json = str(PROJECT / 'diffusion_models.json'),
    model_dir      = str(PROJECT),
)
registry.load_all()
print(f'\nLoaded {len(registry.diffusion_models._wrappers)} diffusion model(s), '
      f'{len(registry.noise_models._wrappers)} noise model(s)')

# ── Files to test ─────────────────────────────────────────────────────────────
FILES = [
    'nt_dorsal_1',
    'nt_dorsal_10',
    'nt_dorsal_15',
]

RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────
for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    print(f'\n{"="*60}')
    print(f'  {stem}')
    print(f'{"="*60}')

    try:
        meas = load_measurement(
            fcs_path = fcs_path,
            data_dir = DATA_DIR,
            registry = registry,
            raw_opts = RAW_OPTS,
        )
        print(f'  Loaded: has_raw={meas.has_raw}, '
              f'dur={meas.total_duration_ms/1000:.0f}s, '
              f'channels={meas.channel_ids}')

        results = run_pipeline(
            meas       = meas,
            registry   = registry,
            out_dir    = OUT_DIR,
            save_csv   = True,
            save_plots = True,
        )

        for res in results:
            print(f'\n  [{res.channel_id}] '
                  f'{res.n_events} events passed, '
                  f'{len(res.rejected)} rejected  '
                  f'(model: {res.d_model_label})')

    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print(f'\nDone. Results in {OUT_DIR}')
