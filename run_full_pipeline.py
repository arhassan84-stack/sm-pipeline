#!/usr/bin/env python3
"""
run_full_pipeline.py
Run stage-3 pipeline on all 19 nt_dorsal measurements and save
events CSVs to results_01082026/full_run/.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s',
                    stream=sys.stdout)

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
OUT_DIR  = PROJECT / 'results_01082026' / 'run_v2_bl_fix'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT))

from model_registry import ModelRegistry
from measurement    import load_measurement
from pipeline       import run_pipeline

registry = ModelRegistry(
    noise_json     = str(PROJECT / 'noise_models.json'),
    diffusion_json = str(PROJECT / 'diffusion_models.json'),
    model_dir      = str(PROJECT),
)
registry.load_all()
print(f'Loaded {len(registry.diffusion_models._wrappers)} diffusion model(s), '
      f'{len(registry.noise_models._wrappers)} noise model(s)')

FILES    = [f'nt_dorsal_{i}' for i in range(1, 20)]
RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)

for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    print(f'\n{"="*60}\n  {stem}\n{"="*60}')
    try:
        meas = load_measurement(fcs_path=fcs_path, data_dir=DATA_DIR,
                                registry=registry, raw_opts=RAW_OPTS)
        results = run_pipeline(meas=meas, registry=registry, out_dir=OUT_DIR,
                               save_csv=True, save_plots=False)
        for res in results:
            print(f'  [{res.channel_id}]  {res.n_events} events  '
                  f'(model: {res.d_model_label})')
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print(f'\nDone. CSVs in {OUT_DIR}')
