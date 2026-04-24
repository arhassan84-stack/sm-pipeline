#!/usr/bin/env python3
"""
extract_sim_traces.py
For each detected event in results_01082026/full_run/*_events.csv,
find a randomly selected simulation trace whose D is closest to D_hat
(in log space), and extract a 2-second window centred at the trace
midpoint (the molecule's closest approach to the beam centre).

Output
------
results_01082026/full_run/sim_traces/
    {stem}_{ch_id}_{t_peak:07d}ms_sim.npy   -- float32, 2000 bins (1ms)
    metadata.csv                              -- stem, ch_id, t_peak_ms,
                                                 D_hat, D_sim, sim_idx
"""

import csv
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
CSV_DIR = PROJECT / 'results_01082026' / 'run_v2_bl_fix'
OUT_DIR = CSV_DIR / 'sim_traces'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_D = PROJECT / 'sims_multidt_noise0_d.npy'
SIM_I = PROJECT / 'sims_multidt_noise0_i_dt100.npy'

N_BINS = 4096
MID    = N_BINS // 2 - 1   # 2047 — molecule at beam centre
HALF   = 1000               # ±1000 ms = 2-second window
WIN_S  = MID - HALF         # 1047  (inclusive)
WIN_E  = MID + HALF         # 3047  (inclusive) → slice [:3048]

rng = np.random.default_rng(seed=42)

print('Loading simulation D values …')
d_sim = np.load(SIM_D)
print(f'  shape {d_sim.shape},  D ∈ [{d_sim.min():.4f}, {d_sim.max():.4f}] µm²/s')

print('Memory-mapping intensity array …')
i_sim = np.load(SIM_I, mmap_mode='r')
print(f'  shape {i_sim.shape},  dtype {i_sim.dtype}')

unique_d   = np.unique(d_sim)
log_unique = np.log(unique_d)

# ── Collect all events ────────────────────────────────────────────────────────
events = []
for csv_path in sorted(CSV_DIR.glob('*_events.csv')):
    stem_ch = csv_path.stem.replace('_events', '')   # e.g. nt_dorsal_5_S1
    sep      = stem_ch.rfind('_')
    stem     = stem_ch[:sep]    # nt_dorsal_5
    ch_id    = stem_ch[sep+1:]  # S1
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            d_hat_str = row.get('D_hat', '')
            if not d_hat_str:
                continue
            d_hat = float(d_hat_str)
            if d_hat > 0:
                events.append((stem, ch_id, int(row['t_peak_ms']), d_hat))

print(f'\nFound {len(events)} events across all CSVs')

# ── Extract matching sim traces ───────────────────────────────────────────────
meta_rows = []
for i, (stem, ch_id, t_peak, d_hat) in enumerate(events):
    if i % 200 == 0:
        print(f'  {i}/{len(events)} …')

    # Closest D in log space
    log_d_hat   = np.log(d_hat)
    bin_idx     = int(np.argmin(np.abs(log_unique - log_d_hat)))
    d_sim_val   = float(unique_d[bin_idx])

    # All replicas with this D value
    same_d  = np.where(d_sim == d_sim_val)[0]
    chosen  = int(rng.choice(same_d))

    # Extract 2-second window; convert counts → kHz (for dt=1ms: kHz = counts)
    trace = i_sim[chosen, WIN_S : WIN_E + 1].astype(np.float32)

    fname = OUT_DIR / f'{stem}_{ch_id}_{t_peak:07d}ms_sim.npy'
    np.save(fname, trace)

    meta_rows.append({
        'stem':      stem,
        'ch_id':     ch_id,
        't_peak_ms': t_peak,
        'D_hat':     d_hat,
        'D_sim':     d_sim_val,
        'sim_idx':   chosen,
    })

# ── Metadata CSV ──────────────────────────────────────────────────────────────
meta_path = OUT_DIR / 'metadata.csv'
with open(meta_path, 'w', newline='') as f:
    writer = csv.DictWriter(
        f, fieldnames=['stem', 'ch_id', 't_peak_ms', 'D_hat', 'D_sim', 'sim_idx'])
    writer.writeheader()
    writer.writerows(meta_rows)

print(f'\nDone. {len(meta_rows)} sim traces → {OUT_DIR}')
print(f'Metadata → {meta_path}')
