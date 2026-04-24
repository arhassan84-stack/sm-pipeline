#!/usr/bin/env python3
"""
regen_event_plots.py
Re-generate per-event zoom plots as 2-panel figures:
  Top    — real event (±1 s window, baseline overlay, event boundaries)
  Bottom — randomly-selected noise-free sim trace with closest D (centred ±1 s)

Overwrites results_01082026/run_v2_bl_fix/event_plots/
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
FULL_DIR = PROJECT / 'results_01082026' / 'run_v2_bl_fix'
SIM_DIR  = FULL_DIR / 'sim_traces'
EVT_DIR  = FULL_DIR / 'event_plots'
EVT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT))
from measurement import load_measurement
from pipeline    import _stage1_baseline

RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)
FILES    = [f'nt_dorsal_{i}' for i in range(1, 20)]

EVT_HALF = 1000   # ±1000 ms = 2-second window

# ── Aesthetics ────────────────────────────────────────────────────────────────
SPINE_LW  = 1.5
FS_TITLE  = 12
FS_LABEL  = 11
FS_TICK   = 9
FS_ANNOT  = 10


class MockRegistry:
    class _DiffModels:
        def best_for(self, **kw): return None
    class _NoiseModels:
        _wrappers = []
        def best_for(self, *a, **kw): return None
    diffusion_models = _DiffModels()
    noise_models     = _NoiseModels()
    dt_min           = 0.1


# ── Load sim metadata ─────────────────────────────────────────────────────────
sim_meta = {}   # (stem, ch_id, t_peak_ms) → {D_hat, D_sim}
meta_path = SIM_DIR / 'metadata.csv'
if meta_path.exists():
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            key = (row['stem'], row['ch_id'], int(row['t_peak_ms']))
            sim_meta[key] = {
                'D_hat': float(row['D_hat']),
                'D_sim': float(row['D_sim']),
            }
print(f'Loaded {len(sim_meta)} sim-metadata entries')


def _make_plot(stem, ch_id, t_peak, i100, baseline,
               t_left, t_right,
               D_hat, sim_trace, D_sim, out_path):
    N  = len(i100)
    t0 = max(0,     t_peak - EVT_HALF)
    t1 = min(N - 1, t_peak + EVT_HALF)

    seg  = i100[t0 : t1 + 1]
    bl   = baseline[t0 : t1 + 1]
    t_s  = np.arange(t0, t0 + len(seg)) * 1e-3   # absolute time in s

    # Sim time axis (centred at 0)
    n_sim   = len(sim_trace)
    t_s_sim = (np.arange(n_sim) - (n_sim // 2)) * 1e-3

    # S1 → normalise both panels to [0, 1]; S2 → raw kHz
    normalize = (ch_id == 'S1')
    if normalize:
        seg_max = seg.max()  if seg.max()  > 0 else 1.0
        sim_max = sim_trace.max() if sim_trace.max() > 0 else 1.0
        seg_plot = seg / seg_max
        bl_plot  = bl  / seg_max
        sim_plot = sim_trace / sim_max
        y_label  = 'Norm. intensity'
        y_lim    = (0, 1.15)
    else:
        seg_plot = seg
        bl_plot  = bl
        sim_plot = sim_trace
        y_label  = 'kHz'
        y_lim    = None

    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 5), facecolor='white',
        gridspec_kw={'hspace': 0.50})

    # ── Top: real event ───────────────────────────────────────────────────────
    ax_top.set_facecolor('white')
    ax_top.plot(t_s, bl_plot,  color='gray', lw=0.8, alpha=0.7, ls='--')
    ax_top.plot(t_s, seg_plot, color=trace_color, lw=0.9)
    ax_top.axvline(t_peak * 1e-3, color='black', lw=0.7, ls=':', alpha=0.5)
    if t_left  is not None and t_left  >= t0:
        ax_top.axvline(t_left  * 1e-3, color='blue', lw=1.5, ls='-')
    if t_right is not None and t_right <= t1:
        ax_top.axvline(t_right * 1e-3, color='red',  lw=1.5, ls='-')

    if D_hat is not None and not np.isnan(D_hat):
        ax_top.text(0.97, 0.93, f'D\u0302 = {D_hat:.3f} \u00b5m\u00b2/s',
                    transform=ax_top.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    if y_lim:
        ax_top.set_ylim(*y_lim)
    else:
        ax_top.set_ylim(0, 120)
    ax_top.set_xlim(t0 * 1e-3, t1 * 1e-3)
    for sp in ax_top.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')
    ax_top.tick_params(colors='black', labelsize=FS_TICK)
    ax_top.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_top.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    ax_top.set_title(f'{stem} / {ch_id}  —  t_peak = {t_peak * 1e-3:.3f} s',
                     fontsize=FS_TITLE, color='black', pad=4)

    # ── Bottom: sim trace ─────────────────────────────────────────────────────
    ax_bot.set_facecolor('white')
    ax_bot.plot(t_s_sim, sim_plot, color='steelblue', lw=0.9)
    ax_bot.axvline(0, color='black', lw=0.7, ls=':', alpha=0.5)

    ax_bot.text(0.97, 0.93, f'sim D = {D_sim:.4f} \u00b5m\u00b2/s',
                transform=ax_bot.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top', color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    if y_lim:
        ax_bot.set_ylim(*y_lim)
    else:
        ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlim(-1.0, 1.0)
    for sp in ax_bot.spines.values():
        sp.set_linewidth(SPINE_LW); sp.set_color('black')
    ax_bot.tick_params(colors='black', labelsize=FS_TICK)
    ax_bot.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_bot.set_xlabel('Time rel. to transit centre (s)', fontsize=FS_LABEL, color='black')
    ax_bot.set_title('Simulated transit (noise-free)', fontsize=FS_TITLE,
                     color='black', pad=4)

    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ── Main loop ─────────────────────────────────────────────────────────────────
registry = MockRegistry()

for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    print(f'\n{"="*50}\n  {stem}\n{"="*50}')

    try:
        meas = load_measurement(fcs_path=fcs_path, data_dir=DATA_DIR,
                                registry=registry, raw_opts=RAW_OPTS)
    except Exception as e:
        print(f'  ERROR loading: {e}')
        continue

    for ch_id in meas.channel_ids:
        csv_path = FULL_DIR / f'{stem}_{ch_id}_events.csv'
        if not csv_path.exists():
            print(f'  [{ch_id}] no CSV — skipping')
            continue

        i100     = meas.channels[ch_id].i100_full
        baseline = _stage1_baseline(i100)

        with open(csv_path) as f:
            events = list(csv.DictReader(f))

        n_ok = 0
        for row in events:
            t_peak  = int(row['t_peak_ms'])
            t_left  = int(row['t_left_ms'])  if 't_left_ms'  in row else None
            t_right = int(row['t_right_ms']) if 't_right_ms' in row else None
            D_hat   = float(row['D_hat'])    if row.get('D_hat') else None

            key      = (stem, ch_id, t_peak)
            sim_file = SIM_DIR / f'{stem}_{ch_id}_{t_peak:07d}ms_sim.npy'
            if key not in sim_meta or not sim_file.exists():
                continue

            sim_trace = np.load(sim_file) * 1e-3   # counts/s → kHz
            D_sim     = sim_meta[key]['D_sim']

            out_path = EVT_DIR / f'{stem}_{ch_id}_event_{t_peak:07d}ms.png'
            _make_plot(stem, ch_id, t_peak, i100, baseline,
                       t_left, t_right, D_hat, sim_trace, D_sim, out_path)
            n_ok += 1

        print(f'  [{ch_id}]  {n_ok} plots saved')

print(f'\nDone. All plots in {EVT_DIR}')
