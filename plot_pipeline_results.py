# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
plot_pipeline_results.py
Generate annotated multi-panel trace plots + D histograms from pipeline CSVs.
Saves to results_01082026/pipeline_plots/
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'
CSV_DIR  = PROJECT / 'results_01082026' / 'run_v2_bl_fix'
OUT_DIR  = PROJECT / 'results_01082026' / 'run_v2_bl_fix'

sys.path.insert(0, str(PROJECT))
from measurement import load_measurement
from pipeline    import _stage1_baseline

# ── Aesthetics ────────────────────────────────────────────────────────────────
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12
N_COLS      = 3
WIN_MS      = 10000   # 10 s per panel

FILES    = [f'nt_dorsal_{i}' for i in range(1, 20)]
RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)

# D colormap shared across all plots
CMAP_D = cm.RdYlBu_r
NORM_D = Normalize(vmin=-2.5, vmax=0.5)


# ── Minimal registry (no models needed for plotting) ─────────────────────────
class MockRegistry:
    class _DiffModels:
        def best_for(self, **kw): return None
    class _NoiseModels:
        _wrappers = []
        def best_for(self, *a, **kw): return None
    diffusion_models = _DiffModels()
    noise_models     = _NoiseModels()
    dt_min           = 0.1


# ── CSV reader ────────────────────────────────────────────────────────────────
def read_events_csv(path):
    events = []
    with open(path) as f:
        for row in csv.DictReader(f):
            events.append({
                't_left':  int(row['t_left_ms']),
                't_peak':  int(row['t_peak_ms']),
                't_right': int(row['t_right_ms']),
                'D_hat':   float(row['D_hat'])   if row['D_hat']   else np.nan,
                'log10_D': float(row['log10_D']) if row['log10_D'] else np.nan,
            })
    return events


# ── Trace plot ────────────────────────────────────────────────────────────────
def make_trace_plot(stem, ch_id, i100, baseline, events, out_path):
    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'
    N    = len(i100)
    t_ms = np.arange(N, dtype=float)

    n_panels = int(np.ceil(N / WIN_MS))
    n_rows   = int(np.ceil(n_panels / N_COLS))

    fig, axes = plt.subplots(n_rows, N_COLS,
                              figsize=(N_COLS * 5, n_rows * 2),
                              facecolor='white')
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{stem} / {ch_id}   {len(events)} events',
                 fontsize=FS_SUPTITLE, color='black')

    for idx, ax in enumerate(axes.flat):
        t0 = idx * WIN_MS
        t1 = t0 + WIN_MS
        if t0 >= N:
            ax.set_visible(False)
            continue

        seg   = i100[t0 : min(t1, N)]
        t_seg = t_ms[t0 : t0 + len(seg)]
        bl    = baseline[t0 : t0 + len(seg)]

        ax.set_facecolor('white')

        # Event boundary lines: blue = start, red = end
        for ev in events:
            if ev['t_right'] < t0 or ev['t_left'] > t1:
                continue
            if ev['t_left'] >= t0:
                ax.axvline(ev['t_left']  * 1e-3, color='blue', lw=1.5, ls='-')
            if ev['t_right'] <= t1:
                ax.axvline(ev['t_right'] * 1e-3, color='red',  lw=1.5, ls='-')

        # D value label at each event peak (fixed y near top)
        for ev in events:
            if not (t0 <= ev['t_peak'] <= min(t1, N - 1)):
                continue
            if not np.isnan(ev['D_hat']):
                ax.text(ev['t_peak'] * 1e-3, 110,
                        f"{ev['D_hat']:.3f}",
                        fontsize=6, ha='center', va='bottom',
                        color='black', rotation=90)

        # Baseline (dashed gray) then trace on top
        ax.plot(t_seg * 1e-3, bl,  color='gray', lw=0.8, alpha=0.7, ls='--')
        ax.plot(t_seg * 1e-3, seg, color=trace_color, lw=0.7)
        ax.set_ylim(bottom=0, top=120)

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW); sp.set_color('black')
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


# ── D histogram ───────────────────────────────────────────────────────────────
def make_d_histogram(stem, ch_id, events, out_path):
    log10_D = np.array([ev['log10_D'] for ev in events
                        if not np.isnan(ev['log10_D'])])
    if len(log10_D) == 0:
        return

    bins = np.arange(-2.5, 1.5 + 0.1, 0.1)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.set_facecolor('white')

    # Bars coloured by D value
    for left, right in zip(bins[:-1], bins[1:]):
        count = int(((log10_D >= left) & (log10_D < right)).sum())
        if count > 0:
            mid = 0.5 * (left + right)
            ax.bar(left, count, width=right - left, align='edge',
                   color=CMAP_D(NORM_D(mid)), edgecolor='white', linewidth=0.5)

    med    = float(np.median(log10_D))
    d_med  = 10.0 ** med
    ax.axvline(med, color='black', lw=1.5, ls='--',
               label=f'median = {d_med:.3f} µm²/s')
    ax.legend(fontsize=FS_TICK)

    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel('log₁₀ D̂  (µm²/s)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Events', fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  D̂ distribution  (n={len(log10_D)})',
                 fontsize=FS_TITLE, color='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path.name}')


# ── Per-event 2-second zoom plots ─────────────────────────────────────────────
EVT_HALF_MS  = 1000   # ±1 000 ms = 2-second window
EVT_DIR_NAME = 'event_plots'   # sub-folder inside OUT_DIR


def make_event_plots(stem, ch_id, i100, baseline, events, evt_dir):
    """
    For each event, plot a ±1 s (2 s total) window of i100 and baseline,
    mark t_left (blue) and t_right (red), and annotate D_hat.
    Files are saved to evt_dir as {stem}_{ch_id}_event_{t_peak_ms:07d}ms.png
    """
    evt_dir.mkdir(parents=True, exist_ok=True)
    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'
    N   = len(i100)
    t_ms = np.arange(N, dtype=float)

    for ev in events:
        tp   = ev['t_peak']
        t0   = max(0,     tp - EVT_HALF_MS)
        t1   = min(N - 1, tp + EVT_HALF_MS)

        seg  = i100[t0 : t1 + 1]
        bl   = baseline[t0 : t1 + 1]
        t_s  = t_ms[t0 : t1 + 1] * 1e-3

        fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')
        ax.set_facecolor('white')

        # Baseline and trace
        ax.plot(t_s, bl,  color='gray', lw=0.8, alpha=0.7, ls='--')
        ax.plot(t_s, seg, color=trace_color, lw=0.9)

        # Event boundaries (only if inside window)
        if ev['t_left']  >= t0:
            ax.axvline(ev['t_left']  * 1e-3, color='blue', lw=1.5, ls='-')
        if ev['t_right'] <= t1:
            ax.axvline(ev['t_right'] * 1e-3, color='red',  lw=1.5, ls='-')

        # Peak marker
        ax.axvline(tp * 1e-3, color='black', lw=0.8, ls=':', alpha=0.5)

        # D label near top-right of axis
        if not np.isnan(ev['D_hat']):
            ax.text(0.97, 0.93, f"D̂ = {ev['D_hat']:.3f} µm²/s",
                    transform=ax.transAxes,
                    fontsize=FS_LABEL, ha='right', va='top', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

        ax.set_ylim(bottom=0, top=120)
        ax.set_xlim(t0 * 1e-3, t1 * 1e-3)

        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW); sp.set_color('black')
        ax.tick_params(colors='black', labelsize=FS_TICK)
        ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
        ax.set_ylabel('kHz',      fontsize=FS_TICK,  color='black')
        ax.set_title(f'{stem} / {ch_id}  —  t_peak = {tp * 1e-3:.3f} s',
                     fontsize=FS_TITLE, color='black')

        fig.tight_layout()
        fname = evt_dir / f'{stem}_{ch_id}_event_{tp:07d}ms.png'
        fig.savefig(fname, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
registry = MockRegistry()

for stem in FILES:
    fcs_path = DATA_DIR / f'{stem}.fcs'
    print(f'\n{"="*60}\n  {stem}\n{"="*60}')

    meas = load_measurement(
        fcs_path = fcs_path,
        data_dir = DATA_DIR,
        registry = registry,
        raw_opts = RAW_OPTS,
    )

    for ch_id in meas.channel_ids:
        csv_path = CSV_DIR / f'{stem}_{ch_id}_events.csv'
        if not csv_path.exists():
            print(f'  [{ch_id}] no CSV — skipping')
            continue

        events   = read_events_csv(csv_path)
        i100     = meas.channels[ch_id].i100_full
        baseline = _stage1_baseline(i100)

        valid_d = [ev['log10_D'] for ev in events if not np.isnan(ev['log10_D'])]
        med_d   = 10.0 ** np.median(valid_d) if valid_d else float('nan')
        print(f'  [{ch_id}]  {len(events)} events  median D={med_d:.3f} µm²/s')

        make_trace_plot(stem, ch_id, i100, baseline, events,
                        OUT_DIR / f'{stem}_{ch_id}_trace.png')
        make_d_histogram(stem, ch_id, events,
                         OUT_DIR / f'{stem}_{ch_id}_d_hist.png')
        make_event_plots(stem, ch_id, i100, baseline, events,
                         OUT_DIR / EVT_DIR_NAME)
        print(f'     {len(events)} event zoom plots → {EVT_DIR_NAME}/')

print('\nDone.')
