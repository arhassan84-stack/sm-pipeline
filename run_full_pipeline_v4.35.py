#!/usr/bin/env python3
"""
run_full_pipeline_v4.35.py
Single-pass integrated pipeline + all figures for cluster execution.

Changes vs v4.33:
  - knee_opt/ subfolder added inside event_plots/.
    For each event the plot at the knee factor is copied there with a
    'g_' prefix (mirroring the pearson_opt/ convention).

Changes vs v4.32 (inherited from v4.33):
  - Knee detection on the photon-count vs factor curve (two-segment linear
    fit / L-method). The knee factor is marked as a vertical orange dashed
    line on each phot_vs_factor plot.
  - metadata.csv gains 'knee_factor' column.
  - Per-measurement and grand-summary bar charts of:
      (a) Pearson best-match factor frequency → pearson_opt/
      (b) Knee factor frequency               → phot_vs_factor/
    Grand-summary charts named all_{ch_id}_*_freq.png, produced once
    across all measurements per channel.

For each measurement (one loop):
  1. run_pipeline (via stage3_v4_3 + pearson_fn)  → {stem}_{ch}_events.csv
  2. make_trace_plot             → {stem}_{ch}_trace.png
  3. make_d_histogram            → {stem}_{ch}_d_hist.png
  4. extract sim trace per event × factor → sim_traces/{stem}_{ch}_{t}ms_sim_f{fac}.npy
  5. make 3-panel event plot     → event_plots/{stem}_{ch}_event_{t}ms_f{fac}.png
  6. make photon-count plot      → event_plots/phot_vs_factor/{stem}_{ch}_event_{t}ms_phot.png

After the loop:
  7. Write sim_traces/metadata.csv
"""

import argparse
import csv
import datetime
import json
import logging
import shutil
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s',
                    stream=sys.stdout)

PIPELINE_VERSION = 'v4.35'   # hardcoded — do not import from version.py

# v4.31 baseline subtraction factors (must match stage3_v4_3.BL_FACTORS)
BL_FACTORS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
              1.40, 1.45, 1.50, 1.60, 1.80, 2.00]

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='FCS pipeline + plots')
parser.add_argument('data_folder', help='Full path to the experiment data folder')
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).parent
DATA_DIR = Path(args.data_folder).resolve()
if not DATA_DIR.is_dir():
    sys.exit(f'ERROR: data folder not found: {DATA_DIR}')
RUN_DATE = 'processed_on_' + datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
OUT_DIR  = DATA_DIR / RUN_DATE
SIM_DIR  = OUT_DIR / 'sim_traces'
EVT_DIR  = OUT_DIR / 'event_plots'
OPT_DIR      = EVT_DIR / 'pearson_opt'
PHOT_DIR     = EVT_DIR / 'phot_vs_factor'
KNEE_OPT_DIR = EVT_DIR / 'knee_opt'
for d in [OUT_DIR, SIM_DIR, EVT_DIR, OPT_DIR, PHOT_DIR, KNEE_OPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STARTED_AT = datetime.datetime.now()

# Simulation libraries (noise-free, dt=1ms)
SIM_D_PATH_S1 = PROJECT / 'sims_multidt_noise0_d.npy'
SIM_I_PATH_S1 = PROJECT / 'sims_multidt_noise0_i_dt100.npy'
SIM_D_PATH_S2 = PROJECT / 'sims_multidt_S2_noise0_d.npy'
SIM_I_PATH_S2 = PROJECT / 'sims_multidt_S2_noise0_i_dt100.npy'

sys.path.insert(0, str(PROJECT))

from model_registry import ModelRegistry
from measurement    import load_measurement
from pipeline       import run_pipeline, _stage1_baseline

# ── Constants ─────────────────────────────────────────────────────────────────
FILES    = sorted(p.stem for p in DATA_DIR.glob('*.fcs'))
if not FILES:
    sys.exit(f'ERROR: no .fcs files found in {DATA_DIR}')
RAW_OPTS = dict(bits=32, n_header=32, clock_rate_hz=15_000_000)

# Sim window extraction (dt=1ms, 4096-bin traces, ±1000 ms around transit centre)
N_BINS_SIM = 4096
MID_SIM    = N_BINS_SIM // 2 - 1   # 2047
HALF_SIM   = 1000
WIN_S      = MID_SIM - HALF_SIM    # 1047
WIN_E      = MID_SIM + HALF_SIM    # 3047  → slice [:3048]

# Event zoom window
EVT_HALF = 1000   # ±1000 ms

# ── Aesthetics ─────────────────────────────────────────────────────────────────
SPINE_LW    = 1.5
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_TICK     = 12
FS_ANNOT    = 11
N_COLS      = 3
WIN_MS      = 10_000   # 10 s per trace-plot panel

CMAP_D = cm.RdYlBu_r
NORM_D = Normalize(vmin=-2.5, vmax=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _spines(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
        sp.set_color('black')
    ax.tick_params(colors='black', labelsize=FS_TICK)


def make_trace_plot(stem, ch_id, i100, baseline, events, out_path):
    """Annotated multi-panel trace with event boundaries and D labels."""
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
        for ev in events:
            if ev['t_right'] < t0 or ev['t_left'] > t1:
                continue
            if ev['t_left'] >= t0:
                ax.axvline(ev['t_left']  * 1e-3, color='blue', lw=1.5)
            if ev['t_right'] <= t1:
                ax.axvline(ev['t_right'] * 1e-3, color='red',  lw=1.5)

        for ev in events:
            if not (t0 <= ev['t_peak'] <= min(t1, N - 1)):
                continue
            if not np.isnan(ev['D_hat']):
                ax.text(ev['t_peak'] * 1e-3, 110,
                        f"{ev['D_hat']:.3f}",
                        fontsize=6, ha='center', va='bottom',
                        color='black', rotation=90)

        ax.plot(t_seg * 1e-3, bl,  color='gray', lw=0.8, alpha=0.7, ls='--')
        ax.plot(t_seg * 1e-3, seg, color=trace_color, lw=0.7)
        ax.set_ylim(0, 120)
        _spines(ax)
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
    print(f'    trace  → {out_path.name}')


def make_d_histogram(stem, ch_id, events, out_path):
    """D̂ histogram with coloured bars on a log x-axis."""
    log10_D = np.array([ev['log10_D'] for ev in events
                        if not np.isnan(ev['log10_D'])])
    if len(log10_D) == 0:
        return

    D_vals = 10.0 ** log10_D
    bins   = np.logspace(-2.5, 1.5, 41)   # 40 log-spaced bins, same extent as before
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.set_facecolor('white')

    for left, right in zip(bins[:-1], bins[1:]):
        count = int(((D_vals >= left) & (D_vals < right)).sum())
        if count > 0:
            mid_log = 0.5 * (np.log10(left) + np.log10(right))
            ax.bar(left, count, width=right - left, align='edge',
                   color=CMAP_D(NORM_D(mid_log)),
                   edgecolor='white', linewidth=0.5)

    med   = float(np.median(log10_D))
    d_med = 10.0 ** med
    ax.axvline(d_med, color='black', lw=1.5, ls='--',
               label=f'median = {d_med:.3f} µm²/s')
    ax.set_xscale('log')
    ax.legend(fontsize=FS_TICK)
    _spines(ax)
    ax.set_xlabel('D̂  (µm²/s)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Events',      fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  D̂ distribution  (n={len(log10_D)})',
                 fontsize=FS_TITLE, color='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'    d_hist → {out_path.name}')


def make_comparison_d_histogram(stem, ch_id, d_pearson, d_knee, out_path):
    """Overlapping D̂ histograms on a log x-axis: Pearson-opt vs Knee-opt."""
    d_p = np.array([v for v in d_pearson if np.isfinite(v) and v > 0])
    d_k = np.array([v for v in d_knee    if np.isfinite(v) and v > 0])
    if len(d_p) == 0 and len(d_k) == 0:
        return

    bins = np.logspace(-2.5, 1.5, 41)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.set_facecolor('white')

    if len(d_p) > 0:
        ax.hist(d_p, bins=bins, alpha=0.55, color='steelblue',
                edgecolor='white', linewidth=0.4,
                label=f'Pearson opt  (n={len(d_p)})')
        med_p = float(np.median(d_p))
        ax.axvline(med_p, color='steelblue', lw=1.5, ls='--',
                   label=f'median = {med_p:.3f} µm²/s')

    if len(d_k) > 0:
        ax.hist(d_k, bins=bins, alpha=0.55, color='orange',
                edgecolor='white', linewidth=0.4,
                label=f'Knee opt     (n={len(d_k)})')
        med_k = float(np.median(d_k))
        ax.axvline(med_k, color='orange', lw=1.5, ls='--',
                   label=f'median = {med_k:.3f} µm²/s')

    ax.set_xscale('log')
    ax.legend(fontsize=FS_TICK)
    _spines(ax)
    ax.set_xlabel('D̂  (µm²/s)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Events',      fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  Pearson vs Knee D̂ distribution',
                 fontsize=FS_TITLE, color='black')
    ax.tick_params(axis='both', labelsize=FS_TICK, colors='black')
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'    d_hist_compare → {out_path.name}')


def make_scatter_window_d(stem, ch_id,
                          ws_pearson, d_pearson,
                          ws_knee,    d_knee,
                          out_path):
    """Scatter: window size (ms) vs D̂ (µm²/s) — Pearson-opt vs Knee-opt."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.set_facecolor('white')

    if len(ws_pearson) > 0:
        ax.scatter(ws_pearson, d_pearson,
                   color='steelblue', marker='o', s=30, alpha=0.7, linewidths=0,
                   label=f'Pearson opt  (n={len(ws_pearson)})')
    if len(ws_knee) > 0:
        ax.scatter(ws_knee, d_knee,
                   color='orange', marker='^', s=30, alpha=0.7, linewidths=0,
                   label=f'Knee opt     (n={len(ws_knee)})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=FS_TICK)
    _spines(ax)
    ax.set_xlabel('Window size (ms)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('D̂ (µm²/s)',        fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  Window size vs D̂',
                 fontsize=FS_TITLE, color='black')
    ax.tick_params(axis='both', labelsize=FS_TICK, colors='black')
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'    scatter_window_d → {out_path.name}')


def _detect_knee(factors, counts):
    """
    Return the BL_FACTOR at the knee of the photon-count curve.

    Uses the L-method (two-segment linear fit): for each candidate
    breakpoint k, fit a line to points [0..k] and another to [k..N-1],
    then return the factor at the breakpoint that minimises total SSE.
    Robust for the 13 sparse points in BL_FACTORS.
    """
    x = np.array(factors, dtype=np.float64)
    y = np.array(counts,  dtype=np.float64)
    n = len(x)
    if n < 3:
        return factors[0]

    best_sse = np.inf
    knee_idx = 1
    for k in range(1, n - 1):
        xl, yl = x[:k + 1], y[:k + 1]
        xr, yr = x[k:],     y[k:]
        pl     = np.polyfit(xl, yl, 1) if len(xl) >= 2 else np.array([0.0, yl[0]])
        pr     = np.polyfit(xr, yr, 1) if len(xr) >= 2 else np.array([0.0, yr[0]])
        sse    = float(np.sum((yl - np.polyval(pl, xl)) ** 2) +
                       np.sum((yr - np.polyval(pr, xr)) ** 2))
        if sse < best_sse:
            best_sse = sse
            knee_idx = k
    return factors[knee_idx]


def make_phot_plot(stem, ch_id, t_peak, phot_by_factor, raw_photon_count, out_path,
                   knee_factor=None):
    """Photon count vs baseline factor for one event.
    If knee_factor is provided, marks it with a vertical orange dashed line.
    """
    factors = sorted(phot_by_factor.keys())
    counts  = [phot_by_factor[f] for f in factors]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(factors, counts, color='steelblue', lw=1.5, marker='o', markersize=5)
    ax.axhline(raw_photon_count, color='gray', lw=1.0, ls='--', alpha=0.7)
    if knee_factor is not None:
        ax.axvline(knee_factor, color='orange', lw=1.5, ls='--',
                   label=f'Knee  ×{knee_factor:.2f}')
        ax.legend(fontsize=FS_TICK, framealpha=0.8)
    _spines(ax)
    ax.set_xlabel('Baseline factor', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Photon count', fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  t_peak = {t_peak * 1e-3:.3f} s',
                 fontsize=FS_TITLE, color='black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def make_factor_freq_plot(title, factor_counts, out_path):
    """Bar chart of how often each BL_FACTOR was selected (best-Pearson or knee).

    The bar with the highest count is gold; all others steelblue.
    A grey dashed line marks the uniform-chance reference (total / n_factors).
    The mode is annotated above its bar.
    """
    factors = sorted(factor_counts.keys())
    counts  = [factor_counts[f] for f in factors]
    total   = sum(counts)
    if total == 0:
        return

    mode_idx = int(np.argmax(counts))
    mode_f   = factors[mode_idx]
    mode_n   = counts[mode_idx]
    colors   = ['#FFD700' if f == mode_f else 'steelblue' for f in factors]
    x_pos    = np.arange(len(factors))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.bar(x_pos, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(total / len(factors), color='gray', lw=1.0, ls='--', alpha=0.7)
    ax.text(mode_idx, mode_n, f'×{mode_f:.2f}\nn={mode_n}',
            ha='center', va='bottom', fontsize=FS_ANNOT,
            color='#8B6914', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{f:.2f}' for f in factors], rotation=45, ha='right')
    _spines(ax)
    ax.set_xlabel('Baseline factor', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Events',          fontsize=FS_LABEL, color='black')
    ax.set_title(title,              fontsize=FS_TITLE, color='black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def make_event_plot_3panel(stem, ch_id, t_peak, i100, baseline,
                           t_left, t_right, D_hat,
                           sim_trace, D_sim, out_path,
                           category=None, factor=1.0,
                           pearson_score=0.0, is_best_pearson=False):
    """3-panel event figure: raw (top) + factor*baseline-subtracted (middle) + sim (bottom).

    Parameters
    ----------
    factor : float
        Baseline subtraction multiplier (one of BL_FACTORS).
        The suptitle shows 'factor = {factor:.2f}'.
        The middle panel subtracts factor * edge_bl from the trace.
    """
    N  = len(i100)
    t0 = max(0,     t_peak - EVT_HALF)
    t1 = min(N - 1, t_peak + EVT_HALF)

    seg = i100[t0 : t1 + 1]
    bl  = baseline[t0 : t1 + 1]

    # Edge baseline (minimum of the two event-window edges), then scaled by factor
    tl_eff  = max(0, min(t_left,  N - 1)) if t_left  is not None else t0
    tr_eff  = max(0, min(t_right, N - 1)) if t_right is not None else t1
    edge_bl = float(min(baseline[tl_eff], baseline[tr_eff]))
    seg_bl  = np.maximum(0.0, seg - factor * edge_bl)   # factor-scaled subtraction

    t_s    = np.arange(t0, t0 + len(seg)) * 1e-3

    n_sim   = len(sim_trace)
    t_s_sim = (np.arange(n_sim) - (n_sim // 2)) * 1e-3

    normalize   = False   # v4.31: both channels show actual intensity (kHz)
    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'

    if normalize:
        seg_max    = seg.max()       if seg.max()       > 0 else 1.0
        seg_bl_max = seg_bl.max()    if seg_bl.max()    > 0 else 1.0
        sim_max    = sim_trace.max() if sim_trace.max() > 0 else 1.0
        seg_plot    = seg    / seg_max
        bl_plot     = bl     / seg_max
        seg_bl_plot = seg_bl / seg_bl_max
        sim_plot    = sim_trace / sim_max
        y_label     = 'Norm. intensity'
        y_lim       = (0, 1.15)
    else:
        seg_plot    = seg
        bl_plot     = bl
        seg_bl_plot = seg_bl
        sim_plot    = sim_trace
        y_label     = 'kHz'
        y_lim       = None

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(6, 7.5), facecolor='white',
        gridspec_kw={'hspace': 0.66})

    if is_best_pearson:
        sup_text  = f'\u2605 Best Pearson  |  factor = {factor:.2f}'
        sup_color = '#FFD700'
    else:
        sup_text  = f'factor = {factor:.2f}'
        sup_color = 'black'
    fig.suptitle(sup_text, fontsize=FS_SUPTITLE, color=sup_color)

    # ── Top: raw trace + baseline ──────────────────────────────────────────
    ax_top.set_facecolor('white')
    ax_top.plot(t_s, bl_plot,  color='gray', lw=0.8, alpha=0.7, ls='--')
    ax_top.plot(t_s, seg_plot, color=trace_color, lw=0.9)
    ax_top.axvline(t_peak * 1e-3, color='black', lw=0.7, ls=':', alpha=0.5)
    if t_left  is not None and t_left  >= t0:
        ax_top.axvline(t_left  * 1e-3, color='blue', lw=1.5)
    if t_right is not None and t_right <= t1:
        ax_top.axvline(t_right * 1e-3, color='red',  lw=1.5)
    if D_hat is not None and not np.isnan(D_hat):
        ax_top.text(0.97, 0.93, f'D\u0302 = {D_hat:.3f} \u00b5m\u00b2/s',
                    transform=ax_top.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    ax_top.set_ylim(*(y_lim if y_lim else (0, 120)))
    ax_top.set_xlim(t0 * 1e-3, t1 * 1e-3)
    _spines(ax_top)
    ax_top.tick_params(axis='x', rotation=90)
    ax_top.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_top.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    _cat_str = f'  —  Cat-{category}' if category is not None else ''
    ax_top.set_title(f'{stem} / {ch_id}  —  t_peak = {t_peak * 1e-3:.3f} s{_cat_str}',
                     fontsize=FS_TITLE, color='black', pad=4)

    # ── Middle: factor*baseline-subtracted real trace ──────────────────────
    ax_mid.set_facecolor('white')
    ax_mid.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax_mid.plot(t_s, seg_bl_plot, color=trace_color, lw=0.9)
    ax_mid.axvline(t_peak * 1e-3, color='black', lw=0.7, ls=':', alpha=0.5)
    if t_left  is not None and t_left  >= t0:
        ax_mid.axvline(t_left  * 1e-3, color='blue', lw=1.5)
    if t_right is not None and t_right <= t1:
        ax_mid.axvline(t_right * 1e-3, color='red',  lw=1.5)
    if y_lim:
        ax_mid.set_ylim(*y_lim)
    else:
        ax_mid.set_ylim(bottom=0)
    ax_mid.set_xlim(t0 * 1e-3, t1 * 1e-3)
    _spines(ax_mid)
    ax_mid.tick_params(axis='x', rotation=90)
    ax_mid.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_mid.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    ax_mid.set_title(f'Baseline-subtracted  (×{factor:.2f})',
                     fontsize=FS_TITLE, color='black', pad=4)

    # ── Bottom: best-match sim trace ───────────────────────────────────────
    ax_bot.set_facecolor('white')
    ax_bot.plot(t_s_sim, sim_plot, color='steelblue', lw=0.9)
    ax_bot.axvline(0, color='black', lw=0.7, ls=':', alpha=0.5)
    box_ec  = '#FFD700' if is_best_pearson else 'none'
    box_lw  = 1.5       if is_best_pearson else 0
    ax_bot.text(0.97, 0.93, f'sim D = {D_sim:.4f} \u00b5m\u00b2/s',
                transform=ax_bot.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top', color='black',
                bbox=dict(facecolor='white', edgecolor=box_ec,
                          alpha=0.8, pad=2, linewidth=box_lw))
    r_color  = '#FFD700' if is_best_pearson else '#888888'
    r_weight = 'bold'    if is_best_pearson else 'normal'
    ax_bot.text(0.97, 0.80, f'r = {pearson_score:.3f}',
                transform=ax_bot.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top',
                color=r_color, fontweight=r_weight)
    if y_lim:
        ax_bot.set_ylim(*y_lim)
    else:
        ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlim(-1.0, 1.0)
    _spines(ax_bot)
    ax_bot.tick_params(axis='x', rotation=90)
    ax_bot.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_bot.set_xlabel('Time rel. to transit centre (s)', fontsize=FS_LABEL, color='black')
    ax_bot.set_title('Simulated transit (noise-free)', fontsize=FS_TITLE,
                     color='black', pad=4)

    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def read_events_csv(path):
    events = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ev = {
                't_left':        int(row['t_left_ms'])    if row.get('t_left_ms')        else None,
                't_peak':        int(row['t_peak_ms']),
                't_right':       int(row['t_right_ms'])   if row.get('t_right_ms')       else None,
                'D_hat':         float(row['D_hat'])       if row.get('D_hat')            else np.nan,
                'log10_D':       float(row['log10_D'])     if row.get('log10_D')          else np.nan,
                'category':      int(row['category'])      if row.get('category')         else None,
                'w12_seed':      int(row['w12_ms'])        if row.get('w12_ms')           else None,
                't_left_hist':    [int(v)   for v in row['t_left_history'].split(';')]
                                  if row.get('t_left_history')   else [],
                't_right_hist':   [int(v)   for v in row['t_right_history'].split(';')]
                                  if row.get('t_right_history')  else [],
                'd_hist':         [float(v) for v in row['D_hat_history'].split(';')]
                                  if row.get('D_hat_history')    else [],
                'baseline_w12':   float(row['baseline_w12_kHz'])
                                  if row.get('baseline_w12_kHz') else None,
                'baseline_hist':  [float(v) for v in row['baseline_history'].split(';')]
                                  if row.get('baseline_history') else [],
                'swallowed_by':   int(row['swallowed_by_ms'])
                                  if row.get('swallowed_by_ms')  else None,
                # v4.1/4.31: per-factor D predictions
                'd_hat_by_factor': {
                    1.05: float(row['d_hat_f105']) if row.get('d_hat_f105') else np.nan,
                    1.10: float(row['d_hat_f110']) if row.get('d_hat_f110') else np.nan,
                    1.15: float(row['d_hat_f115']) if row.get('d_hat_f115') else np.nan,
                    1.20: float(row['d_hat_f120']) if row.get('d_hat_f120') else np.nan,
                    1.25: float(row['d_hat_f125']) if row.get('d_hat_f125') else np.nan,
                    1.30: float(row['d_hat_f130']) if row.get('d_hat_f130') else np.nan,
                    1.35: float(row['d_hat_f135']) if row.get('d_hat_f135') else np.nan,
                    1.40: float(row['d_hat_f140']) if row.get('d_hat_f140') else np.nan,
                    1.45: float(row['d_hat_f145']) if row.get('d_hat_f145') else np.nan,
                    1.50: float(row['d_hat_f150']) if row.get('d_hat_f150') else np.nan,
                    1.60: float(row['d_hat_f160']) if row.get('d_hat_f160') else np.nan,
                    1.80: float(row['d_hat_f180']) if row.get('d_hat_f180') else np.nan,
                    2.00: float(row['d_hat_f200']) if row.get('d_hat_f200') else np.nan,
                },
                # v4.3: factor that drove the stage3 expansion loop
                'pearson_best_factor': float(row['pearson_best_factor'])
                                       if row.get('pearson_best_factor') else 1.0,
            }
            events.append(ev)
    return events


def _write_log(status, n_measurements=0, n_events=0, error=None):
    log = {
        'experiment':       DATA_DIR.name,
        'pipeline_version': PIPELINE_VERSION,
        'started_at':       STARTED_AT.isoformat(timespec='seconds'),
        'finished_at':      datetime.datetime.now().isoformat(timespec='seconds'),
        'status':           status,
        'n_measurements':   n_measurements,
        'n_events':         n_events,
    }
    if error is not None:
        log['error'] = error
    log_path = OUT_DIR / 'pipeline_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'  pipeline_log → {log_path}')


def _pearson_best_match(t_peak, i100, d_sim, i_sim, unique_d, log_unique, D_hat):
    """Return (chosen_row_idx, D_sim_val, pearson_score) for the best Pearson-correlated replica."""
    log_d_hat = np.log(D_hat)
    bin_idx   = int(np.argmin(np.abs(log_unique - log_d_hat)))
    D_sim_val = float(unique_d[bin_idx])

    same_d = np.where(d_sim == D_sim_val)[0]

    win_len  = WIN_E - WIN_S + 1           # 2001 bins
    real_seg = np.zeros(win_len, dtype=np.float32)
    r_t0 = max(0,              t_peak - EVT_HALF)
    r_t1 = min(len(i100) - 1, t_peak + EVT_HALF)
    pad  = max(0, EVT_HALF - t_peak)
    real_seg[pad : pad + (r_t1 - r_t0 + 1)] = i100[r_t0 : r_t1 + 1].astype(np.float32)
    r_max     = real_seg.max()
    real_norm = real_seg / r_max if r_max > 0 else real_seg

    sims      = i_sim[same_d, WIN_S : WIN_E + 1].astype(np.float32)
    s_max     = sims.max(axis=1, keepdims=True)
    s_max[s_max == 0] = 1.0
    sims_norm = sims / s_max

    r_c   = real_norm - real_norm.mean()
    s_c   = sims_norm - sims_norm.mean(axis=1, keepdims=True)
    num   = (s_c * r_c).sum(axis=1)
    denom = (np.sqrt((s_c ** 2).sum(axis=1)) *
             np.sqrt((r_c ** 2).sum())) + 1e-9
    corr     = num / denom
    best_idx = int(np.argmax(corr))
    chosen   = int(same_d[best_idx])
    return chosen, D_sim_val, float(corr[best_idx])


def _make_pearson_fn(i100, d_sim, i_sim, unique_d, log_unique):
    """
    Build a per-channel Pearson callback for the stage3 expansion loop.

    The returned closure has signature:
        pearson_fn(tp, factor_d_dict) -> (best_D_hat, best_factor, best_score)

    where factor_d_dict = {factor: D_hat_f} for all BL_FACTORS at the
    current expansion step.  The function picks the factor whose model
    prediction produces the highest Pearson correlation with the real trace
    window and returns that factor's D_hat_f as best_D_hat (used by stage3
    for tau_max lookup and D_FAST_CUTOFF stopping check).
    """
    def pearson_fn(tp, factor_d_dict):
        best_score  = -np.inf
        best_D_hat  = np.nan
        best_factor = 1.0
        for f, d_hat_f in factor_d_dict.items():
            if not np.isfinite(d_hat_f) or d_hat_f <= 0:
                continue
            _chosen, _D_sim_val, score = _pearson_best_match(
                tp, i100, d_sim, i_sim, unique_d, log_unique, d_hat_f)
            if score > best_score:
                best_score  = score
                best_D_hat  = d_hat_f   # use model prediction (not sim D) for stopping logic
                best_factor = f
        if not np.isfinite(best_D_hat):
            # All factors failed (e.g. D_hat <= 0) — fall back gracefully
            valid = [(f, v) for f, v in factor_d_dict.items()
                     if np.isfinite(v) and v > 0]
            if valid:
                best_factor, best_D_hat = valid[0]
            else:
                best_D_hat  = 0.1
                best_factor = 1.05
            best_score = 0.0
        return best_D_hat, best_factor, best_score

    return pearson_fn


try:
    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Load models
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('Phase 1 — Loading models')
    print('='*60)

    registry = ModelRegistry(
        noise_json     = str(PROJECT / 'noise_models.json'),
        diffusion_json = str(PROJECT / 'diffusion_models.json'),
        model_dir      = str(PROJECT),
    )
    registry.load_all()
    print(f'  {len(registry.diffusion_models._wrappers)} diffusion model(s)  '
          f'{len(registry.noise_models._wrappers)} noise model(s)')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Load simulation libraries (S1 = GFP, S2 = mCherry2)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('Phase 2 — Loading simulation libraries')
    print('='*60)

    d_sim_s1 = np.load(SIM_D_PATH_S1)
    i_sim_s1 = np.load(SIM_I_PATH_S1, mmap_mode='r')
    print(f'  [S1] d_sim {d_sim_s1.shape}  D ∈ [{d_sim_s1.min():.4f}, {d_sim_s1.max():.4f}] µm²/s')
    print(f'  [S1] i_sim {i_sim_s1.shape}  dtype {i_sim_s1.dtype}')
    unique_d_s1   = np.unique(d_sim_s1)
    log_unique_s1 = np.log(unique_d_s1)

    d_sim_s2 = np.load(SIM_D_PATH_S2)
    i_sim_s2 = np.load(SIM_I_PATH_S2, mmap_mode='r')
    print(f'  [S2] d_sim {d_sim_s2.shape}  D ∈ [{d_sim_s2.min():.4f}, {d_sim_s2.max():.4f}] µm²/s')
    print(f'  [S2] i_sim {i_sim_s2.shape}  dtype {i_sim_s2.dtype}')
    unique_d_s2   = np.unique(d_sim_s2)
    log_unique_s2 = np.log(unique_d_s2)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Per-measurement: pipeline → plots
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('Phase 3 — Pipeline + plots (per measurement)')
    print('='*60)

    all_meta_rows  = []
    n_measurements = 0
    seen_events    = set()   # (stem, ch_id, t_peak) — for accurate event count in log

    # v4.33: grand-summary accumulators (ch_id → {factor: count})
    all_best_factor_counts = {}
    all_knee_factor_counts = {}

    for stem in FILES:
        fcs_path = DATA_DIR / f'{stem}.fcs'
        print(f'\n{"─"*50}\n  {stem}\n{"─"*50}')

        try:
            meas = load_measurement(fcs_path=fcs_path, data_dir=DATA_DIR,
                                    registry=registry, raw_opts=RAW_OPTS)
        except Exception as e:
            print(f'  ERROR loading: {e}')
            import traceback; traceback.print_exc()
            continue

        # ── 3a. Build per-channel Pearson callbacks (v4.3) ────────────────────
        pearson_fns = {}
        for ch_id in meas.channel_ids:
            ch_data = meas.channels.get(ch_id)
            if ch_data is None or ch_data.i100_full is None:
                continue
            i100_ch = ch_data.i100_full
            if ch_id == 'S2':
                pearson_fns[ch_id] = _make_pearson_fn(
                    i100_ch, d_sim_s2, i_sim_s2, unique_d_s2, log_unique_s2)
            else:
                pearson_fns[ch_id] = _make_pearson_fn(
                    i100_ch, d_sim_s1, i_sim_s1, unique_d_s1, log_unique_s1)
        print(f'  Pearson callbacks built for channels: {list(pearson_fns)}')

        # ── 3b. Run pipeline ──────────────────────────────────────────────────
        try:
            results = run_pipeline(meas=meas, registry=registry, out_dir=OUT_DIR,
                                   save_csv=True, save_plots=False,
                                   stage3_module='stage3_v4_3',
                                   pearson_fns=pearson_fns)
            for res in results:
                print(f'  [{res.channel_id}]  {res.n_events} events  '
                      f'(model: {res.d_model_label})')
        except Exception as e:
            print(f'  ERROR in pipeline: {e}')
            import traceback; traceback.print_exc()
            continue

        n_measurements += 1

        # ── 3c. Per-channel: plots + sim extraction ───────────────────────────
        for ch_id in meas.channel_ids:
            csv_path = OUT_DIR / f'{stem}_{ch_id}_events.csv'
            if not csv_path.exists():
                print(f'  [{ch_id}] no CSV — skipping')
                continue

            events   = read_events_csv(csv_path)
            i100     = meas.channels[ch_id].i100_full
            baseline = _stage1_baseline(i100)

            # Choose sim library for this channel
            if ch_id == 'S2':
                d_sim     = d_sim_s2
                i_sim     = i_sim_s2
                unique_d  = unique_d_s2
                log_unique = log_unique_s2
            else:
                d_sim     = d_sim_s1
                i_sim     = i_sim_s1
                unique_d  = unique_d_s1
                log_unique = log_unique_s1

            make_trace_plot(stem, ch_id, i100, baseline, events,
                            OUT_DIR / f'{stem}_{ch_id}_trace.png')
            make_d_histogram(stem, ch_id, events,
                             OUT_DIR / f'{stem}_{ch_id}_d_hist.png')

            # v4.33: per-measurement frequency counters (reset each channel)
            best_factor_counts = {f: 0 for f in BL_FACTORS}
            knee_factor_counts = {f: 0 for f in BL_FACTORS}

            # v4.35: D_hat + window-size accumulators for comparison plots
            d_pearson_list  = []
            d_knee_list     = []
            ws_pearson_list = []   # window size (ms) for Pearson-opt events
            ws_knee_list    = []   # window size (ms) for Knee-opt events

            n_plotted = 0
            for ev in events:
                t_peak = ev['t_peak']

                # Pass 1: compute Pearson for all valid factors (post-hoc, for plotting)
                factor_results = {}  # factor → (chosen, D_sim_val, pearson, sim_raw)
                for factor in BL_FACTORS:
                    D_hat_f = ev['d_hat_by_factor'].get(factor, np.nan)
                    if np.isnan(D_hat_f) or D_hat_f <= 0:
                        continue
                    chosen, D_sim_val, pearson = _pearson_best_match(
                        t_peak, i100, d_sim, i_sim, unique_d, log_unique, D_hat_f)
                    sim_raw = i_sim[chosen, WIN_S : WIN_E + 1].astype(np.float32)
                    factor_results[factor] = (chosen, D_sim_val, pearson, sim_raw)

                if not factor_results:
                    continue

                # Best factor by post-hoc Pearson score (for plot highlighting)
                best_factor = max(factor_results, key=lambda f: factor_results[f][2])
                best_factor_counts[best_factor] += 1

                # Stage3 Pearson-guided factor (from expansion loop, recorded in CSV)
                stage3_best_factor = ev.get('pearson_best_factor', 1.0)

                # Photon counts for all 13 factors (independent of D_hat validity)
                N_tr   = len(i100)
                tl_eff = max(0, min(ev['t_left'],  N_tr - 1)) if ev['t_left']  is not None else 0
                tr_eff = max(0, min(ev['t_right'], N_tr - 1)) if ev['t_right'] is not None else N_tr - 1
                edge_bl          = float(min(baseline[tl_eff], baseline[tr_eff]))
                seg_ev           = i100[tl_eff : tr_eff + 1]
                raw_photon_count = float(seg_ev.sum())
                phot_by_factor   = {
                    f: float(np.maximum(0.0, seg_ev - f * edge_bl).sum())
                    for f in BL_FACTORS
                }

                # v4.33: knee detection
                knee_factor = _detect_knee(BL_FACTORS,
                                           [phot_by_factor[f] for f in BL_FACTORS])
                knee_factor_counts[knee_factor] += 1

                # v4.35: accumulate D_hat + window size for comparison plots
                win_ms = int(tr_eff - tl_eff + 1)
                d_pearson_list.append(ev['d_hat_by_factor'][best_factor])
                ws_pearson_list.append(win_ms)
                if knee_factor in factor_results:
                    d_knee_list.append(ev['d_hat_by_factor'][knee_factor])
                    ws_knee_list.append(win_ms)

                # Pass 2: save sim traces + make plots
                for factor, (chosen, D_sim_val, pearson, sim_raw) in factor_results.items():
                    is_best  = (factor == best_factor)
                    is_knee  = (factor == knee_factor)
                    fac_tag  = f'f{int(round(factor * 100)):03d}'
                    npy_path = SIM_DIR / f'{stem}_{ch_id}_{t_peak:07d}ms_sim_{fac_tag}.npy'
                    np.save(npy_path, sim_raw)

                    sim_trace_khz = sim_raw * 1e-3
                    out_path = EVT_DIR / f'{stem}_{ch_id}_event_{t_peak:07d}ms_{fac_tag}.png'
                    make_event_plot_3panel(
                        stem, ch_id, t_peak, i100, baseline,
                        ev['t_left'], ev['t_right'], ev['d_hat_by_factor'][factor],
                        sim_trace_khz, D_sim_val, out_path,
                        category=ev.get('category'), factor=factor,
                        pearson_score=pearson, is_best_pearson=is_best)

                    if is_best:
                        shutil.copy2(out_path, OPT_DIR      / f'g_{out_path.name}')
                    if is_knee:
                        shutil.copy2(out_path, KNEE_OPT_DIR / f'g_{out_path.name}')

                    all_meta_rows.append({
                        'stem':                  stem,
                        'ch_id':                 ch_id,
                        't_peak_ms':             t_peak,
                        'factor':                factor,
                        'D_hat':                 ev['d_hat_by_factor'][factor],
                        'D_sim':                 D_sim_val,
                        'sim_idx':               chosen,
                        'pearson':               pearson,
                        'best_pearson':          is_best,
                        'pearson_best_factor_stage3': stage3_best_factor,
                        'raw_photon_count':      raw_photon_count,
                        'photon_count':          phot_by_factor[factor],
                        'knee_factor':           knee_factor,      # v4.33
                    })
                    seen_events.add((stem, ch_id, t_peak))
                    n_plotted += 1

                # Photon count vs factor plot with knee marker
                phot_path = PHOT_DIR / f'{stem}_{ch_id}_event_{t_peak:07d}ms_phot.png'
                make_phot_plot(stem, ch_id, t_peak, phot_by_factor, raw_photon_count,
                               phot_path, knee_factor=knee_factor)

            # v4.35: Pearson vs Knee comparison plots
            if n_plotted > 0:
                make_comparison_d_histogram(
                    stem, ch_id, d_pearson_list, d_knee_list,
                    OUT_DIR / f'{stem}_{ch_id}_d_hist_compare.png')
                make_scatter_window_d(
                    stem, ch_id,
                    ws_pearson_list, d_pearson_list,
                    ws_knee_list,    d_knee_list,
                    OUT_DIR / f'{stem}_{ch_id}_scatter_window_d.png')

            # v4.33: per-measurement frequency bar charts
            if n_plotted > 0:
                make_factor_freq_plot(
                    f'{stem} / {ch_id}  —  Pearson best-match frequency',
                    best_factor_counts,
                    OPT_DIR  / f'{stem}_{ch_id}_best_factor_freq.png')
                make_factor_freq_plot(
                    f'{stem} / {ch_id}  —  Knee factor frequency',
                    knee_factor_counts,
                    PHOT_DIR / f'{stem}_{ch_id}_knee_factor_freq.png')

                # Accumulate into grand-summary dicts
                if ch_id not in all_best_factor_counts:
                    all_best_factor_counts[ch_id] = {f: 0 for f in BL_FACTORS}
                if ch_id not in all_knee_factor_counts:
                    all_knee_factor_counts[ch_id] = {f: 0 for f in BL_FACTORS}
                for f in BL_FACTORS:
                    all_best_factor_counts[ch_id][f] += best_factor_counts[f]
                    all_knee_factor_counts[ch_id][f]  += knee_factor_counts[f]

            print(f'  [{ch_id}]  {n_plotted} event plots saved  → event_plots/')

    # v4.33: grand-summary frequency bar charts (one per channel across all measurements)
    for ch_id, counts in all_best_factor_counts.items():
        make_factor_freq_plot(
            f'All measurements / {ch_id}  —  Pearson best-match frequency',
            counts,
            OPT_DIR  / f'all_{ch_id}_best_factor_freq.png')
    for ch_id, counts in all_knee_factor_counts.items():
        make_factor_freq_plot(
            f'All measurements / {ch_id}  —  Knee factor frequency',
            counts,
            PHOT_DIR / f'all_{ch_id}_knee_factor_freq.png')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Write metadata.csv
    # ══════════════════════════════════════════════════════════════════════════
    meta_path = SIM_DIR / 'metadata.csv'
    with open(meta_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['stem', 'ch_id', 't_peak_ms', 'factor',
                           'D_hat', 'D_sim', 'sim_idx', 'pearson', 'best_pearson',
                           'pearson_best_factor_stage3',
                           'raw_photon_count', 'photon_count', 'knee_factor'])
        writer.writeheader()
        writer.writerows(all_meta_rows)

    print(f'\n{"="*60}')
    print('Done.')
    print(f'  {len(all_meta_rows)} event×factor plots processed')
    print(f'  CSVs + plots   → {OUT_DIR}')
    print(f'  Event plots    → {EVT_DIR}')
    print(f'  Sim traces     → {SIM_DIR}')
    print(f'  Sim metadata   → {meta_path}')
    print(f'{"="*60}')

    _write_log('completed', n_measurements=n_measurements, n_events=len(seen_events))

except Exception as _top_exc:
    import traceback
    traceback.print_exc()
    _write_log('failed', error=str(_top_exc))
    sys.exit(1)
