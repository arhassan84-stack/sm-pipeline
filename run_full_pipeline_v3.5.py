#!/usr/bin/env python3
"""
run_full_pipeline_v3.5.py
Single-pass integrated pipeline + all figures for cluster execution.

For each measurement (one loop):
  1. run_pipeline                → {stem}_{ch}_events.csv
  2. make_trace_plot             → {stem}_{ch}_trace.png
  3. make_d_histogram            → {stem}_{ch}_d_hist.png
  4. extract sim trace per event → sim_traces/{stem}_{ch}_{t}ms_sim.npy
  5. make 2-panel event plot     → event_plots/{stem}_{ch}_event_{t}ms.png

After the loop:
  6. Write sim_traces/metadata.csv
"""

import argparse
import csv
import datetime
import json
import logging
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch   # (unused but harmless)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s',
                    stream=sys.stdout)

PIPELINE_VERSION = 'v3.5'   # hardcoded — do not import from version.py

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
for d in [OUT_DIR, SIM_DIR, EVT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STARTED_AT = datetime.datetime.now()

# Simulation library (noise-free, dt=1ms)
SIM_D_PATH = PROJECT / 'sims_multidt_noise0_d.npy'
SIM_I_PATH = PROJECT / 'sims_multidt_noise0_i_dt100.npy'

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
    """log₁₀D histogram with coloured bars."""
    log10_D = np.array([ev['log10_D'] for ev in events
                        if not np.isnan(ev['log10_D'])])
    if len(log10_D) == 0:
        return

    bins = np.arange(-2.5, 1.5 + 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.set_facecolor('white')

    for left, right in zip(bins[:-1], bins[1:]):
        count = int(((log10_D >= left) & (log10_D < right)).sum())
        if count > 0:
            ax.bar(left, count, width=right - left, align='edge',
                   color=CMAP_D(NORM_D(0.5 * (left + right))),
                   edgecolor='white', linewidth=0.5)

    med   = float(np.median(log10_D))
    d_med = 10.0 ** med
    ax.axvline(med, color='black', lw=1.5, ls='--',
               label=f'median = {d_med:.3f} µm²/s')
    ax.legend(fontsize=FS_TICK)
    _spines(ax)
    ax.set_xlabel('log₁₀ D̂  (µm²/s)', fontsize=FS_LABEL, color='black')
    ax.set_ylabel('Events',            fontsize=FS_LABEL, color='black')
    ax.set_title(f'{stem} / {ch_id}  —  D̂ distribution  (n={len(log10_D)})',
                 fontsize=FS_TITLE, color='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'    d_hist → {out_path.name}')


def make_event_plot_3panel(stem, ch_id, t_peak, i100, baseline,
                           t_left, t_right, D_hat,
                           sim_trace, D_sim, out_path, category=None):
    """3-panel event figure: raw (top) + baseline-subtracted (middle) + sim (bottom).
    S1 → normalised [0,1];  S2 → raw kHz."""
    N  = len(i100)
    t0 = max(0,     t_peak - EVT_HALF)
    t1 = min(N - 1, t_peak + EVT_HALF)

    seg    = i100[t0 : t1 + 1]
    bl     = baseline[t0 : t1 + 1]
    seg_bl = np.maximum(0.0, seg - bl)          # baseline-subtracted, clipped
    t_s    = np.arange(t0, t0 + len(seg)) * 1e-3

    n_sim   = len(sim_trace)
    t_s_sim = (np.arange(n_sim) - (n_sim // 2)) * 1e-3

    normalize   = (ch_id == 'S1')
    trace_color = '#007700' if ch_id == 'S1' else '#CC0000'

    if normalize:
        seg_max    = seg.max()       if seg.max()       > 0 else 1.0
        seg_bl_max = seg_bl.max()    if seg_bl.max()    > 0 else 1.0
        sim_max    = sim_trace.max() if sim_trace.max() > 0 else 1.0
        seg_plot   = seg    / seg_max
        bl_plot    = bl     / seg_max
        seg_bl_plot= seg_bl / seg_bl_max
        sim_plot   = sim_trace / sim_max
        y_label    = 'Norm. intensity'
        y_lim      = (0, 1.15)
    else:
        seg_plot    = seg
        bl_plot     = bl
        seg_bl_plot = seg_bl
        sim_plot    = sim_trace
        y_label     = 'kHz'
        y_lim       = None

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(6, 7.5), facecolor='white',
        gridspec_kw={'hspace': 0.55})

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
    ax_top.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_top.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    _cat_str = f'  —  Cat-{category}' if category is not None else ''
    ax_top.set_title(f'{stem} / {ch_id}  —  t_peak = {t_peak * 1e-3:.3f} s{_cat_str}',
                     fontsize=FS_TITLE, color='black', pad=4)

    # ── Middle: baseline-subtracted real trace ─────────────────────────────
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
    ax_mid.set_ylabel(y_label, fontsize=FS_LABEL, color='black')
    ax_mid.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')
    ax_mid.set_title('Baseline-subtracted', fontsize=FS_TITLE, color='black', pad=4)

    # ── Bottom: best-match sim trace ───────────────────────────────────────
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
    _spines(ax_bot)
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
            events.append({
                't_left':        int(row['t_left_ms'])    if row.get('t_left_ms')        else None,
                't_peak':        int(row['t_peak_ms']),
                't_right':       int(row['t_right_ms'])   if row.get('t_right_ms')       else None,
                'D_hat':         float(row['D_hat'])       if row.get('D_hat')            else np.nan,
                'log10_D':       float(row['log10_D'])     if row.get('log10_D')          else np.nan,
                'category':      int(row['category'])      if row.get('category')         else None,
                'w12_seed':      int(row['w12_ms'])        if row.get('w12_ms')           else None,
                't_left_hist':   [int(v)   for v in row['t_left_history'].split(';')]
                                 if row.get('t_left_history')  else [],
                't_right_hist':  [int(v)   for v in row['t_right_history'].split(';')]
                                 if row.get('t_right_history') else [],
                'd_hist':        [float(v) for v in row['D_hat_history'].split(';')]
                                 if row.get('D_hat_history')   else [],
            })
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
    # Phase 2 — Load simulation library once
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('Phase 2 — Loading simulation library')
    print('='*60)

    d_sim    = np.load(SIM_D_PATH)
    i_sim    = np.load(SIM_I_PATH, mmap_mode='r')
    print(f'  d_sim shape {d_sim.shape}  D ∈ [{d_sim.min():.4f}, {d_sim.max():.4f}] µm²/s')
    print(f'  i_sim shape {i_sim.shape}  dtype {i_sim.dtype}')

    unique_d   = np.unique(d_sim)
    log_unique = np.log(unique_d)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Per-measurement: pipeline → plots
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('Phase 3 — Pipeline + plots (per measurement)')
    print('='*60)

    all_meta_rows  = []
    n_measurements = 0

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

        # ── 3a. Run pipeline ──────────────────────────────────────────────────
        try:
            results = run_pipeline(meas=meas, registry=registry, out_dir=OUT_DIR,
                                   save_csv=True, save_plots=False,
                                   stage3_module='stage3_v3_5')
            for res in results:
                print(f'  [{res.channel_id}]  {res.n_events} events  '
                      f'(model: {res.d_model_label})')
        except Exception as e:
            print(f'  ERROR in pipeline: {e}')
            import traceback; traceback.print_exc()
            continue

        n_measurements += 1

        # ── 3b. Per-channel: plots + sim extraction ───────────────────────────
        for ch_id in meas.channel_ids:
            csv_path = OUT_DIR / f'{stem}_{ch_id}_events.csv'
            if not csv_path.exists():
                print(f'  [{ch_id}] no CSV — skipping')
                continue

            events   = read_events_csv(csv_path)
            i100     = meas.channels[ch_id].i100_full
            baseline = _stage1_baseline(i100)

            make_trace_plot(stem, ch_id, i100, baseline, events,
                            OUT_DIR / f'{stem}_{ch_id}_trace.png')
            make_d_histogram(stem, ch_id, events,
                             OUT_DIR / f'{stem}_{ch_id}_d_hist.png')

            n_plotted = 0
            for ev in events:
                t_peak = ev['t_peak']
                D_hat  = ev['D_hat']
                if np.isnan(D_hat) or D_hat <= 0:
                    continue

                log_d_hat = np.log(D_hat)
                bin_idx   = int(np.argmin(np.abs(log_unique - log_d_hat)))
                D_sim_val = float(unique_d[bin_idx])

                same_d = np.where(d_sim == D_sim_val)[0]

                # ── Pick best-matching replica by Pearson correlation ──────
                win_len  = WIN_E - WIN_S + 1           # 2001 bins
                real_seg = np.zeros(win_len, dtype=np.float32)
                r_t0 = max(0,     t_peak - EVT_HALF)
                r_t1 = min(len(i100) - 1, t_peak + EVT_HALF)
                pad  = max(0, EVT_HALF - t_peak)        # leading zeros if near trace start
                real_seg[pad : pad + (r_t1 - r_t0 + 1)] = \
                    i100[r_t0 : r_t1 + 1].astype(np.float32)
                r_max = real_seg.max()
                real_norm = real_seg / r_max if r_max > 0 else real_seg

                sims   = i_sim[same_d, WIN_S : WIN_E + 1].astype(np.float32)
                s_max  = sims.max(axis=1, keepdims=True)
                s_max[s_max == 0] = 1.0
                sims_norm = sims / s_max

                r_c   = real_norm - real_norm.mean()
                s_c   = sims_norm - sims_norm.mean(axis=1, keepdims=True)
                num   = (s_c * r_c).sum(axis=1)
                denom = (np.sqrt((s_c ** 2).sum(axis=1)) *
                         np.sqrt((r_c ** 2).sum())) + 1e-9
                corr  = num / denom
                chosen = int(same_d[np.argmax(corr)])
                # ─────────────────────────────────────────────────────────

                sim_raw = i_sim[chosen, WIN_S : WIN_E + 1].astype(np.float32)
                npy_path = SIM_DIR / f'{stem}_{ch_id}_{t_peak:07d}ms_sim.npy'
                np.save(npy_path, sim_raw)

                sim_trace_khz = sim_raw * 1e-3
                out_path = EVT_DIR / f'{stem}_{ch_id}_event_{t_peak:07d}ms.png'
                make_event_plot_3panel(
                    stem, ch_id, t_peak, i100, baseline,
                    ev['t_left'], ev['t_right'], D_hat,
                    sim_trace_khz, D_sim_val, out_path,
                    category=ev.get('category'))

                all_meta_rows.append({
                    'stem':      stem,
                    'ch_id':     ch_id,
                    't_peak_ms': t_peak,
                    'D_hat':     D_hat,
                    'D_sim':     D_sim_val,
                    'sim_idx':   chosen,
                })
                n_plotted += 1

            print(f'  [{ch_id}]  {n_plotted} event plots saved  → event_plots/')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Write metadata.csv
    # ══════════════════════════════════════════════════════════════════════════
    meta_path = SIM_DIR / 'metadata.csv'
    with open(meta_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['stem', 'ch_id', 't_peak_ms', 'D_hat', 'D_sim', 'sim_idx'])
        writer.writeheader()
        writer.writerows(all_meta_rows)

    print(f'\n{"="*60}')
    print('Done.')
    print(f'  {len(all_meta_rows)} events processed')
    print(f'  CSVs + plots   → {OUT_DIR}')
    print(f'  Event plots    → {EVT_DIR}')
    print(f'  Sim traces     → {SIM_DIR}')
    print(f'  Sim metadata   → {meta_path}')
    print(f'{"="*60}')

    _write_log('completed', n_measurements=n_measurements, n_events=len(all_meta_rows))

except Exception as _top_exc:
    import traceback
    traceback.print_exc()
    _write_log('failed', error=str(_top_exc))
    sys.exit(1)
