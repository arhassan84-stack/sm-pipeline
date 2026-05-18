"""
pipeline.py — FCS Single-Molecule Inference Pipeline
═════════════════════════════════════════════════════
Orchestrates Stages 1–6 of algorithm_sm_inference.md.
Stages 0 and −1 are handled by measurement.py and model_registry.py.

Quickstart
----------
    from model_registry import ModelRegistry
    from measurement    import load_measurement
    from pipeline       import run_pipeline

    registry = ModelRegistry('noise_models.json', 'diffusion_models.json',
                              model_dir='.')
    registry.load_all()

    meas = load_measurement('sample.fcs', data_dir='.', registry=registry,
                             raw_opts=dict(bits=32, n_header=32,
                                           clock_rate_hz=15_000_000))

    results = run_pipeline(meas, registry, out_dir='.')

Pipeline stages
---------------
Stage 1  — Sliding-window noise estimation → n_abs (kHz)
Stage 2  — Background subtraction → cleaned channels
Stage 3  — Matched-filter event detection (stage3.run_stage3)
Stage 4  — Per-segment D-prediction with conformal intervals
Stage 5  — Per-event quality filter
Stage 6  — Output: CSV + annotated plots
"""

# Allow 'X | Y' union-type syntax in annotations on Python < 3.10
from __future__ import annotations

import logging                          # standard library: structured log messages at runtime
import math                             # standard library: floor/ceil/log etc. (available if needed)
from dataclasses import dataclass, field  # @dataclass auto-generates __init__/__repr__; field() sets per-attribute defaults
from pathlib import Path                # object-oriented filesystem paths (cross-platform)
from typing import Any, Dict, List, Optional  # type-hint generics; Optional[X] = X | None

import numpy as np                      # numerical arrays — the core data structure throughout the pipeline

from measurement              import Measurement    # Stage 0 output: fully pre-processed FCS acquisition
from model_registry           import ModelRegistry, ModelWrapper  # catalogue of loaded torch models
import importlib                        # dynamic module loading — lets us swap stage3 versions at runtime
from compute_features_multidt import compute_features_for_trace  # computes ACF-based feature vector for WaveNet

log = logging.getLogger(__name__)       # module-level logger; __name__ = 'pipeline' so messages are prefixed with that


# ── Result container ─────────────────────────────────────────────────────────

@dataclass                              # decorator: auto-generates __init__, __repr__, __eq__ from the field annotations below
class PipelineResult:
    """
    Output of run_pipeline for one Measurement / channel.

    Attributes
    ----------
    name          : measurement name
    channel_id    : detector channel label
    events        : list of per-event dicts (Stage 5 survivors)
    rejected      : list of per-event dicts that were discarded by Stage 5
    n_abs         : global background estimate (kHz); None if unavailable
    d_model_label : label of the D model used
    noise_model_label : label of the noise model used; None if skipped
    run_id        : optional experiment identifier
    """
    name:               str             # FCS file stem, e.g. 'lateralDorsal_nt_1'
    channel_id:         str             # detector label, e.g. 'S1' or 'S2'
    events:             List[dict]   = field(default_factory=list)   # field() with default_factory avoids the mutable-default-argument pitfall; each instance gets its own fresh list
    rejected:           List[dict]   = field(default_factory=list)   # events discarded by stage 5; separate list so they can still be inspected
    n_abs:              Optional[float] = None    # global noise floor in kHz; None when no noise model ran
    d_model_label:      Optional[str]   = None    # human-readable label of whichever D model was selected
    noise_model_label:  Optional[str]   = None    # human-readable label of the noise model; None if skipped
    run_id:             Optional[str]   = None    # optional experiment tag appended to output filenames

    @property                           # @property turns n_events into a read-only attribute computed on access
    def n_events(self) -> int:          # return type annotation: int
        return len(self.events)         # dynamically counts how many events survived stage 5


# ── Main entry point ─────────────────────────────────────────────────────────

# Module-level constant: default amplitude and fold thresholds per channel.
# Dict[str, dict] maps channel label → kwargs dict passed to stage3's peak finder.
CHANNEL_THRESHOLDS_DEFAULT: Dict[str, dict] = {
    'S1': dict(amplitude_min_khz=50.0, fold_min=3.0),   # GFP channel: require ≥50 kHz above baseline and ≥3× fold
    'S2': dict(amplitude_min_khz=30.0, fold_min=2.0),   # mCherry2 channel: dimmer, so lower thresholds
}


def run_pipeline(
    meas:               Measurement,             # fully pre-processed acquisition from load_measurement()
    registry:           ModelRegistry,           # loaded model catalogue
    channel_id:         Optional[str] = None,    # None → process all channels in meas.channel_ids
    out_dir:            str | Path = '.',        # destination directory for CSV and PNG outputs
    run_id:             Optional[str] = None,    # optional prefix/tag appended to every output filename
    save_csv:           bool = True,             # whether to write the per-event CSV in stage 6
    save_plots:         bool = True,             # whether to generate the three diagnostic plots in stage 6
    channel_thresholds: Optional[Dict[str, dict]] = None,  # caller-supplied overrides for amplitude_min_khz / fold_min per channel
    stage3_module:      str = 'stage3_v2',       # name of the Python module implementing run_stage3_v2 and TraceBundle
    pearson_fns:        Optional[Dict[str, object]] = None,  # v4.3: per-channel Pearson callback; signature: (tp, {factor: D_hat}) → (best_D_hat, best_factor, best_score)
) -> List[PipelineResult]:             # returns one PipelineResult per processed channel
    """
    Run the full inference pipeline (Stages 1–6) on one Measurement.

    Parameters
    ----------
    meas               : populated Measurement (from measurement.load_measurement)
    registry           : ModelRegistry (from model_registry.ModelRegistry.load_all)
    channel_id         : which channel to process; defaults to all channels
    out_dir            : directory for CSV and PNG outputs
    run_id             : optional label appended to output filenames
    save_csv           : write events CSV (Stage 6)
    save_plots         : generate annotated plots (Stage 6)
    channel_thresholds : per-channel peak filter settings, e.g.
                         {'S1': dict(amplitude_min_khz=50, fold_min=3),
                          'S2': dict(amplitude_min_khz=30, fold_min=2)}.
                         Falls back to CHANNEL_THRESHOLDS_DEFAULT for any
                         channel not explicitly listed.

    Returns
    -------
    list of PipelineResult — one per processed channel
    """
    out_dir  = Path(out_dir)            # coerce string → Path so all subsequent path operations use / operator
    out_dir.mkdir(parents=True, exist_ok=True)  # create output directory and any missing parents; no error if it already exists

    thresholds = dict(CHANNEL_THRESHOLDS_DEFAULT)  # shallow copy of the module constant so we don't mutate it
    if channel_thresholds:              # if the caller supplied overrides (non-None, non-empty)
        thresholds.update(channel_thresholds)  # merge caller overrides; keys present in both → caller wins

    _s3 = importlib.import_module(stage3_module)  # dynamically import e.g. 'stage3_v4_1' at runtime; allows version switching without code changes

    channels = [channel_id] if channel_id else meas.channel_ids  # if a specific channel was requested wrap it in a list; otherwise use all channels from the measurement
    results  = []                       # accumulator: will hold one PipelineResult per channel

    for ch_id in channels:              # iterate over each channel label, e.g. ['S1', 'S2']
        log.info(f'Processing {meas.name} / channel {ch_id}')  # f-string log message; shows which measurement+channel is being processed
        ch_data = meas.channels[ch_id]  # look up the ChannelData object for this channel from the measurement's dict

        if ch_data.i100_full is None or len(ch_data.i100_full) == 0:  # guard: skip channels with no 1ms intensity trace (can happen when has_raw was downgraded)
            log.warning(f'  {ch_id}: no i100_full — skipping')  # log at WARNING level so it's visible even in production runs
            continue                    # skip to the next channel; 'continue' jumps back to the top of the for loop

        ch_thresh  = thresholds.get(ch_id, {})  # look up this channel's threshold dict; {} means no amplitude/fold filtering if channel not in thresholds
        pearson_fn = pearson_fns.get(ch_id) if pearson_fns else None  # per-channel Pearson callback; None when pearson_fns not supplied (v4.1 behaviour)
        res = _run_channel(meas, ch_id, ch_data, registry,   # delegate all stage 1–6 logic to the per-channel helper
                           out_dir, run_id, save_csv, save_plots,
                           _run_stage3=_s3.run_stage3_v2,    # pass the stage3 function as a callable so _run_channel doesn't import it directly
                           _TraceBundle=_s3.TraceBundle,     # pass the TraceBundle class from the same module so it's consistent
                           pearson_fn=pearson_fn,            # v4.3: thread Pearson callback into stage3 via _run_channel
                           **ch_thresh)                      # unpack e.g. {'amplitude_min_khz': 50.0, 'fold_min': 3.0} as keyword args
        results.append(res)             # add this channel's result to the accumulator

    return results                      # return list of PipelineResult, one per successfully processed channel


# ── Per-channel pipeline ─────────────────────────────────────────────────────

def _run_channel(
    meas:               Measurement,        # full measurement object (needed for metadata like total_duration_ms)
    ch_id:              str,                # channel label, e.g. 'S1'
    ch_data,                               # ChannelData for this channel (i100_full, i050_full, i_min_full)
    registry:           ModelRegistry,     # model catalogue for selecting best D model
    out_dir:            Path,              # output directory for CSV and plots
    run_id:             Optional[str],     # optional filename prefix
    save_csv:           bool,              # whether stage 6 writes the CSV
    save_plots:         bool,              # whether stage 6 generates the three plots
    amplitude_min_khz:  Optional[float] = None,  # minimum peak amplitude above baseline (kHz); None = use fractional threshold instead
    fold_min:           Optional[float] = None,   # minimum fold-over-baseline ratio; None = no fold filter
    _run_stage3=None,                      # injected stage3 entry-point function; defaults to stage3_v2 if None
    _TraceBundle=None,                     # injected TraceBundle class; must match _run_stage3's module
    pearson_fn=None,                       # v4.3: optional Pearson callback; signature (tp, {factor: D_hat}) → (best_D_hat, best_factor, best_score)
) -> PipelineResult:                       # returns the populated result for this channel

    result = PipelineResult(               # initialise result container; fields default to [] / None
        name       = meas.name,           # copy measurement name into result for downstream reference
        channel_id = ch_id,               # record which channel this result belongs to
        run_id     = run_id,              # carry the optional run tag through
    )

    # ── Stage 1: Sliding-window baseline estimation ───────────────────────────
    log.info('[Stage 1] Baseline estimation')   # progress marker in the log
    baseline = _stage1_baseline(ch_data.i100_full)  # compute per-bin (1ms) background level from the 1ms trace; returns (N,) float64 array in kHz

    # ── Stage 2: Background subtraction ──────────────────────────────────────
    log.info('[Stage 2] Background subtraction')  # progress marker
    i_min_c, i050_c, i100_c = _stage2_subtract(ch_data, baseline, meas)  # subtract the baseline from all three dt channels; returns cleaned (non-negative) kHz arrays

    # Resolve stage3 callables (injected by run_pipeline; fall back to defaults)
    if _run_stage3 is None or _TraceBundle is None:  # safety fallback: if either callable is missing (e.g. direct call not through run_pipeline)
        _s3 = importlib.import_module('stage3_v2')   # import the oldest/default stage3 version
        _run_stage3  = _s3.run_stage3_v2             # use its entry-point function
        _TraceBundle = _s3.TraceBundle               # use its TraceBundle class

    # ── Stage 3: D-guided event detection ────────────────────────────────────
    log.info('[Stage 3] Event detection')   # progress marker
    d_model = registry.diffusion_models.best_for(   # query the model catalogue for the best available D model
        trace_length_ms = meas.total_duration_ms,   # some models require a minimum trace length
        has_raw         = meas.has_raw,             # sub-ms channels only available when RAW files were loaded
        channel_id      = ch_id,                   # models may be channel-specific (e.g. wavenet_multidt_S2 for S2)
    )
    if d_model is None:                 # no model satisfies the constraints (e.g. trace too short, no RAW)
        log.warning('  No admissible D model — aborting')  # warn and bail out; result will have empty events list
        return result                   # early return with empty PipelineResult

    result.d_model_label = d_model.label  # record which model was selected so it appears in the CSV and log

    # ── Stage 3: D-guided event detection (stage3_v2) ─────────────────────────
    # TraceBundle: raw kHz traces — stage3._build_traces applies the single
    # baseline subtraction internally for all dt channels.  Passing the
    # pre-subtracted i050_c / i_min_c here caused double subtraction for
    # the 0.5ms and 0.1ms branches, making peaks look artificially narrow.
    bundle = _TraceBundle(              # construct the TraceBundle from the version-matched class
        i1ms     = ch_data.i100_full,  # raw (not baseline-subtracted) 1ms kHz trace; stage3 subtracts internally
        baseline = baseline,            # per-bin 1ms baseline array so stage3 can do event-local subtraction
        i_dt050  = ch_data.i050_full,  # raw 0.5ms kHz trace; None if has_raw=False
        i_dt010  = ch_data.i_min_full, # raw 0.1ms kHz trace; None if has_raw=False
    )

    events_raw, z_score = _run_stage3(bundle, d_model,          # run the full detection pipeline; returns list of event dicts and z-score array
                                        amplitude_min_khz=amplitude_min_khz,  # forwarded to stage3's peak finder
                                        fold_min=fold_min,                    # forwarded to stage3's peak finder
                                        pearson_fn=pearson_fn)                # v4.3: Pearson callback for expansion stopping (None → v4.1 behaviour)

    if not events_raw:                  # empty list → no events were detected in this channel
        log.info('  No events detected')  # informational; not a warning because zero events is a valid outcome
        return result                   # return early with empty events list

    log.info(f'  {len(events_raw)} raw events detected')  # count before stage 5 quality filter

    # ── Stage 4: Conformal intervals ──────────────────────────────────────────
    # D_hat is already predicted for each event inside run_stage3_v2.
    # Stage 4 here only adds conformal uncertainty intervals.
    log.info('[Stage 4] Conformal intervals')   # progress marker
    events_pred = _stage4_conformal(events_raw, d_model)  # augments each event dict with D_low, D_high, flags; operates in-place but also returns the list

    # ── Stage 5: Quality filter ───────────────────────────────────────────────
    log.info('[Stage 5] Quality filter')   # progress marker
    passing, rejected = _stage5_filter(events_pred, meas)  # splits events into passing and rejected based on duration, D range, and signal quality
    result.events   = passing            # store the accepted events on the result object
    result.rejected = rejected           # store the rejected events so they can be inspected / plotted in grey
    log.info(f'  {len(passing)} events passed, {len(rejected)} rejected')  # summary counts

    # ── Stage 6: Output ───────────────────────────────────────────────────────
    log.info('[Stage 6] Output')          # progress marker
    _stage6_output(result, i100_c, out_dir, save_csv, save_plots)  # write CSV and/or generate the three diagnostic plots

    return result                         # return the fully populated PipelineResult


# ── Stage 1 — Baseline estimation ────────────────────────────────────────────

_BL_WINDOW_MS = 512   # sliding window width in ms; wide enough to contain many background bins but short enough to track slow drift
_BL_STRIDE_MS = 64    # step between successive windows; overlap = window - stride = 448 ms → smooth interpolation


def _histogram_mode(seg: np.ndarray, n_bins: int = 128) -> float:
    """
    Most common intensity value estimated via histogram mode (kHz).
    Uses 128 bins giving ~0.5 kHz resolution for typical FCS ranges.
    Equivalent to KDE mode for unimodal baseline distributions but
    orders-of-magnitude faster.
    """
    counts, edges = np.histogram(seg, bins=n_bins)  # bin the intensity values into 128 equal-width bins; counts = bin heights, edges = bin boundaries (length n_bins+1)
    i = int(np.argmax(counts))           # argmax returns the index of the tallest bin as a numpy int; int() converts to plain Python int
    return float(0.5 * (edges[i] + edges[i + 1]))  # midpoint of the tallest bin = histogram mode estimate; float() ensures a plain Python float is returned


def _stage1_baseline(
    i100:       np.ndarray,             # (N,) float64 — raw 1ms intensity trace in kHz
    window_ms:  int = _BL_WINDOW_MS,   # window width, default 512 ms
    stride_ms:  int = _BL_STRIDE_MS,   # stride between windows, default 64 ms
) -> np.ndarray:                        # returns (N,) float64 baseline array in kHz
    """
    Estimate the per-bin baseline by sliding-window KDE mode.

    Slides a `window_ms` window across the 1ms intensity trace with
    `stride_ms` stride.  The KDE mode of each window is the most common
    intensity level — the membrane baseline between molecular events.
    Window-centre estimates are linearly interpolated to a full per-bin
    (1ms) array.

    Parameters
    ----------
    i100       : (N,) float — raw 1ms intensity trace (kHz)
    window_ms  : window length in ms (= bins at 1ms)
    stride_ms  : stride between windows in ms

    Returns
    -------
    baseline : (N,) float64 — per-bin baseline in kHz
    """
    N         = len(i100)              # total number of 1ms bins in the trace
    t_centers = []                     # list of window-centre positions (in ms = bin index); will become x-coords for interpolation
    bl_hats   = []                     # list of baseline estimates (kHz) at each window centre; will become y-coords for interpolation

    for t0 in range(0, N - window_ms + 1, stride_ms):  # slide window from bin 0 to the last valid start position, stepping by stride_ms; range() generates start positions
        w = i100[t0 : t0 + window_ms]  # slice the window from the trace; Python slice [a:b] returns elements a..b-1
        t_centers.append(t0 + window_ms / 2)  # window centre in ms; division by 2 gives float
        bl_hats.append(_histogram_mode(w))     # estimate the background level within this window via the histogram mode

    if not bl_hats:                    # empty list is falsy; this happens if N < window_ms (trace shorter than one window)
        # Trace shorter than one window: global histogram mode
        return np.full(N, _histogram_mode(i100), dtype=np.float64)  # np.full(N, val) creates length-N array filled with val; use global mode as a constant baseline

    t_centers = np.array(t_centers, dtype=float)      # convert Python list → numpy array of floats for use with np.interp
    bl_hats   = np.array(bl_hats,   dtype=np.float64) # convert list → float64 array

    # Linearly interpolate to all bins; clamp extrapolation to nearest value
    t_all    = np.arange(N, dtype=float)  # [0, 1, 2, ..., N-1] — x-coordinates for all bins
    baseline = np.interp(t_all, t_centers, bl_hats,   # np.interp(x, xp, fp): for each x find interpolated value from (xp, fp) piecewise-linear curve
                         left=bl_hats[0], right=bl_hats[-1])  # left/right: clamp extrapolation beyond the window centres to the nearest estimate rather than extrapolating

    log.info(
        f'  Baseline: {len(bl_hats)} windows, '              # how many windows were processed
        f'range [{bl_hats.min():.1f}, {bl_hats.max():.1f}] kHz, '  # min/max of the estimated baseline across the trace
        f'median {float(np.median(bl_hats)):.1f} kHz'        # median baseline level; float() ensures clean formatting
    )
    return baseline                    # (N,) float64 array, same length as i100, in kHz


# ── Stage 1 — Noise model (retained for optional diagnostics) ─────────────────

def _stage1_noise(
    meas:     Measurement,             # measurement object for duration and has_raw flag
    ch_data,                           # ChannelData: i100_full, i050_full, i_min_full
    registry: ModelRegistry,           # model catalogue to select the noise model
) -> tuple:                            # returns (n_abs_kHz, model_label) — both may be None
    """
    Sliding-window noise estimation.

    Returns
    -------
    (n_abs: float | None, model_label: str | None)
    """
    noise_model = registry.noise_models.best_for(   # query the catalogue for the best noise model given the measurement constraints
        trace_length_ms = meas.total_duration_ms,   # minimum duration requirement
        has_raw         = meas.has_raw,             # sub-ms channels only if RAW files were loaded
    )
    if noise_model is None:            # no suitable noise model available
        log.warning('  No admissible noise model — Stage 1 skipped')  # warn and bail
        return None, None              # tuple of two Nones: n_abs and label both unavailable

    W        = noise_model.window_ms   # window length expected by this noise model (e.g. 4096 ms)
    stride   = W // 8                  # stride = 1/8 of the window; // is integer floor division; gives 87.5% overlap
    dt_n0    = sorted(noise_model.dt_channels)[0]   # finest dt among this model's channels, e.g. 0.1 ms; sorted() returns ascending list
    R_n0     = round(1.0 / dt_n0)                   # upsampling ratio: bins per 1ms at dt_n0 (e.g. 10 for dt=0.1ms); round() converts float to int
    B_100    = W                       # window size in 1ms bins (same as W since i100 is at 1ms)

    i100     = ch_data.i100_full       # reference to the 1ms trace; no copy, just an alias
    total    = len(i100)               # total number of 1ms bins in the trace
    n_win    = max(1, total // stride - 1)  # number of windows that fit; -1 avoids an off-by-one at the end; max(1,...) ensures at least one window

    def _stats(arr):                   # inner helper: compute [mean, std] of an array; returns [0,0] if arr is None
        if arr is None:                # sub-ms channel may be absent when has_raw=False
            return [0.0, 0.0]         # placeholder zeros preserve feature vector length
        return [float(arr.mean()), float(arr.std())]  # mean and std as plain Python floats

    n_abs_local = []                   # accumulates noise estimates from each window; we take the 20th percentile later

    for j in range(n_win):            # iterate over each window position
        t0 = j * stride               # window start in 1ms bins
        t1 = t0 + B_100               # window end (exclusive) in 1ms bins
        if t1 > total:                 # guard: don't read past the end of the trace
            break                     # exit the loop; remaining windows would be incomplete

        w100 = i100[t0:t1]            # 1ms window slice; shape (W,) in kHz

        w050 = (ch_data.i050_full[t0 * 2 : t1 * 2]   # 0.5ms window: multiply indices by 2 because 0.5ms has twice as many bins per ms
                if ch_data.i050_full is not None else None)  # conditional expression: None if 0.5ms channel not available

        w_min = (ch_data.i_min_full[t0 * R_n0 : t1 * R_n0]  # finest-dt window: multiply by R_n0 (e.g. 10) to convert 1ms indices to 0.1ms indices
                 if ch_data.i_min_full is not None else None)  # None if sub-ms channel not available

        # 6 raw stats: [mean, std] × 3 channels — used by all noise models
        feats = np.array(
            _stats(w_min) + _stats(w050) + _stats(w100), dtype=np.float64  # concatenate three [mean, std] pairs into a length-6 feature vector; + on lists is concatenation
        )

        # Build traces dict keyed by dt_ms
        _win_by_dt = {dt_n0: w_min, 0.5: w050, 1.0: w100}  # mapping from dt value → window array; keys are floats matching the model's dt_channels set
        noise_traces = {dt: _win_by_dt[dt]                  # dict comprehension: keep only dt channels the noise model actually uses
                        for dt in noise_model.dt_channels if dt in _win_by_dt}  # 'if dt in _win_by_dt' guards against requesting a dt not in our mapping

        try:                           # noise model inference can fail for pathological windows (e.g. all zeros)
            n_hat = noise_model.predict(traces=noise_traces, features=feats)  # run inference; returns scalar n_abs estimate in kHz
            n_abs_local.append(n_hat)  # accumulate this window's estimate
        except Exception as e:         # catch any error (torch, numpy, etc.) so one bad window doesn't abort the whole run
            log.debug(f'  Noise window j={j} failed: {e}')  # debug level: only visible with verbose logging

    if not n_abs_local:                # if no windows produced a valid estimate (e.g. trace too short after all guards)
        return None, noise_model.label  # return None noise level but still report the model label

    n_abs = float(np.percentile(n_abs_local, 20))  # 20th percentile of window estimates; robust to occasional high-noise windows; float() converts numpy scalar → Python float
    log.info(f'  n_abs = {n_abs:.3f} kHz  ({len(n_abs_local)} windows)')  # :.3f → 3 decimal places; report how many windows contributed
    return n_abs, noise_model.label    # return (noise_floor_kHz, model_label_string)


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def _stage2_subtract(
    ch_data,                           # ChannelData with i100_full, i050_full, i_min_full (all in kHz)
    baseline: np.ndarray,              # (N,) float64 — per-bin 1ms baseline in kHz, from _stage1_baseline
    meas:     Measurement,             # Measurement object; used only for dt_min_ms
) -> tuple:                            # returns (i_min_clean, i050_clean, i100_clean) — all float64 kHz, clipped at 0
    """
    Baseline subtraction.

    Subtracts the per-bin baseline (kHz) from each multi-dt channel.
    For sub-ms channels, the 1ms baseline is upsampled via np.repeat
    (step-function approximation — appropriate given the 512ms window
    over which each baseline estimate is computed).

    Parameters
    ----------
    ch_data  : ChannelData with i100_full, i050_full, i_min_full
    baseline : (N,) float64 — per-bin baseline at 1ms resolution (kHz)
    meas     : Measurement (provides dt_min_ms for upsampling factor)

    Returns
    -------
    (i_min_clean, i050_clean, i100_clean) — float64, kHz, clipped at 0
    """
    N         = len(baseline)          # number of 1ms bins; used only for documentation / guards elsewhere
    R_min     = round(1.0 / meas.dt_min_ms)   # upsampling factor for the finest channel: e.g. 1.0/0.1 = 10; round() gives int

    def _sub(arr, upsample: int = 1):  # inner helper: subtract the baseline from one channel with the given upsampling factor
        if arr is None:                # channel absent (has_raw=False for sub-ms channels)
            return None                # propagate None so callers can check before using
        bl  = np.repeat(baseline, upsample)[:len(arr)]  # np.repeat repeats each baseline value 'upsample' times → step-function upsampling; [:len(arr)] trims to exact array length in case of rounding
        out = np.maximum(0.0, arr.astype(np.float64) - bl)  # subtract upsampled baseline; np.maximum clips any negative result to 0 (photon counts can't be negative); astype ensures float64 arithmetic
        clip_frac = float((out == 0).mean())  # fraction of bins that were clipped to zero; (out==0) is a boolean array; .mean() gives fraction of True values
        if clip_frac > 0.50:           # if more than half the bins are clipped, the baseline may be overestimated
            log.warning(
                f'  {clip_frac:.1%} of bins clipped to 0 after baseline subtraction'  # :.1% formats as percentage with 1 decimal place
            )
        return out                     # (len(arr),) float64 array in kHz, non-negative

    i_min_c = _sub(ch_data.i_min_full, upsample=R_min)  # subtract from 0.1ms channel; upsample=10 repeats each 1ms baseline value 10×
    i050_c  = _sub(ch_data.i050_full,  upsample=2)       # subtract from 0.5ms channel; upsample=2 repeats each 1ms value 2×
    i100_c  = _sub(ch_data.i100_full,  upsample=1)       # subtract from 1ms channel; upsample=1 means no repetition (baseline and trace are already the same resolution)

    log.info(
        f'  Stage 2: subtracted baseline '
        f'[{baseline.min():.1f}, {baseline.max():.1f}] kHz '   # range of baseline values across the trace; :.1f = 1 decimal place
        f'({float((i100_c == 0).mean()):.1%} of 1ms bins at zero)'  # fraction of 1ms bins that were clipped; diagnostic for over-subtraction
    )
    return i_min_c, i050_c, i100_c    # return all three cleaned channels; callers unpack as: i_min_c, i050_c, i100_c = _stage2_subtract(...)


# ── Stage 4 ──────────────────────────────────────────────────────────────────

def _stage4_conformal(events: list, d_model: ModelWrapper) -> list:
    """
    Augment events (already carrying D_hat from run_stage3_v2) with
    conformal uncertainty intervals and pipeline bookkeeping fields.
    """
    for ev in events:                  # iterate over each event dict in-place; modifications to ev affect the original dict
        D_hat    = ev.get('D_hat', np.nan)        # retrieve predicted diffusion coefficient; default np.nan if missing
        log10_D  = ev.get('log10_D', float(np.log10(max(D_hat, 1e-10))))  # retrieve or compute log10(D_hat); max with 1e-10 prevents log(0)
        pad_frac = ev.get('pad_frac', 0.0)        # fraction of the 4096-bin window that was zero-padded; 0 = no padding (long transit)

        D_low = D_high = None          # initialise interval bounds; remains None if no conformal calibration is available
        if d_model.conformal_intervals is not None and not np.isnan(D_hat):  # only compute intervals if the model has calibration data and D_hat is valid
            q      = d_model.conformal_intervals.lookup(pad_frac)  # look up the calibrated half-width (in log10 units) for this pad_frac
            D_low  = 10.0 ** (log10_D - q)        # lower bound: shift log10 down by q, exponentiate back to linear scale
            D_high = 10.0 ** (log10_D + q)        # upper bound: shift log10 up by q, exponentiate back

        flags = list(ev.get('flags', []))          # copy existing flags list (or start fresh); list() prevents aliasing the original
        if pad_frac > 0.75:            # if more than 75% of the inference window was padding, signal is very short
            flags.append('low_signal') # add soft flag; event is kept but flagged for downstream review

        ev.update(                     # dict.update() adds/overwrites multiple keys at once
            D_low         = D_low,             # lower conformal bound (µm²/s); None if unavailable
            D_high        = D_high,            # upper conformal bound (µm²/s); None if unavailable
            d_hat_all     = [D_hat],           # list wrapping D_hat for compatibility with multi-window events (n_win > 1)
            d_model_label = d_model.label,     # record which model produced this prediction
            flags         = flags,             # updated flags list (may include 'low_signal')
        )
    return events                      # return the same list (modified in-place) for chaining


# ── Stage 5 ──────────────────────────────────────────────────────────────────

# Quality filter thresholds — module-level constants for easy tuning
QF_MIN_DURATION_MS = 64     # hard discard: transit shorter than 64ms is likely noise or a sub-threshold artifact
QF_D_MIN           = 0.0    # hard discard: D ≤ 0 is non-physical (model occasionally predicts near-zero)
QF_D_MAX           = 14.0   # hard discard: D > 14 µm²/s is outside the training range; prediction unreliable
QF_N_FRAC_HARD     = 0.90   # hard discard threshold: >90% of bins at background → reserved for future use
QF_N_FRAC_SOFT     = 0.50   # soft flag threshold: >50% at background → reserved for future use
QF_D_CV_SOFT       = 0.35   # soft flag: std of log10(D) across windows > 0.35 → D is inconsistent within the event


def _stage5_filter(
    events: list,              # list of event dicts augmented by stage 4
    meas:   Measurement,       # measurement object; available for future filters (e.g. n_abs-based n_frac)
) -> tuple:                    # returns (passing: list, rejected: list)
    """
    Per-event quality filter.

    Returns
    -------
    (passing: list, rejected: list)
    """
    passing  = []              # events that pass all hard criteria
    rejected = []              # events that fail at least one hard criterion

    for ev in events:          # process each event dict
        dur     = ev.get('t_right', 0) - ev.get('t_left', 0)  # event duration in ms (t_right and t_left are in ms bin indices); default 0 if keys missing
        D_hat   = ev.get('D_hat', np.nan)   # predicted D; np.nan default triggers 'd_out_of_range' flag below
        flags   = list(ev.get('flags', []))  # copy flag list from stage 4; list() prevents aliasing
        discard = False                      # accumulator: set True if any hard criterion fails

        # Hard: too short
        if dur < QF_MIN_DURATION_MS:         # transit duration below 64ms hard threshold
            flags.append('too_short')        # tag the reason for rejection
            discard = True                   # mark for rejection; don't break — continue checking other criteria to collect all flags

        # Hard: D out of physical / training range
        if not (QF_D_MIN < D_hat < QF_D_MAX):   # chained comparison: D must be strictly between 0 and 14; np.nan always returns False here → rejected
            flags.append('d_out_of_range')        # tag the reason
            discard = True                        # mark for rejection

        # Soft: D_cv (Case 3 only, n_win ≥ 2)
        n_win     = ev.get('n_win', 1)            # number of sub-windows used for this event (>1 for very long transits)
        d_hat_all = ev.get('d_hat_all', [])       # list of D_hat values across sub-windows
        if n_win >= 2 and len(d_hat_all) >= 2:    # only meaningful when multiple sub-windows contributed
            log10_vals = np.log10(np.clip(d_hat_all, 1e-10, None))  # np.clip prevents log(0); convert all D values to log10 scale
            d_cv       = float(np.std(log10_vals))   # standard deviation of log10(D) across windows; float() converts numpy scalar → Python float
            ev['D_cv'] = d_cv                        # store for CSV output
            if d_cv > QF_D_CV_SOFT:                  # variability exceeds soft threshold
                flags.append('variable_D')           # soft flag only — event is NOT discarded, just tagged
        else:
            ev['D_cv'] = None                        # single-window event: D_cv not applicable

        ev['flags'] = flags            # write the (possibly extended) flags list back to the event dict

        if discard:                    # at least one hard criterion failed
            rejected.append(ev)        # route to rejected list
        else:
            passing.append(ev)         # route to accepted list

    return passing, rejected           # unpack at call site: passing, rejected = _stage5_filter(...)


# ── Stage 6 ──────────────────────────────────────────────────────────────────

def _stage6_output(
    result:     PipelineResult,        # fully populated result (events + rejected + metadata)
    i100_clean: np.ndarray,            # baseline-subtracted 1ms trace for the trace plot
    out_dir:    Path,                  # directory where output files are written
    save_csv:   bool,                  # if True, write the per-event CSV
    save_plots: bool,                  # if True, generate the three diagnostic plots
) -> None:                             # no return value; side effects only (file I/O)
    """Stage 6: write CSV and annotated plots."""

    stem = f'{result.name}_{result.channel_id}'   # base filename stem, e.g. 'lateralDorsal_nt_1_S1'
    if result.run_id:                  # if an optional run ID was provided
        stem = f'{result.run_id}_{stem}'  # prepend it: 'run42_lateralDorsal_nt_1_S1'

    # ── CSV ───────────────────────────────────────────────────────────────────
    if save_csv:                       # conditional: skip file I/O if caller set save_csv=False
        csv_path = out_dir / f'{stem}_events.csv'   # / operator on Path objects joins paths; constructs full output path
        _write_csv(result.events, csv_path, result)  # write all passing events to CSV
        log.info(f'  CSV → {csv_path}')              # log the output path for confirmation

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots:                     # conditional: skip plotting if caller set save_plots=False
        _plot_trace(result, i100_clean, out_dir, stem)    # Plot 1: full intensity trace with event markers
        _plot_d_scatter(result, out_dir, stem)             # Plot 2: D̂ vs time scatter coloured by pad_fraction
        _plot_d_histogram(result, out_dir, stem)           # Plot 3: log10(D̂) histogram


def _write_csv(events: list, path: Path, result: PipelineResult) -> None:
    """Write per-event results as CSV."""
    import csv                         # import inside function: csv is only needed here; avoids top-level import if CSV writing is skipped

    fieldnames = [                     # ordered list of column names; DictWriter uses this to determine column order and which keys to write
        't_peak_ms', 't_left_ms', 't_right_ms', 'duration_ms',  # event timing in ms
        'D_hat', 'log10_D', 'D_low', 'D_high',                  # diffusion coefficient and conformal interval
        'pad_fraction', 'n_win', 'D_cv',                         # quality metrics
        'k_win', 'score', 'category',                            # detection bookkeeping
        'n_frac', 'model_used', 'noise_model', 'flags',          # metadata and quality flags
        'w12_ms',                                                  # half-width at 30% of peak (ms)
        't_left_history', 't_right_history', 'D_hat_history',   # expansion loop history (;-separated)
        'baseline_w12_kHz', 'baseline_history',                  # baseline values during expansion
        'swallowed_by_ms',                                         # if this event was absorbed into a larger one
        # v4.0/4.1 — baseline-factor D predictions (one column per factor)
        'd_hat_f105', 'd_hat_f110', 'd_hat_f115', 'd_hat_f120',  # D_hat for factor 1.05, 1.10, 1.15, 1.20
        'd_hat_f125', 'd_hat_f130', 'd_hat_f135', 'd_hat_f140', 'd_hat_f145',  # v4.31: 1.25–1.45
        'd_hat_f150', 'd_hat_f160', 'd_hat_f180', 'd_hat_f200',  # D_hat for factor 1.50, 1.60, 1.80, 2.00
        'd_hist_f105', 'd_hist_f110', 'd_hist_f115', 'd_hist_f120',  # expansion history for each factor (;-separated)
        'd_hist_f125', 'd_hist_f130', 'd_hist_f135', 'd_hist_f140', 'd_hist_f145',  # v4.31
        'd_hist_f150', 'd_hist_f160', 'd_hist_f180', 'd_hist_f200',
        # v4.3 — Pearson-guided expansion: which BL_FACTOR drove the final expansion step
        'pearson_best_factor',
    ]

    with open(path, 'w', newline='') as f:   # open file for writing; newline='' prevents csv module from double-writing \r\n on Windows
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')  # DictWriter maps dict keys → columns; extrasaction='ignore' silently drops any event dict keys not in fieldnames
        writer.writeheader()                 # writes the column name row as the first line
        for ev in events:                    # iterate over each accepted event dict
            row = {                          # build the CSV row dict by mapping event dict keys → fieldname keys
                't_peak_ms':   ev.get('t_peak'),    # peak bin index (= ms from start); .get() returns None if key absent → CSV cell is empty
                't_left_ms':   ev.get('t_left'),    # left boundary of the detected transit (ms)
                't_right_ms':  ev.get('t_right'),   # right boundary of the detected transit (ms)
                'duration_ms': ev.get('t_right', 0) - ev.get('t_left', 0),  # transit duration; default 0 prevents TypeError if either key is missing
                'D_hat':       ev.get('D_hat'),     # primary diffusion coefficient prediction (µm²/s)
                'log10_D':     ev.get('log10_D'),   # log10 of D_hat; stored separately for convenience in analysis
                'D_low':       ev.get('D_low'),     # lower conformal bound (µm²/s); None if no calibration
                'D_high':      ev.get('D_high'),    # upper conformal bound (µm²/s); None if no calibration
                'pad_fraction': ev.get('pad_frac'),  # fraction of 4096-bin window that was zero-padded
                'n_win':       ev.get('n_win'),     # number of sub-windows (1 for most events)
                'D_cv':        ev.get('D_cv'),      # std of log10(D) across sub-windows; None for single-window events
                'k_win':       ev.get('k_win'),     # index of the highest-scoring sub-window
                'score':       ev.get('score'),     # SNR-like score: (peak - baseline) / sqrt(baseline)
                'category':    ev.get('category'),  # 1 = wide transit (Cat-1), 2 = narrow transit (Cat-2)
                'n_frac':      ev.get('n_frac'),    # fraction of event bins at noise floor; reserved
                'w12_ms':           ev.get('w12_seed'),   # half-width of the smoothed peak at 30% height (seed step)
                't_left_history':   ';'.join(str(v) for v in ev.get('t_left_hist',  [])),  # expansion loop left boundary at each step; ';'.join serialises the list as a single ;-delimited string
                't_right_history':  ';'.join(str(v) for v in ev.get('t_right_hist', [])),  # expansion loop right boundary at each step
                'D_hat_history':    ';'.join(f'{v:.6g}' for v in ev.get('d_hist',   [])),  # D_hat at each expansion step; :.6g = 6 significant figures, removes trailing zeros
                'baseline_w12_kHz': ev.get('baseline_w12'),  # baseline level at the seed w12 step (kHz)
                'baseline_history': ';'.join(f'{v:.6g}' for v in ev.get('baseline_hist', [])),  # edge baseline at each expansion step
                'swallowed_by_ms':  ev.get('swallowed_by'),   # t_peak of the event that absorbed this one; None if not swallowed
                # v4.0 factor predictions — one value per factor, from the final expansion step
                'd_hat_f105': ev.get('d_hat_by_factor', {}).get(1.05),   # D_hat when baseline was multiplied by 1.05; {} default prevents KeyError if key absent
                'd_hat_f110': ev.get('d_hat_by_factor', {}).get(1.10),   # factor 1.10
                'd_hat_f115': ev.get('d_hat_by_factor', {}).get(1.15),   # factor 1.15
                'd_hat_f120': ev.get('d_hat_by_factor', {}).get(1.20),   # factor 1.20
                'd_hat_f125': ev.get('d_hat_by_factor', {}).get(1.25),   # factor 1.25 (v4.31)
                'd_hat_f130': ev.get('d_hat_by_factor', {}).get(1.30),   # factor 1.30 (v4.31)
                'd_hat_f135': ev.get('d_hat_by_factor', {}).get(1.35),   # factor 1.35 (v4.31)
                'd_hat_f140': ev.get('d_hat_by_factor', {}).get(1.40),   # factor 1.40 (v4.31)
                'd_hat_f145': ev.get('d_hat_by_factor', {}).get(1.45),   # factor 1.45 (v4.31)
                'd_hat_f150': ev.get('d_hat_by_factor', {}).get(1.50),   # factor 1.50
                'd_hat_f160': ev.get('d_hat_by_factor', {}).get(1.60),   # factor 1.60 (v4.31)
                'd_hat_f180': ev.get('d_hat_by_factor', {}).get(1.80),   # factor 1.80 (v4.31)
                'd_hat_f200': ev.get('d_hat_by_factor', {}).get(2.00),   # factor 2.00
                'd_hist_f105': ';'.join(f'{v:.6g}' for v in              # full expansion history for factor 1.05; ;-delimited
                               ev.get('d_hist_by_factor', {}).get(1.05, [])),  # {} then [] defaults handle missing outer or inner key
                'd_hist_f110': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.10, [])),
                'd_hist_f115': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.15, [])),
                'd_hist_f120': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.20, [])),
                'd_hist_f125': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.25, [])),
                'd_hist_f130': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.30, [])),
                'd_hist_f135': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.35, [])),
                'd_hist_f140': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.40, [])),
                'd_hist_f145': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.45, [])),
                'd_hist_f150': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.50, [])),
                'd_hist_f160': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.60, [])),
                'd_hist_f180': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(1.80, [])),
                'd_hist_f200': ';'.join(f'{v:.6g}' for v in
                               ev.get('d_hist_by_factor', {}).get(2.00, [])),
                # v4.3: Pearson-guided factor; 1.0 when pearson_fn was not supplied (backward-compatible default)
                'pearson_best_factor': ev.get('pearson_best_factor', 1.0),
                'model_used':  result.d_model_label,   # same model for all events in this result; taken from result not ev
                'noise_model': result.noise_model_label,  # noise model label (may be None)
                'flags':       ','.join(ev.get('flags', [])),  # comma-separated flag strings, e.g. 'low_signal,variable_D'
            }
            writer.writerow(row)       # write one CSV row for this event; DictWriter uses fieldnames order and handles None → empty cell


def _plot_trace(
    result:    PipelineResult,   # result object containing events, rejected, name, channel_id
    i100_c:   np.ndarray,        # baseline-subtracted 1ms trace (kHz) for plotting the signal
    out_dir:   Path,             # output directory
    stem:      str,              # filename stem (no extension)
) -> None:                       # side effect: writes PNG file
    """
    Plot 1: i100_clean intensity trace with events shaded by log10(D̂).
    """
    try:                                # matplotlib is an optional dependency on the cluster
        import matplotlib.pyplot as plt             # main plotting interface
        import matplotlib.cm as cm                  # colourmap utilities
        from matplotlib.colors import Normalize     # maps data values → [0,1] for a colormap
    except ImportError:                 # if matplotlib is not installed
        log.warning('matplotlib not available — skipping plots')  # warn and skip; not fatal
        return

    # ── aesthetics (from MEMORY.md) ──────────────────────────────────────────
    CURVE_COLOR = '#00FF00'             # bright green for the intensity trace line
    SPINE_LW    = 1.5                   # spine (axis border) linewidth
    FS_TITLE    = 14; FS_LABEL = 15; FS_TICK = 12   # font sizes for title, axis labels, tick labels; semicolons allow multiple assignments on one line

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')   # create a 14×3.5 inch figure with a white background; returns (Figure, Axes)
    ax.set_facecolor('white')           # set the axes interior background to white (separate from figure background)

    t_ms = np.arange(len(i100_c))      # [0, 1, 2, ..., N-1] — time axis in ms (1 bin = 1 ms)
    ax.plot(t_ms * 1e-3, i100_c, color=CURVE_COLOR, lw=0.5, alpha=0.85)   # plot the trace; * 1e-3 converts ms → seconds; lw=linewidth; alpha=opacity

    # shade events by log10(D)
    d_vals = [ev.get('log10_D') for ev in result.events if ev.get('log10_D') is not None]  # list comprehension: collect all non-None log10(D) values; used only to check if any events have valid D
    if d_vals:                          # only set up colormap if there is at least one event with a valid D
        cmap = cm.get_cmap('RdYlBu_r') # red-yellow-blue reversed: red = fast, blue = slow diffusers
        norm = Normalize(vmin=-2, vmax=1)  # map log10(D) range [-2, 1] → [0, 1] for the colormap
        for ev in result.events:        # iterate over accepted events to shade them
            if ev.get('log10_D') is None:   # skip events with no valid D prediction
                continue
            c = cmap(norm(ev['log10_D']))   # look up the RGBA colour for this event's log10(D)
            ax.axvspan(ev['t_left'] * 1e-3, ev['t_right'] * 1e-3,
                       alpha=0.30, color=c)  # axvspan draws a vertical shaded rectangle from t_left to t_right; alpha=0.30 makes it semi-transparent
            ax.axvline(ev['t_peak'] * 1e-3, color=c, lw=0.8, ls='--', alpha=0.7)  # axvline draws a vertical dashed line at t_peak; ls='--' = dashed linestyle
        # rejected events in grey
        for ev in result.rejected:          # shade rejected events separately in grey so they're visible but distinct
            ax.axvspan(ev['t_left'] * 1e-3, ev['t_right'] * 1e-3,
                       alpha=0.15, color='grey')  # lighter alpha so they don't dominate

    for sp in ax.spines.values():       # iterate over all four axis spines (top, bottom, left, right)
        sp.set_linewidth(SPINE_LW); sp.set_color('black')  # thicken and blacken each spine
    ax.tick_params(colors='black', labelsize=FS_TICK)   # set tick marks and tick labels to black at FS_TICK font size
    ax.set_xlabel('Time (s)', fontsize=FS_LABEL, color='black')          # x-axis label
    ax.set_ylabel('Intensity (kHz)', fontsize=FS_LABEL, color='black')   # y-axis label
    ax.set_title(f'{result.name} / {result.channel_id}  '
                 f'({result.n_events} events)', fontsize=FS_TITLE, color='black')  # title: measurement name / channel and event count

    fig.tight_layout()                  # automatically adjust subplot parameters to prevent label clipping
    fig.savefig(out_dir / f'{stem}_trace.png', dpi=150,
                facecolor='white', bbox_inches='tight')  # save as PNG at 150 dpi; facecolor='white' ensures white background in saved file; bbox_inches='tight' crops to content
    plt.close(fig)                      # release the figure from memory; important in loops to prevent memory accumulation


def _plot_d_scatter(
    result:  PipelineResult,    # result with events list
    out_dir: Path,              # output directory
    stem:    str,               # filename stem
) -> None:                      # side effect: writes PNG file
    """
    Plot 2: D̂ scatter — t_peak (s) vs log10(D̂), coloured by pad_fraction.
    """
    if not result.events:       # no events → nothing to scatter-plot; guard prevents empty-array errors
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
    except ImportError:
        return                  # silently skip if matplotlib unavailable

    SPINE_LW = 1.5; FS_LABEL = 15; FS_TICK = 12; FS_TITLE = 14  # aesthetics constants

    t_peaks   = np.array([ev['t_peak'] for ev in result.events]) * 1e-3   # list comprehension → numpy array of t_peak values; * 1e-3 converts ms → s
    log10_D   = np.array([ev['log10_D'] for ev in result.events])          # y-axis values
    pad_fracs  = np.array([ev.get('pad_frac', 0) for ev in result.events]) # colour values; default 0 if key missing
    D_low     = np.array([ev.get('D_low')  or np.nan for ev in result.events])   # lower conformal bounds; 'or np.nan' replaces None with nan so numpy handles it gracefully
    D_high    = np.array([ev.get('D_high') or np.nan for ev in result.events])   # upper conformal bounds

    cmap = cm.get_cmap('viridis_r')    # reversed viridis: low pad_frac (good signal) = yellow, high pad_frac (weak signal) = purple
    norm = Normalize(vmin=0, vmax=1)   # pad_fraction is already in [0,1]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')   # 10×4 inch scatter plot
    ax.set_facecolor('white')

    sc = ax.scatter(t_peaks, log10_D, c=pad_fracs, cmap=cmap, norm=norm,  # scatter plot: x=time, y=log10(D), colour=pad_frac
                    s=40, zorder=3, edgecolors='grey', linewidths=0.3)     # s=marker size; zorder=3 draws markers on top of error bars; grey edge outlines

    # Conformal intervals as error bars (in log10 units)
    if np.any(~np.isnan(D_low)):       # only draw error bars if at least one event has a valid conformal interval; ~ = bitwise NOT on boolean array; np.any checks if any are True
        yerr_lo = log10_D - np.log10(np.clip(D_low, 1e-10, None))   # lower error bar length in log10 units; np.clip prevents log(0); np.log10 applies element-wise
        yerr_hi = np.log10(np.clip(D_high, 1e-10, None)) - log10_D  # upper error bar length
        ax.errorbar(t_peaks, log10_D,
                    yerr=[yerr_lo, yerr_hi],            # [lower_lengths, upper_lengths] format for asymmetric error bars
                    fmt='none', ecolor='#888888', elinewidth=0.8, capsize=2, zorder=2)  # fmt='none' = no marker; ecolor = bar colour; capsize = horizontal cap width; zorder=2 draws behind markers

    plt.colorbar(sc, ax=ax, label='pad_fraction', shrink=0.8)  # add colorbar tied to scatter plot 'sc'; shrink=0.8 makes it 80% of axes height
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)     # thicken all four spines
    ax.tick_params(labelsize=FS_TICK)  # tick label font size
    ax.set_xlabel('t_peak (s)', fontsize=FS_LABEL)           # x-axis label
    ax.set_ylabel('log₁₀ D̂  (µm²/s)', fontsize=FS_LABEL)   # y-axis label; ₁₀ and ² are Unicode subscript/superscript
    ax.set_title(f'{result.name} — D̂ scatter ({result.n_events} events)',
                 fontsize=FS_TITLE)
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_d_scatter.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    plt.close(fig)                     # release memory


def _plot_d_histogram(
    result:  PipelineResult,   # result with events list
    out_dir: Path,             # output directory
    stem:    str,              # filename stem
) -> None:                     # side effect: writes PNG file
    """
    Plot 3: log10(D̂) histogram across all passing events.
    """
    if not result.events:      # guard: no events → nothing to histogram
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    SPINE_LW = 1.5; FS_LABEL = 15; FS_TICK = 12; FS_TITLE = 14  # aesthetics constants

    log10_D = np.array([ev['log10_D'] for ev in result.events])   # collect all log10(D) values as a numpy array
    bins    = np.arange(-2.5, 1.5 + 0.1, 0.1)   # bin edges from -2.5 to 1.5 in steps of 0.1; + 0.1 ensures the right edge 1.5 is included (np.arange excludes stop)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.hist(log10_D, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)  # histogram of log10(D) values; steelblue fill; thin white borders between bars for readability
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)     # thicken all spines
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel('log₁₀ D̂  (µm²/s)', fontsize=FS_LABEL)
    ax.set_ylabel('Events', fontsize=FS_LABEL)
    ax.set_title(f'{result.name} — D distribution  (n={len(log10_D)})',   # n= shows total event count in histogram
                 fontsize=FS_TITLE)
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_d_histogram.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    plt.close(fig)                     # release memory; essential in batch runs over many measurements
