"""
stage3_v3_7.py — Stage 3 event detection pipeline, version 3.7
═══════════════════════════════════════════════════════════

Detects single-molecule dwell events in a raw FCS intensity trace via
a peak-finding + D-guided window expansion algorithm.

Algorithm (Steps 1–2; Step 3 / rehashing reserved for future work)
────────────────────────────────────────────────────────────────────
Input: TraceBundle containing
  • i1ms     — raw intensity at 1ms resolution (kHz)
  • baseline — per-bin background from noise models (kHz)
  • i_dt050  — raw intensity at 0.5ms resolution (kHz)  [optional]
  • i_dt010  — raw intensity at 0.1ms resolution (kHz)  [optional]

Step 1 — Peak finding + claiming
  a. Find local maxima in i1ms at or above baseline.
  b. Compute half-width w12 at TRNST_LVL × signal[tp] around each peak.
  c. Classify: Cat 1 (2*w12 >= CAT1_MIN_WIDTH_MS) or Cat 2 (narrower).
  d. Cat 1 claiming loop (descending h_peak):
       claim [tp-w12, tp+w12]; exclude all peaks whose tp falls inside.
  e. Cat 2 claiming loop (descending h_peak):
       claim [tp-W_START2_MS//2, tp+W_START2_MS//2]; exclude Cat 2 peaks inside.
  f. Clean up excluded peaks; sort survivors by h_peak descending.

Step 2 — D-guided window expansion (per event)
  a. Predict D on a fixed W_D=4096ms window centred on tp.
  b. D > D_FAST_CUTOFF → fast group; keep seed window, no expansion.
  c. Else: look up tau_max = TRANSIT_TIMES[i] where D <= D_CUTOFFS[i].
  d. Double window w → 2w until:
       Stop 1 — w_new >= W_MAX_MS  OR  overlap with existing claimed bins
       Stop 2 — w_new >= tau_max
  e. After each accepted doubling: re-predict D, update tau_max.

Output dict keys
────────────────
  t_peak, t_left, t_right, W_seg,
  D_hat, log10_D, category, fast_group,
  score, pad_frac, n_win, k_win
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.signal import find_peaks

# ── Detection parameters ─────────────────────────────────────────────────────
W_D               = 4096   # D-model inference window (ms); fixed for all models
TRNST_LVL         = 0.30   # width measured at this fraction of peak value
CAT1_MIN_WIDTH_MS = 200    # 2*w12 >= this → Cat 1 (broad / slow)
W_START2_MS       = 128    # Cat 2 seed window total width (ms)
W_MAX_MS          = 4096   # maximum event window width (ms)
D_FAST_CUTOFF     = 1.00   # D > this → fast group; keep seed, no expansion (µm²/s)
PEAK_DISTANCE_MS  = 50     # minimum inter-peak distance for peak finder (ms)
AMPLITUDE_FRAC_MIN = 0.50  # discard peaks below this fraction of the global peak max
W12_SMOOTH_MS      = 50    # box-smooth window (ms) used only for w12 / Cat-1 classification

# Expansion table from defaultRecParameters.m
D_CUTOFFS     = [0.02, 0.06, 0.20, 0.50, 1.00]  # µm²/s
TRANSIT_TIMES = [2048, 1024,  512,  256,  128]   # ms


# ── TraceBundle ──────────────────────────────────────────────────────────────

@dataclass
class TraceBundle:
    """
    Multi-resolution signal container for Stage 3 v2.

    Attributes
    ----------
    i1ms     : (N,) float — raw intensity, 1ms bins, kHz
    baseline : scalar or (N,) float — per-bin background from noise models, kHz.
               A scalar is broadcast to shape (N,).
    i_dt050  : (2N,) float or None — intensity at 0.5ms bins, kHz
    i_dt010  : (10N,) float or None — intensity at 0.1ms bins, kHz
    """
    i1ms:    np.ndarray
    baseline: object                   # broadened to (N,) in __post_init__
    i_dt050: Optional[np.ndarray] = None
    i_dt010: Optional[np.ndarray] = None

    def __post_init__(self):
        self.i1ms = np.asarray(self.i1ms, dtype=np.float64)
        N  = len(self.i1ms)
        bl = np.asarray(self.baseline, dtype=np.float64).ravel()
        self.baseline = np.full(N, bl[0]) if bl.size == 1 else bl
        if self.i_dt050 is not None:
            self.i_dt050 = np.asarray(self.i_dt050, dtype=np.float64)
        if self.i_dt010 is not None:
            self.i_dt010 = np.asarray(self.i_dt010, dtype=np.float64)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _lookup_tau_max(D_hat: float) -> int:
    """Return max allowable transit time (ms) for D in µm²/s."""
    for cutoff, tmax in zip(D_CUTOFFS, TRANSIT_TIMES):
        if D_hat <= cutoff:
            return tmax
    return TRANSIT_TIMES[-1]


def _compute_w12(signal: np.ndarray, tp: int,
                 baseline: Optional[np.ndarray] = None) -> int:
    """
    Half-width (ms bins) at TRNST_LVL above-baseline amplitude around peak at tp.

    I_cut = bl_edge + TRNST_LVL * (signal[tp] - bl_edge)

    where bl_edge is the minimum of baseline over ±W12_MAX_MS around tp.
    Using the minimum rather than baseline[tp] prevents the noise model's
    elevated background estimate within a bright transit from artificially
    narrowing the measured width.

    If baseline is None, falls back to TRNST_LVL * signal[tp].

    t_left_w  = last index < tp where signal < I_cut  (or 0 if none)
    t_right_w = first index > tp where signal < I_cut (or N-1 if none)
    w12       = (t_right_w - t_left_w) // 2
    """
    if baseline is not None:
        lo    = max(0, tp - W12_MAX_MS)
        hi    = min(len(baseline) - 1, tp + W12_MAX_MS)
        bl    = float(baseline[lo : hi + 1].min())
        I_cut = bl + TRNST_LVL * max(float(signal[tp]) - bl, 0.0)
    else:
        I_cut = TRNST_LVL * signal[tp]
    N     = len(signal)

    left_idx  = np.where(signal[:tp] < I_cut)[0]
    t_left_w  = int(left_idx[-1]) if len(left_idx)  > 0 else 0

    right_idx = np.where(signal[tp + 1:] < I_cut)[0]
    t_right_w = int(right_idx[0]) + tp + 1 if len(right_idx) > 0 else N - 1

    return max((t_right_w - t_left_w) // 2, 1)


def _build_traces(bundle: TraceBundle, tp: int, d_model,
                  baseline_scalar: Optional[float] = None) -> dict:
    """
    Build padded inference windows of exactly W_D ms centred on tp,
    one per dt channel required by d_model.

    Parameters
    ----------
    baseline_scalar : if provided, subtract this constant from all bins instead
        of the per-bin bundle.baseline array.  Used during expansion to anchor
        the background estimate to the edges of the current event window rather
        than the model-estimated baseline within the window itself.

    Returns {dt_ms: ndarray} — each array has the length expected by the model.
    """
    N     = len(bundle.i1ms)
    inf_s = max(0,     tp - W_D // 2)
    inf_e = min(N - 1, tp + W_D // 2)

    def _pad(arr, target: int) -> np.ndarray:
        L = len(arr)
        if L >= target:
            return arr[:target]
        pl = (target - L) // 2
        pr = target - L - pl
        return np.concatenate([np.zeros(pl), arr, np.zeros(pr)])

    traces = {}
    for dt in sorted(d_model.dt_channels):
        if abs(dt - 1.0) < 1e-9:
            # Subtract baseline so the D model receives background-free input,
            # matching the training data (background-subtracted, clipped at 0).
            bl  = (np.full(inf_e + 1 - inf_s, baseline_scalar)
                   if baseline_scalar is not None
                   else bundle.baseline[inf_s : inf_e + 1])
            seg = np.maximum(0.0, bundle.i1ms[inf_s : inf_e + 1] - bl)
            traces[dt] = _pad(seg, W_D)

        elif abs(dt - 0.5) < 1e-9:
            if bundle.i_dt050 is None:
                raise ValueError(
                    f'd_model requires dt=0.5ms but TraceBundle.i_dt050 is None')
            R   = 2
            raw = bundle.i_dt050[inf_s * R : (inf_e + 1) * R]
            bl  = (np.full(len(raw), baseline_scalar)
                   if baseline_scalar is not None
                   else np.repeat(bundle.baseline[inf_s : inf_e + 1], R)[:len(raw)])
            seg = np.maximum(0.0, raw - bl)
            traces[dt] = _pad(seg, W_D * R)

        elif abs(dt - 0.1) < 1e-9:
            if bundle.i_dt010 is None:
                raise ValueError(
                    f'd_model requires dt=0.1ms but TraceBundle.i_dt010 is None')
            R   = 10
            raw = bundle.i_dt010[inf_s * R : (inf_e + 1) * R]
            bl  = (np.full(len(raw), baseline_scalar)
                   if baseline_scalar is not None
                   else np.repeat(bundle.baseline[inf_s : inf_e + 1], R)[:len(raw)])
            seg = np.maximum(0.0, raw - bl)
            traces[dt] = _pad(seg, W_D * R)

        else:
            raise ValueError(f'Unsupported dt channel in model: {dt}ms')

    return traces


def _edge_baseline(bundle: TraceBundle, t_left: int, t_right: int) -> float:
    """Mean of bundle.baseline at the left and right edges of a window."""
    return float(0.5 * (bundle.baseline[t_left] + bundle.baseline[t_right]))


def _predict_D(bundle: TraceBundle, tp: int, d_model,
               baseline_scalar: Optional[float] = None) -> float:
    """
    Call the D model on a fixed 4096ms window centred on tp.
    Returns D in µm²/s.
    """
    traces = _build_traces(bundle, tp, d_model, baseline_scalar=baseline_scalar)
    return d_model.predict(traces=traces, features=None)


def _predict_D_batch(
    bundle:           TraceBundle,
    tps:              list,
    d_model,
    baseline_scalars: Optional[list] = None,
    batch_size:       int = 128,
) -> list:
    """
    Batch D prediction for a list of peak positions.

    Builds trace windows for all tps, then calls predict_batch()
    so the GPU processes batch_size events per forward pass instead
    of one at a time.  Returns D values (µm²/s) in the same order
    as tps.

    baseline_scalars : optional list of per-event scalar baselines (same length
        as tps).  When provided, passed to _build_traces so that each event's
        inference uses a constant background anchored to its window edges.
    """
    if not tps:
        return []
    traces_list = []
    feats_list  = [] if d_model.requires_acf_features else None
    for i, tp in enumerate(tps):
        bl     = baseline_scalars[i] if baseline_scalars is not None else None
        traces = _build_traces(bundle, tp, d_model, baseline_scalar=bl)
        traces_list.append(traces)
        if feats_list is not None:
            feats_list.append(None)   # wrapper computes features internally
    return d_model.predict_batch(traces_list, feats_list, batch_size=batch_size)


# ── Step 1: peak finding + claiming ─────────────────────────────────────────

W12_MAX_MS = 2048   # hard cap on half-width used for Cat-1 seed window


def _step1(bundle: TraceBundle,
           amplitude_min_khz: Optional[float] = None,
           fold_min: Optional[float] = None,
           ) -> Tuple[List[dict], List[dict], np.ndarray, np.ndarray]:
    """
    Find peaks, classify, and claim seed windows.

    Parameters
    ----------
    amplitude_min_khz : absolute amplitude threshold in kHz (signal value,
        not baseline-subtracted).  If None, falls back to
        AMPLITUDE_FRAC_MIN * global_max.
    fold_min : minimum fold-over-baseline required to keep a peak.
        A peak at tp passes if
            (signal[tp] - baseline[tp]) / baseline[tp] >= fold_min.
        Applied in addition to amplitude_min_khz.  If None, not applied.

    Returns
    -------
    cat1     : surviving Cat 1 event seeds, sorted by h_peak desc
    cat2     : surviving Cat 2 event seeds, sorted by h_peak desc
    claimed1 : bool (N,) array — Cat-1 seed bins marked True
    claimed2 : bool (N,) array — Cat-2 seed bins marked True
    """
    N        = len(bundle.i1ms)
    signal   = bundle.i1ms
    claimed1 = np.zeros(N, dtype=bool)   # Cat-1 territory only
    claimed2 = np.zeros(N, dtype=bool)   # Cat-2 territory only

    # 1a. Find local maxima at or above baseline.
    # No distance constraint — all local maxima are kept and the claiming /
    # exclusion logic in steps 1d–1e collapses nearby shot-noise spikes.
    peak_idx, _ = find_peaks(
        signal,
        height = bundle.baseline,   # signal[tp] >= baseline[tp]
    )

    if len(peak_idx) == 0:
        return [], [], claimed1, claimed2

    # 1a'. Sort peaks by amplitude descending immediately after find_peaks.
    # Processing the highest-amplitude local maximum first ensures it claims
    # territory before any lower nearby local maxima, so tp_orig is always the
    # true intensity maximum in every neighbourhood — never a shot-noise artifact
    # that happened to satisfy the strict local-max criterion first.
    h_order  = np.argsort(signal[peak_idx])[::-1]
    peak_idx = peak_idx[h_order]

    # 1a''. Amplitude threshold
    h_arr = signal[peak_idx].astype(np.float64)
    if amplitude_min_khz is not None:
        threshold = float(amplitude_min_khz)
        print(f'  Amplitude filter (abs {threshold:.1f} kHz): '
              f'{(h_arr >= threshold).sum()}/{len(h_arr)} peaks kept')
    else:
        global_max = float(h_arr.max())
        threshold  = AMPLITUDE_FRAC_MIN * global_max
        print(f'  Amplitude filter ({AMPLITUDE_FRAC_MIN*100:.0f}% of '
              f'{global_max:.1f} kHz = {threshold:.1f} kHz): '
              f'{(h_arr >= threshold).sum()}/{len(h_arr)} peaks kept')
    peak_idx = peak_idx[h_arr >= threshold]

    if len(peak_idx) == 0:
        return [], [], claimed1, claimed2

    # 1a''. Fold-over-baseline threshold
    if fold_min is not None:
        bl_arr   = bundle.baseline[peak_idx]
        fold_arr = (signal[peak_idx] - bl_arr) / np.maximum(bl_arr, 1e-9)
        keep     = fold_arr >= float(fold_min)
        print(f'  Fold filter (>={fold_min:.2f}× baseline): '
              f'{keep.sum()}/{len(peak_idx)} peaks kept')
        peak_idx = peak_idx[keep]

    if len(peak_idx) == 0:
        return [], [], claimed1, claimed2

    # 1b–c. Compute w12 and classify
    # w12 is measured on a box-smoothed trace to avoid noisy width estimates.
    # W12_MAX_MS caps the seed window so one broad event cannot claim too large
    # a territory and suppress many neighbouring real events.
    kernel        = np.ones(W12_SMOOTH_MS, dtype=np.float64) / W12_SMOOTH_MS
    signal_smooth = np.convolve(signal, kernel, 'same')

    cat1_raw: List[dict] = []
    cat2_raw: List[dict] = []
    for pid, tp in enumerate(peak_idx):
        h_peak = float(signal[tp])
        w12    = min(_compute_w12(signal_smooth, int(tp), bundle.baseline), W12_MAX_MS)
        entry  = dict(id=pid, tp=int(tp), h_peak=h_peak, w12=w12)
        if 2 * w12 >= CAT1_MIN_WIDTH_MS:
            cat1_raw.append(entry)
        else:
            cat2_raw.append(entry)

    # Sort both lists by h_peak descending before claiming loops
    cat1_raw.sort(key=lambda e: e['h_peak'], reverse=True)
    cat2_raw.sort(key=lambda e: e['h_peak'], reverse=True)

    print(f'  Peak finder: {len(cat1_raw)} Cat-1  |  {len(cat2_raw)} Cat-2')

    excluded1: set = set()   # Cat-1 exclusions only
    excluded2: set = set()   # Cat-2 exclusions only

    # 1d. Cat 1 claiming loop (descending h_peak)
    # Cat-1 events are mutually exclusive with each other only; Cat-2 is
    # processed independently and may overlap any Cat-1 window.
    for ev in cat1_raw:
        if ev['id'] in excluded1:
            continue
        tp, w12 = ev['tp'], ev['w12']
        cl_s = max(0,     tp - w12)
        cl_e = min(N - 1, tp + w12)

        claimed1[cl_s : cl_e + 1] = True
        ev['t_start'] = cl_s
        ev['t_end']   = cl_e
        # Exclude Cat-1 peaks whose seed window OVERLAPS with this event's seed
        # window, as long as they have not already been processed.  Checking
        # window overlap (rather than just whether tp falls inside) prevents two
        # neighbouring events with interleaved seeds from permanently blocking
        # each other's expansion in Step 2.
        # Cat-2 peaks are never touched here — they are processed independently.
        for other in cat1_raw:
            if other['id'] != ev['id'] and 't_start' not in other:
                other_cl_s = max(0,     other['tp'] - other['w12'])
                other_cl_e = min(N - 1, other['tp'] + other['w12'])
                if cl_s <= other_cl_e and other_cl_s <= cl_e:   # windows overlap
                    excluded1.add(other['id'])

    # Clean up after Cat 1 loop — Cat-2 is completely unaffected
    cat1 = [e for e in cat1_raw if e['id'] not in excluded1 and 't_start' in e]
    cat2 = list(cat2_raw)   # Cat-2 is independent; never excluded by Cat-1

    # 1e. Cat 2 claiming loop (descending h_peak)
    # Pre-build numpy arrays for vectorised range queries; a dict maps id → entry
    # for O(1) 't_start' lookup in the guard below.
    half         = W_START2_MS // 2
    cat2_by_id   = {e['id']: e for e in cat2}
    cat2_tp_arr  = np.array([e['tp'] for e in cat2], dtype=np.int64)
    cat2_id_arr  = np.array([e['id'] for e in cat2], dtype=np.int64)

    for ev in cat2:
        if ev['id'] in excluded2:
            continue
        tp   = ev['tp']
        cl_s = max(0,     tp - half)
        cl_e = min(N - 1, tp + half)
        claimed2[cl_s : cl_e + 1] = True
        ev['t_start'] = cl_s
        ev['t_end']   = cl_e
        # Exclude Cat-2 peaks whose tp falls inside this window, skipping any
        # that were already processed (have 't_start') — same guard as Cat-1.
        mask = (cat2_tp_arr >= cl_s) & (cat2_tp_arr <= cl_e) & (cat2_id_arr != ev['id'])
        for i in cat2_id_arr[mask]:
            other_c2 = cat2_by_id[int(i)]
            if 't_start' not in other_c2:
                # and other_c2['h_peak'] <= ev['h_peak']:  # amplitude guard removed in v3.5
                excluded2.add(int(i))

    # Clean up after Cat 2 loop
    cat2 = [e for e in cat2 if e['id'] not in excluded2 and 't_start' in e]

    # 1f. Sort by h_peak descending
    cat1.sort(key=lambda e: e['h_peak'], reverse=True)
    cat2.sort(key=lambda e: e['h_peak'], reverse=True)

    print(f'  After claiming: {len(cat1)} Cat-1  |  {len(cat2)} Cat-2')
    return cat1, cat2, claimed1, claimed2


# ── Step 2: D-guided window expansion ────────────────────────────────────────

def _step2(bundle: TraceBundle,
           seed_events: List[dict],
           claimed_self: np.ndarray,
           d_model,
           category: int) -> List[dict]:
    """
    Window-doubling expansion guided by the D model.
    Modifies `claimed_self` in-place.

    Parameters
    ----------
    bundle       : TraceBundle
    seed_events  : list of seed dicts from _step1 (have t_start, t_end, tp)
    claimed_self : bool (N,) array for THIS category only — updated as windows
                   expand.  Expansion stops when it would enter territory already
                   owned by another event of the SAME category; territory owned
                   by the opposite category (Cat-1 vs Cat-2) is transparent and
                   does not block expansion.  This allows a slow-diffuser Cat-1
                   event and a fast-diffuser Cat-2 event to have overlapping
                   windows, reflecting the physical possibility of two different
                   molecular species co-occurring in the focal volume.
    d_model     : ModelWrapper — loaded D-inference model
    category    : 1 or 2 — passed through to output dicts

    Returns
    -------
    list of finalised event dicts
    """
    N       = len(bundle.i1ms)
    results = []

    if not seed_events:
        return results

    # Batch the initial D predictions for all seed events in one GPU call
    # (significant speedup vs. sequential one-at-a-time calls).
    # Baseline is anchored to the edges of each seed window.
    tps_all          = [ev['tp'] for ev in seed_events]
    bl_scalars_init  = [_edge_baseline(bundle, ev['t_start'], ev['t_end'])
                        for ev in seed_events]
    initial_d_hats   = _predict_D_batch(bundle, tps_all, d_model,
                                        baseline_scalars=bl_scalars_init)

    for ev, D_hat in zip(seed_events, initial_d_hats):
        tp      = ev['tp']
        t_start = ev['t_start']
        t_end   = ev['t_end']
        w       = max(t_end - t_start, 1)

        # Cat-1 events always expand — D-guided tau_max and fast-group
        # classification are bypassed.  Only a territorial collision with
        # another Cat-1 event (or W_MAX_MS) stops the expansion.
        # Cat-2 events retain the full D-guided stopping logic.
        fast_group = False if category == 1 else (D_hat > D_FAST_CUTOFF)

        # Expansion history — index 0 = seed, index k = after k-th doubling
        t_left_hist  = [t_start]
        t_right_hist = [t_end]
        d_hist       = [float(D_hat)]

        if not fast_group:
            tau_max = _lookup_tau_max(D_hat)   # used only for Cat-2

            while w < W_MAX_MS:
                w_new  = 2 * w
                ts_new = max(0,     tp - w_new // 2)
                te_new = min(N - 1, tp + w_new // 2)

                # Stop 1 (both categories): W_MAX reached OR territory already
                # claimed by another event of the SAME category.
                if (w_new >= W_MAX_MS
                        or claimed_self[ts_new : t_start].any()
                        or claimed_self[t_end + 1 : te_new + 1].any()):
                    break

                # Stop 2 (Cat-2 only): transit time for current D range reached.
                # tau_max is the MAXIMUM ALLOWED window width, so expansion
                # must be permitted to reach exactly tau_max (strict >).
                if category != 1 and w_new > tau_max:
                    break

                # Accept expansion
                claimed_self[ts_new   : t_start]      = True
                claimed_self[t_end + 1 : te_new + 1]  = True
                t_start, t_end = ts_new, te_new
                w = w_new
                t_left_hist.append(t_start)
                t_right_hist.append(t_end)

                # Re-predict D on the expanded window; baseline anchored to
                # the new window edges (t_start/t_end updated above).
                D_hat = _predict_D(bundle, tp, d_model,
                                   baseline_scalar=_edge_baseline(
                                       bundle, t_start, t_end))
                d_hist.append(float(D_hat))
                # Cat-2 only: re-classify as fast if D crosses threshold
                if category != 1:
                    if D_hat > D_FAST_CUTOFF:
                        fast_group = True
                        break
                    tau_max = _lookup_tau_max(D_hat)

        # Refine t_peak to the global maximum of the raw trace within the
        # final window.  For Cat-2 events this is nearly always the original
        # detected tp; for broad Cat-1 events the detected tp is a shot-noise
        # local maximum that may be well offset from the true transit maximum.
        tp = int(t_start + np.argmax(bundle.i1ms[t_start : t_end + 1]))

        # Finalise event
        dur   = t_end - t_start + 1
        W_seg = dur // 2
        pf    = max(0.0, (W_D - dur) / W_D) if dur < W_D else 0.0
        nw    = 1 if dur <= W_D else max(1, (dur - W_D) // (W_D // 8) + 1)
        score = float((bundle.i1ms[tp] - bundle.baseline[tp])
                      / np.sqrt(max(bundle.baseline[tp], 1e-9)))

        results.append(dict(
            t_peak       = tp,
            t_left       = t_start,
            t_right      = t_end,
            W_seg        = W_seg,
            D_hat        = float(D_hat),
            log10_D      = float(np.log10(max(D_hat, 1e-10))),
            category     = category,
            fast_group   = fast_group,
            score        = score,
            pad_frac     = pf,
            n_win        = nw,
            k_win        = 0,
            h_peak       = float(bundle.i1ms[tp]),
            w12_seed     = ev.get('w12'),
            t_left_hist  = t_left_hist,
            t_right_hist = t_right_hist,
            d_hist       = d_hist,
        ))

    return results


# ── Post-expansion absorption merge ─────────────────────────────────────────

def _post_absorption_merge(events: List[dict]) -> List[dict]:
    """
    Absorb lower-amplitude events that were prevented from being claimed by a
    larger neighbour during Step 1, but whose t_peak falls inside that
    neighbour's D-guided maximum transit window.

    Algorithm
    ---------
    Sort events by h_peak (raw amplitude) descending.
    For each surviving event A, compute its D-guided half-window:
        tau_half = min(tau_max(D_hat_A), W_MAX_MS) // 2
    Any event B with lower amplitude whose t_peak lies in
        [A.t_peak − tau_half,  A.t_peak + tau_half]
    is absorbed into A:
        A.t_left  = min(A.t_left,  B.t_left)
        A.t_right = max(A.t_right, B.t_right)
    B is removed.  Absorption cascades naturally because A's window grows
    and each inner loop re-checks the updated boundaries against fixed tau_half.
    """
    if len(events) <= 1:
        return events

    events = sorted(events, key=lambda e: e['h_peak'], reverse=True)
    absorbed: set = set()

    for i, A in enumerate(events):
        if i in absorbed:
            continue
        tau_half = min(_lookup_tau_max(A['D_hat']), W_MAX_MS) // 2
        centre   = A['t_peak']

        for j, B in enumerate(events):
            if j == i or j in absorbed:
                continue
            if B['h_peak'] >= A['h_peak']:
                continue
            if centre - tau_half <= B['t_peak'] <= centre + tau_half:
                A['t_left']  = min(A['t_left'],  B['t_left'])
                A['t_right'] = max(A['t_right'], B['t_right'])
                absorbed.add(j)

    result = [e for i, e in enumerate(events) if i not in absorbed]
    result.sort(key=lambda e: e['t_peak'])
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def run_stage3_v2(bundle: TraceBundle, d_model, radius_cap: int = 128,
                  amplitude_min_khz: Optional[float] = None,
                  fold_min: Optional[float] = None):
    """
    Run Stage 3 v2 event detection on a TraceBundle.

    Parameters
    ----------
    bundle     : TraceBundle — multi-resolution signal container
    d_model    : ModelWrapper — loaded diffusion model
    radius_cap : unused; kept for backward compatibility with callers

    Returns
    -------
    events  : list of dicts — detected events sorted by t_peak
    z_score : ndarray (N,) — Poisson z-score for visualisation,
              z = (i1ms - baseline) / sqrt(baseline)
    """
    N = len(bundle.i1ms)
    if N == 0:
        return [], np.array([], dtype=np.float64)

    # Visualisation z-score (Poisson: σ = √baseline)
    z_score = ((bundle.i1ms - bundle.baseline)
               / np.sqrt(np.maximum(bundle.baseline, 1e-9)))

    # Step 1 — peak finding + claiming
    # Returns separate claimed arrays so each category's expansion only blocks
    # events of the same category (Cat-1 ↔ Cat-2 overlap is permitted).
    cat1, cat2, claimed1, claimed2 = _step1(bundle,
                                            amplitude_min_khz=amplitude_min_khz,
                                            fold_min=fold_min)
    if not cat1 and not cat2:
        return [], z_score

    # Step 2 — D-guided expansion (category-local blocking)
    events  = _step2(bundle, cat1, claimed1, d_model, category=1)
    events += _step2(bundle, cat2, claimed2, d_model, category=2)

    # Post-expansion absorption merge:
    # High-amplitude events absorb lower-amplitude neighbours whose t_peak
    # falls within the larger event's D-guided transit window.
    n_before = len(events)
    events   = _post_absorption_merge(events)
    n_after  = len(events)
    if n_before != n_after:
        print(f'  Absorption merge: {n_before} → {n_after} events '
              f'({n_before - n_after} absorbed)')

    events.sort(key=lambda e: e['t_peak'])
    return events, z_score
