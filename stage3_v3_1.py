"""
stage3_v3_1.py — Stage 3 event detection pipeline, version 3.1
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

    I_cut = baseline[tp] + TRNST_LVL * (signal[tp] - baseline[tp])

    If baseline is None, falls back to TRNST_LVL * signal[tp].

    t_left_w  = last index < tp where signal < I_cut  (or 0 if none)
    t_right_w = first index > tp where signal < I_cut (or N-1 if none)
    w12       = (t_right_w - t_left_w) // 2
    """
    if baseline is not None:
        bl    = float(baseline[tp])
        I_cut = bl + TRNST_LVL * max(float(signal[tp]) - bl, 0.0)
    else:
        I_cut = TRNST_LVL * signal[tp]
    N     = len(signal)

    left_idx  = np.where(signal[:tp] < I_cut)[0]
    t_left_w  = int(left_idx[-1]) if len(left_idx)  > 0 else 0

    right_idx = np.where(signal[tp + 1:] < I_cut)[0]
    t_right_w = int(right_idx[0]) + tp + 1 if len(right_idx) > 0 else N - 1

    return max((t_right_w - t_left_w) // 2, 1)


def _build_traces(bundle: TraceBundle, tp: int, d_model) -> dict:
    """
    Build padded inference windows of exactly W_D ms centred on tp,
    one per dt channel required by d_model.

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
            seg = np.maximum(0.0,
                             bundle.i1ms[inf_s: inf_e + 1]
                             - bundle.baseline[inf_s: inf_e + 1])
            traces[dt] = _pad(seg, W_D)

        elif abs(dt - 0.5) < 1e-9:
            if bundle.i_dt050 is None:
                raise ValueError(
                    f'd_model requires dt=0.5ms but TraceBundle.i_dt050 is None')
            R       = 2
            raw     = bundle.i_dt050[inf_s * R : (inf_e + 1) * R]
            bl_up   = np.repeat(bundle.baseline[inf_s : inf_e + 1], R)[:len(raw)]
            seg     = np.maximum(0.0, raw - bl_up)
            traces[dt] = _pad(seg, W_D * R)

        elif abs(dt - 0.1) < 1e-9:
            if bundle.i_dt010 is None:
                raise ValueError(
                    f'd_model requires dt=0.1ms but TraceBundle.i_dt010 is None')
            R       = 10
            raw     = bundle.i_dt010[inf_s * R : (inf_e + 1) * R]
            bl_up   = np.repeat(bundle.baseline[inf_s : inf_e + 1], R)[:len(raw)]
            seg     = np.maximum(0.0, raw - bl_up)
            traces[dt] = _pad(seg, W_D * R)

        else:
            raise ValueError(f'Unsupported dt channel in model: {dt}ms')

    return traces


def _predict_D(bundle: TraceBundle, tp: int, d_model) -> float:
    """
    Call the D model on a fixed 4096ms window centred on tp.
    Returns D in µm²/s.
    """
    traces = _build_traces(bundle, tp, d_model)
    return d_model.predict(traces=traces, features=None)


def _predict_D_batch(
    bundle:     TraceBundle,
    tps:        list,
    d_model,
    batch_size: int = 128,
) -> list:
    """
    Batch D prediction for a list of peak positions.

    Builds trace windows for all tps, then calls predict_batch()
    so the GPU processes batch_size events per forward pass instead
    of one at a time.  Returns D values (µm²/s) in the same order
    as tps.
    """
    if not tps:
        return []
    traces_list = []
    feats_list  = [] if d_model.requires_acf_features else None
    for tp in tps:
        traces = _build_traces(bundle, tp, d_model)
        traces_list.append(traces)
        if feats_list is not None:
            feats_list.append(None)   # wrapper computes features internally
    return d_model.predict_batch(traces_list, feats_list, batch_size=batch_size)


# ── Step 1: peak finding + claiming ─────────────────────────────────────────

W12_MAX_MS = 2048   # hard cap on half-width used for Cat-1 seed window


def _step1(bundle: TraceBundle,
           amplitude_min_khz: Optional[float] = None,
           fold_min: Optional[float] = None,
           ) -> Tuple[List[dict], List[dict], np.ndarray]:
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
    cat1    : surviving Cat 1 event seeds, sorted by h_peak desc
    cat2    : surviving Cat 2 event seeds, sorted by h_peak desc
    claimed : bool (N,) array with seed bins marked True
    """
    N       = len(bundle.i1ms)
    signal  = bundle.i1ms
    claimed = np.zeros(N, dtype=bool)

    # 1a. Find local maxima at or above baseline
    peak_idx, _ = find_peaks(
        signal,
        height   = bundle.baseline,   # signal[tp] >= baseline[tp]
        distance = PEAK_DISTANCE_MS,
    )

    if len(peak_idx) == 0:
        return [], [], claimed

    # 1a'. Amplitude threshold
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
        return [], [], claimed

    # 1a''. Fold-over-baseline threshold
    if fold_min is not None:
        bl_arr   = bundle.baseline[peak_idx]
        fold_arr = (signal[peak_idx] - bl_arr) / np.maximum(bl_arr, 1e-9)
        keep     = fold_arr >= float(fold_min)
        print(f'  Fold filter (>={fold_min:.2f}× baseline): '
              f'{keep.sum()}/{len(peak_idx)} peaks kept')
        peak_idx = peak_idx[keep]

    if len(peak_idx) == 0:
        return [], [], claimed

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

    excluded: set = set()

    # 1d. Cat 1 claiming loop (descending h_peak)
    for ev in cat1_raw:
        if ev['id'] in excluded:
            continue
        tp, w12 = ev['tp'], ev['w12']
        cl_s = max(0,     tp - w12)
        cl_e = min(N - 1, tp + w12)
        claimed[cl_s : cl_e + 1] = True
        ev['t_start'] = cl_s
        ev['t_end']   = cl_e
        # Exclude all peaks (cat1 + cat2) whose tp falls inside this window
        for other in cat1_raw + cat2_raw:
            if other['id'] != ev['id'] and cl_s <= other['tp'] <= cl_e:
                excluded.add(other['id'])

    # Clean up after Cat 1 loop
    cat1 = [e for e in cat1_raw if e['id'] not in excluded and 't_start' in e]
    cat2 = [e for e in cat2_raw if e['id'] not in excluded]

    # 1e. Cat 2 claiming loop (descending h_peak)
    # Pre-build numpy arrays so the inner exclusion scan is vectorised
    # instead of O(n²) in pure Python.
    half        = W_START2_MS // 2
    cat2_tp_arr = np.array([e['tp'] for e in cat2], dtype=np.int64)
    cat2_id_arr = np.array([e['id'] for e in cat2], dtype=np.int64)

    for ev in cat2:
        if ev['id'] in excluded:
            continue
        tp   = ev['tp']
        cl_s = max(0,     tp - half)
        cl_e = min(N - 1, tp + half)
        claimed[cl_s : cl_e + 1] = True
        ev['t_start'] = cl_s
        ev['t_end']   = cl_e
        # Exclude other Cat 2 peaks whose tp falls inside this window
        mask = (cat2_tp_arr >= cl_s) & (cat2_tp_arr <= cl_e) & (cat2_id_arr != ev['id'])
        excluded.update(int(i) for i in cat2_id_arr[mask])

    # Clean up after Cat 2 loop
    cat2 = [e for e in cat2 if e['id'] not in excluded and 't_start' in e]

    # 1f. Sort by h_peak descending
    cat1.sort(key=lambda e: e['h_peak'], reverse=True)
    cat2.sort(key=lambda e: e['h_peak'], reverse=True)

    print(f'  After claiming: {len(cat1)} Cat-1  |  {len(cat2)} Cat-2')
    return cat1, cat2, claimed


# ── Step 2: D-guided window expansion ────────────────────────────────────────

def _step2(bundle: TraceBundle,
           seed_events: List[dict],
           claimed: np.ndarray,
           d_model,
           category: int) -> List[dict]:
    """
    Window-doubling expansion guided by the D model.
    Modifies `claimed` in-place.

    Parameters
    ----------
    bundle      : TraceBundle
    seed_events : list of seed dicts from _step1 (have t_start, t_end, tp)
    claimed     : bool (N,) array — updated as windows expand
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
    tps_all          = [ev['tp'] for ev in seed_events]
    initial_d_hats   = _predict_D_batch(bundle, tps_all, d_model)

    for ev, D_hat in zip(seed_events, initial_d_hats):
        tp      = ev['tp']
        t_start = ev['t_start']
        t_end   = ev['t_end']
        w       = max(t_end - t_start, 1)

        fast_group = D_hat > D_FAST_CUTOFF

        if not fast_group:
            tau_max = _lookup_tau_max(D_hat)

            while w < W_MAX_MS:
                w_new  = 2 * w
                ts_new = max(0,     tp - w_new // 2)
                te_new = min(N - 1, tp + w_new // 2)

                # Stop 1: W_MAX reached OR new territory already claimed
                if (w_new >= W_MAX_MS
                        or claimed[ts_new : t_start].any()
                        or claimed[t_end + 1 : te_new + 1].any()):
                    break

                # Stop 2: transit time for current D range reached.
                # tau_max is the MAXIMUM ALLOWED window width, so expansion
                # must be permitted to reach exactly tau_max (strict >).
                if w_new > tau_max:
                    break

                # Accept expansion
                claimed[ts_new   : t_start]      = True
                claimed[t_end + 1 : te_new + 1]  = True
                t_start, t_end = ts_new, te_new
                w = w_new

                # Re-predict D on the expanded window; update tau_max
                D_hat = _predict_D(bundle, tp, d_model)
                if D_hat > D_FAST_CUTOFF:
                    fast_group = True
                    break
                tau_max = _lookup_tau_max(D_hat)

        # Finalise event
        dur   = t_end - t_start + 1
        W_seg = dur // 2
        pf    = max(0.0, (W_D - dur) / W_D) if dur < W_D else 0.0
        nw    = 1 if dur <= W_D else max(1, (dur - W_D) // (W_D // 8) + 1)
        score = float((bundle.i1ms[tp] - bundle.baseline[tp])
                      / np.sqrt(max(bundle.baseline[tp], 1e-9)))

        results.append(dict(
            t_peak     = tp,
            t_left     = t_start,
            t_right    = t_end,
            W_seg      = W_seg,
            D_hat      = float(D_hat),
            log10_D    = float(np.log10(max(D_hat, 1e-10))),
            category   = category,
            fast_group = fast_group,
            score      = score,
            pad_frac   = pf,
            n_win      = nw,
            k_win      = 0,
            h_peak     = float(bundle.i1ms[tp]),
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
    cat1, cat2, claimed = _step1(bundle, amplitude_min_khz=amplitude_min_khz,
                                 fold_min=fold_min)
    if not cat1 and not cat2:
        return [], z_score

    # Step 2 — D-guided expansion
    events  = _step2(bundle, cat1, claimed, d_model, category=1)
    events += _step2(bundle, cat2, claimed, d_model, category=2)

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
