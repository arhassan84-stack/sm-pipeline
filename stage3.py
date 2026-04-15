"""
stage3.py — Stage 3 event detection pipeline
═════════════════════════════════════════════
Detects single-molecule dwell events in a background-subtracted FCS
intensity trace via a local z-score step detector.

This replaces the earlier matched-filter approach, which was designed for
freely-diffusing (solution) FCS and gave poor results on membrane proteins
(ITGA5/ITGB1) where molecules dwell — producing sustained intensity plateaus
rather than Gaussian transit bursts.

Algorithm
---------
1. Rolling baseline
     median_filter(signal, W_BASELINE_MS) estimates the slowly-drifting
     cellular background without being pulled up by brief bright events.

2. Local noise (robust)
     noise(t) = 1.4826 × median_filter(|signal − baseline|, W_BASELINE_MS)
     (MAD-based standard-deviation estimate; 1.4826 converts MAD → σ for
      a Gaussian distribution.)
     A hard floor NOISE_FLOOR_KHZ prevents division-by-zero.

3. Local z-score
     z(t) = (signal(t) − baseline(t)) / noise(t)

4. Connected-region labelling
     Bins where z ≥ Z_THRESHOLD form candidate event segments.
     Gaps < GAP_FILL_MS between adjacent regions are closed (filled in) to
     avoid splitting a single molecule dwell that briefly dips below threshold.

5. Duration filter
     Regions shorter than MIN_DURATION_MS are discarded (likely noise spikes).

6. Amplitude filter
     Events whose peak signal is below MIN_PEAK_FRAC × the brightest event's
     peak are discarded (per-channel filter; applied after step 5).

7. Output
     One event dict per surviving region, sorted by t_peak.

Output dict keys
----------------
    t_peak    int   — index of maximum intensity within the event
    t_left    int   — segment left boundary (ms index, inclusive)
    t_right   int   — segment right boundary (ms index, inclusive)
    W_seg     int   — half-width of the event = (t_right − t_left) // 2
    k_win     int   — always 0 (no template bank; kept for pipeline compat.)
    score     float — peak z-score within the event
    pad_frac  float — fraction of W_D that would be zero-padded
    n_win     int   — number of W_D sliding windows in segment (≥1)
"""

import numpy as np
from scipy.ndimage import median_filter, label
from pathlib import Path

# ── paths (kept for backward compat; burst_templates.npz no longer used) ───────
BASE     = Path(__file__).parent
TPL_FILE = BASE / 'burst_templates.npz'

# ── detection parameters ────────────────────────────────────────────────────────
W_D              = 4096   # D-model input window length (ms)

Z_THRESHOLD      = 2.5    # z-score threshold for event detection
W_BASELINE_MS    = 10000  # rolling baseline / noise window half-width (ms)
                           # must be >> expected single-event duration
NOISE_FLOOR_KHZ  = 0.5    # hard floor on the local noise estimate (kHz)
GAP_FILL_MS      = 500    # close intra-event gaps shorter than this (ms)
MIN_DURATION_MS  = 200    # discard events shorter than this (ms)
MIN_PEAK_FRAC    = 0.25   # keep events ≥ this fraction of the channel's
                           # brightest surviving event peak (per-channel)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _rolling_baseline_noise(signal: np.ndarray):
    """
    Compute a robust rolling baseline and noise estimate.

    Returns
    -------
    baseline : (N,) float — local median of the signal
    noise    : (N,) float — local MAD-based noise std (≥ NOISE_FLOOR_KHZ)
    """
    W = min(W_BASELINE_MS, max(3, len(signal) // 4))
    baseline = median_filter(signal, size=W, mode='reflect')
    residual = signal - baseline
    mad      = median_filter(np.abs(residual), size=W, mode='reflect')
    noise    = np.maximum(1.4826 * mad, NOISE_FLOOR_KHZ)
    return baseline, noise


def _fill_gaps(mask: np.ndarray, gap_ms: int) -> np.ndarray:
    """
    Close short gaps (< gap_ms bins) within a boolean mask.
    Prevents single-molecule dwells that briefly dip below Z_THRESHOLD
    from being split into multiple events.
    """
    mask = mask.copy()
    above = np.where(mask)[0]
    if len(above) < 2:
        return mask
    gaps_start = np.where(np.diff(above) > 1)[0]
    for gs in gaps_start:
        lo, hi = above[gs], above[gs + 1]
        if hi - lo - 1 < gap_ms:
            mask[lo:hi + 1] = True
    return mask


# ── public API ──────────────────────────────────────────────────────────────────

def run_stage3(signal, radius_cap: int = 128):
    """
    Run the full Stage 3 detection pipeline on a single intensity trace.

    Parameters
    ----------
    signal     : array-like, shape (N,)
                 Background-subtracted intensity (kHz, 1 ms bins).
    radius_cap : int
                 Unused — kept for backward compatibility with callers.

    Returns
    -------
    events   : list of dicts  — detected events sorted by t_peak
    z_score  : ndarray (N,)   — local z-score trace (for visualisation)
    """
    signal = np.asarray(signal, dtype=np.float64)
    N      = len(signal)

    if N == 0:
        return [], np.array([], dtype=np.float64)

    # ── 1-2. Baseline and noise ────────────────────────────────────────────────
    baseline, noise = _rolling_baseline_noise(signal)
    z_score = (signal - baseline) / noise

    # ── 3. Threshold and gap-fill ─────────────────────────────────────────────
    above = z_score >= Z_THRESHOLD
    above = _fill_gaps(above, GAP_FILL_MS)

    # ── 4. Label connected regions ────────────────────────────────────────────
    labeled, n_regions = label(above)

    # ── 5-6. Build events ─────────────────────────────────────────────────────
    events = []
    for region_id in range(1, n_regions + 1):
        indices = np.where(labeled == region_id)[0]
        t_l, t_r = int(indices[0]), int(indices[-1])
        dur = t_r - t_l + 1

        if dur < MIN_DURATION_MS:
            continue

        t_peak = int(indices[np.argmax(signal[indices])])
        score  = float(z_score[t_peak])
        W_seg  = dur // 2
        L      = dur
        pf     = max(0.0, (W_D - L) / W_D) if L < W_D else 0.0
        nw     = 1 if L <= W_D else max(1, (L - W_D) // (W_D // 8) + 1)

        events.append(dict(
            t_peak   = t_peak,
            t_left   = t_l,
            t_right  = t_r,
            W_seg    = W_seg,
            k_win    = 0,
            score    = score,
            pad_frac = pf,
            n_win    = nw,
        ))

    # ── 7. Per-channel amplitude filter ───────────────────────────────────────
    if events and MIN_PEAK_FRAC > 0:
        best = max(float(signal[ev['t_peak']]) for ev in events)
        events = [ev for ev in events
                  if float(signal[ev['t_peak']]) >= MIN_PEAK_FRAC * best]

    events.sort(key=lambda e: e['t_peak'])
    return events, z_score
