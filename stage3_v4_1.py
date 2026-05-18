"""
stage3_v4_1.py — Stage 3 event detection pipeline, version 4.1
═════════════════════════════════════════════════════════════

Identical to stage3_v4.py with one change:

1. BL_FACTORS = [1.05, 1.10, 1.15, 1.20, 1.50, 2.00]
   At every prediction step (seed + each accepted expansion), D is also
   predicted with each factor applied to the edge baseline:
       baseline_scalar_f = edge_bl * factor
   The factor predictions are stored in the event dict but do NOT affect
   the expansion stopping criterion (which continues to use factor=1.0).

2. Event dict gains:
       d_hat_by_factor  : {factor: D_final}
       d_hist_by_factor : {factor: [D_step0, D_step1, ...]}

All other logic is unchanged from v3.95.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.signal import find_peaks

# ── Detection parameters (unchanged from v3.95) ───────────────────────────────
W_D               = 4096
TRNST_LVL         = 0.30
CAT1_MIN_WIDTH_MS = 200
W_START2_MS       = 128
W_MAX_MS          = 4096
D_FAST_CUTOFF     = 1.00
PEAK_DISTANCE_MS  = 50
AMPLITUDE_FRAC_MIN = 0.50
W12_SMOOTH_MS      = 50

D_CUTOFFS     = [0.02, 0.06, 0.20, 0.50, 1.00]
TRANSIT_TIMES = [2048, 1024,  512,  256,  128]

# ── v4.0: baseline subtraction factors ───────────────────────────────────────
BL_FACTORS = [1.05, 1.10, 1.15, 1.20, 1.50, 2.00]


# ── TraceBundle ───────────────────────────────────────────────────────────────

@dataclass
class TraceBundle:
    i1ms:    np.ndarray
    baseline: object
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lookup_tau_max(D_hat: float) -> int:
    for cutoff, tmax in zip(D_CUTOFFS, TRANSIT_TIMES):
        if D_hat <= cutoff:
            return tmax
    return TRANSIT_TIMES[-1]


def _compute_w12(signal: np.ndarray, tp: int,
                 baseline: Optional[np.ndarray] = None) -> Tuple[int, Optional[float]]:
    bl_min: Optional[float]
    if baseline is not None:
        lo    = max(0, tp - W12_MAX_MS)
        hi    = min(len(baseline) - 1, tp + W12_MAX_MS)
        bl    = float(baseline[lo : hi + 1].min())
        bl_min = bl
        I_cut = bl + TRNST_LVL * max(float(signal[tp]) - bl, 0.0)
    else:
        bl_min = None
        I_cut  = TRNST_LVL * signal[tp]
    N = len(signal)

    left_idx  = np.where(signal[:tp] < I_cut)[0]
    t_left_w  = int(left_idx[-1]) if len(left_idx)  > 0 else 0

    right_idx = np.where(signal[tp + 1:] < I_cut)[0]
    t_right_w = int(right_idx[0]) + tp + 1 if len(right_idx) > 0 else N - 1

    return max((t_right_w - t_left_w) // 2, 1), bl_min


def _build_traces(bundle: TraceBundle, tp: int, d_model,
                  baseline_scalar: Optional[float] = None) -> dict:
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
            bl  = (np.full(inf_e + 1 - inf_s, baseline_scalar)
                   if baseline_scalar is not None
                   else bundle.baseline[inf_s : inf_e + 1])
            seg = np.maximum(0.0, bundle.i1ms[inf_s : inf_e + 1] - bl)
            traces[dt] = _pad(seg, W_D)

        elif abs(dt - 0.5) < 1e-9:
            if bundle.i_dt050 is None:
                raise ValueError('d_model requires dt=0.5ms but TraceBundle.i_dt050 is None')
            R   = 2
            raw = bundle.i_dt050[inf_s * R : (inf_e + 1) * R]
            bl  = (np.full(len(raw), baseline_scalar)
                   if baseline_scalar is not None
                   else np.repeat(bundle.baseline[inf_s : inf_e + 1], R)[:len(raw)])
            seg = np.maximum(0.0, raw - bl)
            traces[dt] = _pad(seg, W_D * R)

        elif abs(dt - 0.1) < 1e-9:
            if bundle.i_dt010 is None:
                raise ValueError('d_model requires dt=0.1ms but TraceBundle.i_dt010 is None')
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
    return float(min(bundle.baseline[t_left], bundle.baseline[t_right]))


def _predict_D(bundle: TraceBundle, tp: int, d_model,
               baseline_scalar: Optional[float] = None) -> float:
    traces = _build_traces(bundle, tp, d_model, baseline_scalar=baseline_scalar)
    return d_model.predict(traces=traces, features=None)


def _predict_D_batch(
    bundle:           TraceBundle,
    tps:              list,
    d_model,
    baseline_scalars: Optional[list] = None,
    batch_size:       int = 128,
) -> list:
    if not tps:
        return []
    traces_list = []
    feats_list  = [] if d_model.requires_acf_features else None
    for i, tp in enumerate(tps):
        bl     = baseline_scalars[i] if baseline_scalars is not None else None
        traces = _build_traces(bundle, tp, d_model, baseline_scalar=bl)
        traces_list.append(traces)
        if feats_list is not None:
            feats_list.append(None)
    return d_model.predict_batch(traces_list, feats_list, batch_size=batch_size)


# ── Step 1 (unchanged from v3.95) ─────────────────────────────────────────────

W12_MAX_MS = 2048


def _step1(bundle: TraceBundle,
           amplitude_min_khz: Optional[float] = None,
           fold_min: Optional[float] = None,
           ) -> Tuple[List[dict], List[dict], np.ndarray, np.ndarray]:
    N        = len(bundle.i1ms)
    signal   = bundle.i1ms
    owner1   = np.full(N, -1, dtype=np.int32)
    claimed2 = np.zeros(N, dtype=bool)

    peak_idx, _ = find_peaks(signal, height=bundle.baseline)
    if len(peak_idx) == 0:
        return [], [], owner1, claimed2

    h_order  = np.argsort(signal[peak_idx])[::-1]
    peak_idx = peak_idx[h_order]

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
        return [], [], owner1, claimed2

    if fold_min is not None:
        bl_arr   = bundle.baseline[peak_idx]
        fold_arr = (signal[peak_idx] - bl_arr) / np.maximum(bl_arr, 1e-9)
        keep     = fold_arr >= float(fold_min)
        print(f'  Fold filter (>={fold_min:.2f}× baseline): '
              f'{keep.sum()}/{len(peak_idx)} peaks kept')
        peak_idx = peak_idx[keep]

    if len(peak_idx) == 0:
        return [], [], owner1, claimed2

    kernel        = np.ones(W12_SMOOTH_MS, dtype=np.float64) / W12_SMOOTH_MS
    signal_smooth = np.convolve(signal, kernel, 'same')

    cat1_raw: List[dict] = []
    cat2_raw: List[dict] = []
    for pid, tp in enumerate(peak_idx):
        h_peak           = float(signal[tp])
        w12_raw, bl_w12  = _compute_w12(signal_smooth, int(tp), bundle.baseline)
        w12              = min(w12_raw, W12_MAX_MS)
        entry            = dict(id=pid, tp=int(tp), h_peak=h_peak, w12=w12,
                                baseline_w12=bl_w12)
        if 2 * w12 >= CAT1_MIN_WIDTH_MS:
            cat1_raw.append(entry)
        else:
            cat2_raw.append(entry)

    cat1_raw.sort(key=lambda e: e['h_peak'], reverse=True)
    cat2_raw.sort(key=lambda e: e['h_peak'], reverse=True)

    print(f'  Peak finder: {len(cat1_raw)} Cat-1  |  {len(cat2_raw)} Cat-2')

    excluded1: set = set()
    excluded2: set = set()

    for ev in cat1_raw:
        if ev['id'] in excluded1:
            continue
        tp, w12 = ev['tp'], ev['w12']
        cl_s = max(0,     tp - w12)
        cl_e = min(N - 1, tp + w12)
        owner1[cl_s : cl_e + 1] = ev['id']
        ev['t_start'] = cl_s
        ev['t_end']   = cl_e
        for other in cat1_raw:
            if other['id'] != ev['id'] and 't_start' not in other:
                other_cl_s = max(0,     other['tp'] - other['w12'])
                other_cl_e = min(N - 1, other['tp'] + other['w12'])
                if cl_s <= other_cl_e and other_cl_s <= cl_e:
                    excluded1.add(other['id'])

    cat1 = [e for e in cat1_raw if e['id'] not in excluded1 and 't_start' in e]
    cat2 = list(cat2_raw)

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
        mask = (cat2_tp_arr >= cl_s) & (cat2_tp_arr <= cl_e) & (cat2_id_arr != ev['id'])
        for i in cat2_id_arr[mask]:
            other_c2 = cat2_by_id[int(i)]
            if 't_start' not in other_c2:
                excluded2.add(int(i))

    cat2 = [e for e in cat2 if e['id'] not in excluded2 and 't_start' in e]

    cat1.sort(key=lambda e: e['h_peak'], reverse=True)
    cat2.sort(key=lambda e: e['h_peak'], reverse=True)

    print(f'  After claiming: {len(cat1)} Cat-1  |  {len(cat2)} Cat-2')
    return cat1, cat2, owner1, claimed2


# ── Step 2: D-guided expansion with factor predictions ────────────────────────

def _step2(bundle: TraceBundle,
           seed_events: List[dict],
           claimed_self: np.ndarray,
           d_model,
           category: int,
           allow_swallow: bool = False) -> List[dict]:
    N       = len(bundle.i1ms)
    results = []

    if not seed_events:
        return results

    # Batch initial D predictions — factor=1.0 (for stopping logic)
    tps_all         = [ev['tp'] for ev in seed_events]
    bl_scalars_init = [_edge_baseline(bundle, ev['t_start'], ev['t_end'])
                       for ev in seed_events]
    initial_d_hats  = _predict_D_batch(bundle, tps_all, d_model,
                                       baseline_scalars=bl_scalars_init)

    # v4.0: batch initial predictions for each factor
    factor_initial_d = {}
    for f in BL_FACTORS:
        bl_f = [bl * f for bl in bl_scalars_init]
        factor_initial_d[f] = _predict_D_batch(bundle, tps_all, d_model,
                                                baseline_scalars=bl_f)

    if allow_swallow:
        finalized_ids: set = set()
        swallowed_ids: set = set()
        ev_by_id            = {ev['id']: ev for ev in seed_events}

    for i, (ev, D_hat_init) in enumerate(zip(seed_events, initial_d_hats)):
        tp      = ev['tp']
        t_start = ev['t_start']
        t_end   = ev['t_end']
        bl_init = bl_scalars_init[i]

        # ── v4.0: per-factor history, seeded with initial step ────────────
        factor_d_hist = {f: [float(factor_initial_d[f][i])] for f in BL_FACTORS}

        # ── Swallowed event ───────────────────────────────────────────────
        if allow_swallow and ev['id'] in swallowed_ids:
            dur   = t_end - t_start + 1
            W_seg = dur // 2
            pf    = max(0.0, (W_D - dur) / W_D) if dur < W_D else 0.0
            score = float((bundle.i1ms[tp] - bundle.baseline[tp])
                          / np.sqrt(max(bundle.baseline[tp], 1e-9)))
            results.append(dict(
                t_peak        = tp,
                t_left        = t_start,
                t_right       = t_end,
                W_seg         = W_seg,
                D_hat         = float(D_hat_init),
                log10_D       = float(np.log10(max(D_hat_init, 1e-10))),
                category      = category,
                fast_group    = False,
                score         = score,
                pad_frac      = pf,
                n_win         = 1,
                k_win         = 0,
                h_peak        = float(bundle.i1ms[tp]),
                w12_seed      = ev.get('w12'),
                baseline_w12  = ev.get('baseline_w12'),
                t_left_hist   = [t_start],
                t_right_hist  = [t_end],
                d_hist        = [float(D_hat_init)],
                baseline_hist = [bl_init],
                swallowed_by  = ev.get('swallowed_by'),
                d_hat_by_factor  = {f: factor_d_hist[f][-1] for f in BL_FACTORS},
                d_hist_by_factor = factor_d_hist,
            ))
            continue

        # ── Normal expansion ──────────────────────────────────────────────
        D_hat = D_hat_init
        w     = max(t_end - t_start, 1)

        fast_group = False if category == 1 else (D_hat > D_FAST_CUTOFF)

        t_left_hist   = [t_start]
        t_right_hist  = [t_end]
        d_hist        = [float(D_hat)]
        baseline_hist = [bl_init]

        if not fast_group:
            tau_max = _lookup_tau_max(D_hat)

            while w < W_MAX_MS:
                w_new  = 2 * w
                ts_new = max(0,     tp - w_new // 2)
                te_new = min(N - 1, tp + w_new // 2)

                if w_new >= W_MAX_MS:
                    break

                if allow_swallow:
                    left_owners  = set(claimed_self[ts_new : t_start].tolist())
                    right_owners = set(claimed_self[t_end + 1 : te_new + 1].tolist())
                    owners_in_range = (left_owners | right_owners) - {-1, ev['id']}

                    if owners_in_range:
                        if any(o in finalized_ids for o in owners_in_range):
                            break
                        for swallowee_id in owners_in_range:
                            swallowee = ev_by_id[swallowee_id]
                            sw_s = swallowee['t_start']
                            sw_e = swallowee['t_end']
                            claimed_self[sw_s : sw_e + 1] = -1
                            swallowed_ids.add(swallowee_id)
                            swallowee['swallowed_by'] = tp
                else:
                    if (claimed_self[ts_new : t_start].any()
                            or claimed_self[t_end + 1 : te_new + 1].any()):
                        break

                if category != 1 and w_new > tau_max:
                    break

                if allow_swallow:
                    claimed_self[ts_new   : t_start]      = ev['id']
                    claimed_self[t_end + 1 : te_new + 1]  = ev['id']
                else:
                    claimed_self[ts_new   : t_start]      = True
                    claimed_self[t_end + 1 : te_new + 1]  = True
                t_start, t_end = ts_new, te_new
                w = w_new
                t_left_hist.append(t_start)
                t_right_hist.append(t_end)

                new_bl = _edge_baseline(bundle, t_start, t_end)
                D_hat  = _predict_D(bundle, tp, d_model, baseline_scalar=new_bl)
                d_hist.append(float(D_hat))
                baseline_hist.append(new_bl)

                # v4.0: factor predictions at this expansion step
                for f in BL_FACTORS:
                    d_f = _predict_D(bundle, tp, d_model,
                                     baseline_scalar=new_bl * f)
                    factor_d_hist[f].append(float(d_f))

                if category != 1:
                    if D_hat > D_FAST_CUTOFF:
                        fast_group = True
                        break
                    tau_max = _lookup_tau_max(D_hat)

        if allow_swallow:
            finalized_ids.add(ev['id'])

        tp = int(t_start + np.argmax(bundle.i1ms[t_start : t_end + 1]))

        dur   = t_end - t_start + 1
        W_seg = dur // 2
        pf    = max(0.0, (W_D - dur) / W_D) if dur < W_D else 0.0
        nw    = 1 if dur <= W_D else max(1, (dur - W_D) // (W_D // 8) + 1)
        score = float((bundle.i1ms[tp] - bundle.baseline[tp])
                      / np.sqrt(max(bundle.baseline[tp], 1e-9)))

        results.append(dict(
            t_peak        = tp,
            t_left        = t_start,
            t_right       = t_end,
            W_seg         = W_seg,
            D_hat         = float(D_hat),
            log10_D       = float(np.log10(max(D_hat, 1e-10))),
            category      = category,
            fast_group    = fast_group,
            score         = score,
            pad_frac      = pf,
            n_win         = nw,
            k_win         = 0,
            h_peak        = float(bundle.i1ms[tp]),
            w12_seed      = ev.get('w12'),
            baseline_w12  = ev.get('baseline_w12'),
            t_left_hist   = t_left_hist,
            t_right_hist  = t_right_hist,
            d_hist        = d_hist,
            baseline_hist = baseline_hist,
            d_hat_by_factor  = {f: factor_d_hist[f][-1] for f in BL_FACTORS},
            d_hist_by_factor = factor_d_hist,
        ))

    return results


# ── Post-expansion absorption merge (unchanged) ───────────────────────────────

def _post_absorption_merge(events: List[dict]) -> List[dict]:
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


# ── Public API ────────────────────────────────────────────────────────────────

def run_stage3_v2(bundle: TraceBundle, d_model, radius_cap: int = 128,
                  amplitude_min_khz: Optional[float] = None,
                  fold_min: Optional[float] = None):
    N = len(bundle.i1ms)
    if N == 0:
        return [], np.array([], dtype=np.float64)

    z_score = ((bundle.i1ms - bundle.baseline)
               / np.sqrt(np.maximum(bundle.baseline, 1e-9)))

    cat1, cat2, owner1, claimed2 = _step1(bundle,
                                          amplitude_min_khz=amplitude_min_khz,
                                          fold_min=fold_min)
    if not cat1 and not cat2:
        return [], z_score

    all_cat1 = _step2(bundle, cat1, owner1,   d_model, category=1, allow_swallow=True)
    all_cat2 = _step2(bundle, cat2, claimed2, d_model, category=2)

    finalized = [e for e in all_cat1 + all_cat2 if 'swallowed_by' not in e]
    swallowed = [e for e in all_cat1             if 'swallowed_by' in  e]

    n_before = len(finalized)
    finalized = _post_absorption_merge(finalized)
    n_after   = len(finalized)
    if n_before != n_after:
        print(f'  Absorption merge: {n_before} → {n_after} events '
              f'({n_before - n_after} absorbed)')
    if swallowed:
        print(f'  Swallowed Cat-1 seeds: {len(swallowed)}')

    events = finalized + swallowed
    events.sort(key=lambda e: e['t_peak'])
    return events, z_score
