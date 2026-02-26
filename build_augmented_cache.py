"""
Build augmented training cache from original 90k data + new simulation data.

Workflow:
  1. Load existing 90pct caches (already have all features computed)
  2. Load new simulation data from .npy files (output of generate_simulations.py)
  3. Compute full feature pipeline for new data:
       207 base → +5 transit → 212 → +71 expanded → 283
       → +19 ACF ratio (if existing cache has 302 features)
  4. Stack with existing 90k train set → augmented caches

Test set: UNCHANGED — same cache_*_test_90pct.npy files used for evaluation

Outputs:
  cache_i_train_aug.npy   (90k+N, 4096)   float32 raw traces
  cache_X_train_aug.npy   (90k+N, F)      float64 features (F=283 or 302)
  cache_d_train_aug.npy   (90k+N,)        float64 diffusion coefficients

Usage:
  python build_augmented_cache.py --i_file new_sims_noise0_i.npy --d_file new_sims_noise0_d.npy
"""
import numpy as np
import argparse
import time
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count

# ── Configuration ─────────────────────────────────────────────────────────────
N = 4096

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Constants (must match build_cache_90pct.py / add_transit_features.py /
#              add_expanded_features.py / add_acf_ratio_features.py exactly) ──
TRANSIT_THRESHOLDS = [0.80, 0.60, 0.40, 0.20, 0.10]

EXP_THRESHOLDS   = [0.80, 0.60, 0.40, 0.20, 0.10]
WIN_SIZES        = [32, 64, 128, 512, 1024]
BIN_TIMES        = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DECAY_FRACS      = [0.90, 0.75, 0.25, 0.10]
LAGS_CURR        = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
LAGS_ARR         = LAGS_CURR.astype(np.float64)
NEW_SHORT_LAGS   = np.setdiff1d(np.arange(1, 21), LAGS_CURR)
COL_G0           = 8
COL_HALF_DECAY   = 9
COL_ACF_START    = 10

# ACF ratio feature columns (for 283→302 step)
LAG_COL = [
    ( 1, 10), ( 2, 11), ( 3, 12), ( 4, 13), ( 5,222),
    ( 6, 14), ( 7, 15), ( 8,223), ( 9, 16), (10,224),
    (11,225), (12, 17), (13,226), (14,227), (15, 18),
    (16,228), (17,229), (18,230), (19, 19), (20,231),
]
COLS_BY_LAG = {lag: col for lag, col in LAG_COL}


# ── Feature extraction functions ──────────────────────────────────────────────

def extract_base_features(i, label=""):
    """Compute 207 base features (same as build_cache_90pct.py extract_features)."""
    t = time.time()
    log(f"{label} Intensity statistics ({len(i)} traces)...")
    means     = i.mean(axis=1)
    variances = i.var(axis=1)
    q_mandel  = (variances - means) / means
    skewness  = skew(i, axis=1)
    kurt      = kurtosis(i, axis=1)
    kappa1    = means
    kappa2    = variances - means
    kappa2_norm = kappa2 / kappa1**2

    log(f"{label} ACF via FFT...")
    lags  = LAGS_CURR
    fluct = i - means[:, np.newaxis]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F)**2, n=n_fft, axis=1)
    acf   = np.zeros((len(i), len(lags)))
    for j, lag in enumerate(lags):
        acf[:, j] = corr[:, lag] / (N - lag) / means**2
    g0 = variances / means**2
    half_g0 = g0 / 2
    half_decay = np.zeros(len(i))
    for k in range(len(i)):
        crossed = np.where(acf[k] < half_g0[k])[0]
        half_decay[k] = lags[crossed[0]] if len(crossed) > 0 else lags[-1]

    log(f"{label} Scattering transform (J=8, Q=2)...")
    J, Q  = 8, 2
    n_sc  = 2 ** int(np.ceil(np.log2(N)))
    freqs = np.fft.rfftfreq(n_sc)
    psi_bank = []
    for j in range(J):
        for q in range(Q):
            xi    = 0.5 / (2 ** (j + q / Q))
            sigma = xi / (Q * 2 * np.sqrt(2 * np.log(2)))
            psi_bank.append(np.exp(-0.5 * ((freqs - xi) / sigma) ** 2))
    X_fft  = np.fft.rfft(i, n=n_sc, axis=1)
    n_filt = len(psi_bank)
    # S1: keep only the mean per filter (not the full u1 arrays) to save memory
    S1 = []
    for psi in psi_bank:
        u1 = np.abs(np.fft.irfft(X_fft * psi, n=n_sc, axis=1)[:, :N])
        S1.append(u1.mean(axis=1))
    S1 = np.column_stack(S1)
    # S2: recompute u1 on-the-fly to avoid storing 16 large arrays simultaneously.
    # Peak memory per worker: X_fft + u1_k1 + U1_fft + u2 ≈ 4 × 2.5 GB = 10 GB
    # vs storing all U1: 16 × 2.5 GB = 40 GB per worker.
    S2 = []
    for k1 in range(n_filt):
        u1_k1  = np.abs(np.fft.irfft(X_fft * psi_bank[k1], n=n_sc, axis=1)[:, :N])
        U1_fft = np.fft.rfft(u1_k1, n=n_sc, axis=1)
        del u1_k1
        for k2 in range(k1 + 1, n_filt):
            u2 = np.abs(np.fft.irfft(U1_fft * psi_bank[k2], n=n_sc, axis=1)[:, :N])
            S2.append(u2.mean(axis=1))
        del U1_fft
    S2 = np.column_stack(S2)

    log(f"{label} PSD (32 bins)...")
    n_fft2 = 2 ** int(np.ceil(np.log2(N)))
    freqs2 = np.fft.rfftfreq(n_fft2)
    psd    = np.abs(np.fft.rfft(fluct, n=n_fft2, axis=1)) ** 2 / N
    bin_edges = np.logspace(np.log10(freqs2[1]), np.log10(freqs2[-1]), 33)
    psd_bins  = np.zeros((len(i), 32))
    for b in range(32):
        mask = (freqs2 >= bin_edges[b]) & (freqs2 < bin_edges[b+1])
        if mask.sum() > 0:
            psd_bins[:, b] = psd[:, mask].mean(axis=1)

    log(f"{label} Done base features — {int(time.time()-t)}s elapsed")
    return np.column_stack([
        means, variances, q_mandel, skewness, kurt,
        kappa1, kappa2, kappa2_norm, g0, half_decay,
        acf, S1, S2, psd_bins
    ])


def compute_transit_features(i_raw):
    """Compute 5 transit time features (same as add_transit_features.py)."""
    i_max = i_raw.max(axis=1, keepdims=True)
    result = np.zeros((len(i_raw), 5), dtype=np.float64)
    for j, thr in enumerate(TRANSIT_THRESHOLDS):
        result[:, j] = (i_raw >= thr * i_max).sum(axis=1) / N
    return result


def compute_expanded_features(i_raw, X_old, label=""):
    """Compute 71 expanded features (same as add_expanded_features.py)."""
    n = len(i_raw)
    i_f32 = i_raw.astype(np.float32)
    i_f64 = i_raw.astype(np.float64)

    means = X_old[:, 0].astype(np.float64)
    G0    = X_old[:, COL_G0].astype(np.float64)
    acf   = X_old[:, COL_ACF_START:COL_ACF_START + len(LAGS_CURR)].astype(np.float64)

    log(f"{label} FFT correlation for short lags...")
    fluct = i_f64 - means[:, None]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F) ** 2, n=n_fft, axis=1)
    acf_short = np.zeros((n, len(NEW_SHORT_LAGS)))
    for j, lag in enumerate(NEW_SHORT_LAGS):
        acf_short[:, j] = corr[:, lag] / (N - lag) / (means ** 2 + 1e-12)

    log(f"{label} Exponential ACF fit...")
    log_acf = np.log(np.maximum(acf, 1e-10))
    X_ols   = np.column_stack([np.ones(len(LAGS_ARR)), LAGS_ARR])
    XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
    coeffs  = log_acf @ X_ols @ XtX_inv
    A_exp     = np.exp(np.clip(coeffs[:, 0], -10, 10))
    inv_tau   = -coeffs[:, 1]
    tau_D_exp = np.where(inv_tau > 1e-10, 1.0 / np.maximum(inv_tau, 1e-10), LAGS_ARR[-1])
    tau_D_exp = np.clip(tau_D_exp, 0.1, N)
    pred_exp  = A_exp[:, None] * np.exp(-LAGS_ARR[None, :] / tau_D_exp[:, None])
    rmse_exp  = np.sqrt(((acf - pred_exp) ** 2).mean(axis=1))

    log(f"{label} Stretched exponential fit...")
    ratio_s  = acf / np.maximum(G0[:, None], 1e-10)
    ratio_s  = np.clip(ratio_s, 1e-10, 1.0 - 1e-10)
    y_ll     = np.log(-np.log(ratio_s))
    log_lags = np.log(np.maximum(LAGS_ARR, 1.0))
    X_ll     = np.column_stack([log_lags, np.ones(len(LAGS_ARR))])
    XtX_inv_ll = np.linalg.inv(X_ll.T @ X_ll)
    coeffs_ll  = y_ll @ X_ll @ XtX_inv_ll
    beta       = np.clip(coeffs_ll[:, 0], 0.05, 5.0)
    c_ll       = coeffs_ll[:, 1]
    tau_D_s    = np.exp(np.clip(-c_ll / np.maximum(beta, 0.05), 0, np.log(N)))
    pred_s     = G0[:, None] * np.exp(-(LAGS_ARR[None, :] / np.maximum(tau_D_s[:, None], 0.1)) ** beta[:, None])
    rmse_s     = np.sqrt(((acf - pred_s) ** 2).mean(axis=1))

    log(f"{label} ACF decay percentiles...")
    decay_lags = np.zeros((n, len(DECAY_FRACS)))
    for fi, frac in enumerate(DECAY_FRACS):
        thresh     = frac * G0
        below_mask = acf < thresh[:, None]
        has_below  = below_mask.any(axis=1)
        first_idx  = np.where(has_below, below_mask.argmax(axis=1), len(LAGS_CURR) - 1)
        i0 = np.maximum(first_idx - 1, 0)
        i1 = np.minimum(first_idx, len(LAGS_CURR) - 1)
        t0 = LAGS_ARR[i0]; t1 = LAGS_ARR[i1]
        a0 = acf[np.arange(n), i0]; a1 = acf[np.arange(n), i1]
        denom  = np.where(np.abs(a1 - a0) > 1e-12, a1 - a0, 1.0)
        t_lerp = t0 + (thresh - a0) / denom * (t1 - t0)
        t_lerp = np.clip(t_lerp, t0, t1)
        decay_lags[:, fi] = np.where(has_below, t_lerp, LAGS_ARR[-1])

    log(f"{label} Threshold crossing statistics...")
    i_max  = i_f32.max(axis=1)
    cross_feats = np.zeros((n, 4 * len(EXP_THRESHOLDS)))
    for ti, thr in enumerate(EXP_THRESHOLDS):
        thresh_vals = (thr * i_max).astype(np.float32)
        above       = i_f32 >= thresh_vals[:, None]
        trans       = np.diff(above.astype(np.int8), axis=1)
        n_cross     = (trans == 1).sum(axis=1)
        cross_feats[:, 4 * ti] = n_cross / N
        max_burst  = np.zeros(n)
        std_burst  = np.zeros(n)
        mean_inter = np.ones(n)
        for k in range(n):
            a      = above[k]
            chg    = np.diff(a.view(np.int8), prepend=np.int8(0), append=np.int8(0))
            starts = np.where(chg ==  1)[0]
            ends   = np.where(chg == -1)[0]
            if len(starts) > 0:
                durs = ends - starts
                max_burst[k]  = durs.max() / N
                std_burst[k]  = durs.std() / N
                mean_inter[k] = (np.diff(starts).mean() / N) if len(starts) > 1 else 1.0
        cross_feats[:, 4 * ti + 1] = max_burst
        cross_feats[:, 4 * ti + 2] = std_burst
        cross_feats[:, 4 * ti + 3] = mean_inter

    log(f"{label} Multi-scale segment statistics...")
    seg_feats = np.zeros((n, 3 * len(WIN_SIZES)))
    for wi, w in enumerate(WIN_SIZES):
        n_segs = N // w
        segs   = i_f64[:, :n_segs * w].reshape(n, n_segs, w)
        sv = segs.var(axis=2)
        sm = segs.mean(axis=2)
        mean_sv = sv.mean(axis=1)
        std_sm  = sm.std(axis=1)
        cv_sv   = np.where(mean_sv > 0, sv.std(axis=1) / np.maximum(mean_sv, 1e-10), 0.0)
        seg_feats[:, 3 * wi]     = mean_sv
        seg_feats[:, 3 * wi + 1] = std_sm
        seg_feats[:, 3 * wi + 2] = cv_sv

    log(f"{label} Higher-order cumulants + multi-tau variance...")
    dev         = fluct
    kappa3_norm = (dev ** 3).mean(axis=1) / np.maximum(means ** 3, 1e-10)
    kappa4_norm = (dev ** 4).mean(axis=1) / np.maximum(means ** 4, 1e-10)
    multitau = np.zeros((n, len(BIN_TIMES)))
    for bi, T in enumerate(BIN_TIMES):
        n_bins = N // T
        binned = i_f64[:, :n_bins * T].reshape(n, n_bins, T).sum(axis=2)
        mu_T   = binned.mean(axis=1)
        var_T  = binned.var(axis=1)
        multitau[:, bi] = var_T / np.maximum(mu_T ** 2, 1e-10)

    log(f"{label} Non-stationarity features...")
    half   = N // 2
    mean1  = i_f64[:, :half].mean(axis=1)
    mean2  = i_f64[:, half:].mean(axis=1)
    var1   = i_f64[:, :half].var(axis=1)
    var2   = i_f64[:, half:].var(axis=1)
    mean_ratio = mean1 / np.maximum(mean2, 1e-10)
    var_ratio  = var1  / np.maximum(var2,  1e-10)
    segs16      = i_f64[:, :16 * 256].reshape(n, 16, 256)
    seg_means16 = segs16.mean(axis=2)
    idx_c       = np.arange(16, dtype=np.float64) - 7.5
    slope       = (seg_means16 * idx_c).sum(axis=1) / (idx_c ** 2).sum()
    slope_norm  = slope / np.maximum(means, 1e-10)
    h1    = i_f64[:, :half] - mean1[:, None]
    h2    = i_f64[:, half:] - mean2[:, None]
    numer = (h1 * h2).mean(axis=1)
    denom = np.sqrt(var1 * var2)
    corr_halves = np.where(denom > 1e-10, numer / denom, 0.0)
    nonstationarity = np.column_stack([mean_ratio, var_ratio, slope_norm, corr_halves])

    return np.column_stack([
        A_exp, tau_D_exp, rmse_exp,
        tau_D_s, beta, rmse_s,
        decay_lags,
        acf_short,
        cross_feats,
        seg_feats,
        kappa3_norm[:, None], kappa4_norm[:, None],
        multitau,
        nonstationarity,
    ])


def compute_acf_ratio_features(X_283):
    """Compute 19 log-ACF ratio features (same as add_acf_ratio_features.py)."""
    n = X_283.shape[0]
    out = np.zeros((n, 19), dtype=np.float64)
    for i in range(19):
        lag_lo, lag_hi = i + 1, i + 2
        col_lo = COLS_BY_LAG[lag_lo]
        col_hi = COLS_BY_LAG[lag_hi]
        g_lo = np.maximum(X_283[:, col_lo], 1e-10)
        g_hi = np.maximum(X_283[:, col_hi], 1e-10)
        out[:, i] = np.log(g_hi / g_lo)
    return out


# ── Parallel worker (module-level for multiprocessing pickling) ───────────────
def _feature_chunk(args):
    """
    Compute the full feature pipeline for one chunk of traces.
    Called by each worker process — no logging to avoid garbled output.

    args: (i_chunk, n_feat_target)
      i_chunk       : (n, 4096) float64 intensity traces
      n_feat_target : 283 or 302 — must match existing cache feature count
    """
    i_chunk, n_feat_target = args

    # 207 base features
    X_base    = extract_base_features(i_chunk, label="")
    # +5 transit → 212
    X_transit = compute_transit_features(i_chunk)
    X_212     = np.hstack([X_base, X_transit])
    # +71 expanded → 283
    X_exp     = compute_expanded_features(i_chunk, X_212, label="")
    X_283     = np.hstack([X_212, X_exp])
    # +19 ACF ratios → 302 (if needed)
    if n_feat_target >= 302:
        X_283 = np.hstack([X_283, compute_acf_ratio_features(X_283)])

    return X_283


if __name__ == '__main__':
    # ── Main ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Build augmented feature cache')
    parser.add_argument('--i_file', required=True,
                        help='Path to new intensity traces .npy  (output of generate_simulations.py)')
    parser.add_argument('--d_file', required=True,
                        help='Path to new d-value labels .npy    (output of generate_simulations.py)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help='Parallel worker processes for feature extraction (default: all CPUs)')
    args = parser.parse_args()

    t0_total = time.time()

    # Step 1: Load existing 90pct caches
    log("=== Step 1: Loading existing 90pct training caches ===")
    i_train_90      = np.load('cache_i_train_90pct.npy').astype(np.float32)
    X_train_90      = np.load('cache_X_train_90pct.npy')
    d_train_90      = np.load('cache_d_train_90pct.npy')
    n_feat_existing = X_train_90.shape[1]
    log(f"  Existing train: {i_train_90.shape}  features: {X_train_90.shape}  labels: {d_train_90.shape}")
    log(f"  Feature count detected: {n_feat_existing} {'(with ACF ratios)' if n_feat_existing == 302 else '(without ACF ratios)'}")

    # Step 2: Load new simulation data from .npy files
    log(f"\n=== Step 2: Loading new simulation data ===")
    log(f"  traces: {args.i_file}")
    log(f"  labels: {args.d_file}")
    i_new = np.load(args.i_file).astype(np.float64)   # (N, 4096)
    d_new = np.load(args.d_file)                       # (N,)
    log(f"  New traces shape: {i_new.shape}  d range: [{d_new.min():.3f}, {d_new.max():.3f}]")

    # Step 3: Compute features for new data (parallel over chunks of traces)
    n_workers = min(args.workers, len(i_new))
    log(f"\n=== Step 3: Computing features for {len(i_new)} new traces ({n_workers} workers) ===")

    chunks    = np.array_split(i_new, n_workers)
    args_list = [(chunk, n_feat_existing) for chunk in chunks]

    t3 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_feature_chunk, args_list)
    X_new_full = np.vstack(results)
    log(f"  Feature extraction complete in {time.time()-t3:.1f}s")
    log(f"  Features shape: {X_new_full.shape}")

    assert X_new_full.shape[1] == n_feat_existing, \
        f"Feature count mismatch: computed {X_new_full.shape[1]}, expected {n_feat_existing}"

    # Step 4: Combine with existing 90k training set
    log(f"\n=== Step 4: Combining existing 90k + {len(i_new)} new traces ===")
    i_new_f32   = i_new.astype(np.float32)
    i_train_aug = np.vstack([i_train_90, i_new_f32])
    X_train_aug = np.vstack([X_train_90, X_new_full])
    d_train_aug = np.concatenate([d_train_90, d_new])
    log(f"  Augmented train: {i_train_aug.shape}  features: {X_train_aug.shape}  labels: {d_train_aug.shape}")
    log(f"  d range (aug): [{d_train_aug.min():.3f}, {d_train_aug.max():.3f}]")

    # Step 5: Save augmented caches
    log("\n=== Step 5: Saving augmented caches ===")
    np.save('cache_i_train_aug.npy', i_train_aug)
    np.save('cache_X_train_aug.npy', X_train_aug)
    np.save('cache_d_train_aug.npy', d_train_aug)
    log(f"  Saved cache_i_train_aug.npy  shape={i_train_aug.shape}  dtype={i_train_aug.dtype}")
    log(f"  Saved cache_X_train_aug.npy  shape={X_train_aug.shape}  dtype={X_train_aug.dtype}")
    log(f"  Saved cache_d_train_aug.npy  shape={d_train_aug.shape}  dtype={d_train_aug.dtype}")

    log(f"\n=== Done! Total time: {time.time()-t0_total:.1f}s ===")
    log(f"  Original train: {len(d_train_90)}  New traces: {len(d_new)}  Augmented: {len(d_train_aug)}")
    log(f"  Test set unchanged: cache_*_test_90pct.npy")
    log(f"  Use _aug caches for retrained models (see pt_*_aug_gpu.py scripts)")
