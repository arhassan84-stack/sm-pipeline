"""
Expand the 212-feature cache with 71 additional physics-informed features.

New features appended (cols 212-282):
  212-214 ( 3)  ACF exponential fit:      A_exp, tau_D_exp, rmse_exp
  215-217 ( 3)  ACF stretched-exp fit:    tau_D_stretch, beta, rmse_stretch
  218-221 ( 4)  ACF decay percentiles:    lag at 90/75/25/10 % of G0
  222-231 (10)  ACF at short linear lags: [5,8,10,11,13,14,16,17,18,20]
  232-251 (20)  Threshold crossing stats: n_bursts, max_burst, std_burst,
                                          mean_inter × {80,60,40,20,10}% thresholds
  252-266 (15)  Multi-scale segment stats: mean_var, std_mean, cv_var
                                           × window sizes [32,64,128,512,1024]
  267-268 ( 2)  Higher-order cumulants:   kappa3_norm, kappa4_norm
  269-278 (10)  Multi-tau variance:       kappa2_norm(T) at T=2,4,8,...,1024
  279-282 ( 4)  Non-stationarity:         mean_ratio, var_ratio, slope_norm,
                                           corr_halves

Updates in-place:
  cache_X_train_90pct.npy  (90k, 212) -> (90k, 283)
  cache_X_test_90pct.npy   (10k, 212) -> (10k, 283)
"""
import numpy as np
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

N = 4096
THRESHOLDS   = [0.80, 0.60, 0.40, 0.20, 0.10]
WIN_SIZES    = [32, 64, 128, 512, 1024]
BIN_TIMES    = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DECAY_FRACS  = [0.90, 0.75, 0.25, 0.10]

# Existing ACF lag positions (must match build_cache_90pct.py exactly)
LAGS_CURR = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
LAGS_ARR  = LAGS_CURR.astype(np.float64)
NEW_SHORT_LAGS = np.setdiff1d(np.arange(1, 21), LAGS_CURR)  # [5,8,10,11,13,14,16,17,18,20]

# Column indices in the 212-feature cache
COL_G0         = 8
COL_HALF_DECAY = 9
COL_ACF_START  = 10   # cols 10-38 = ACF at LAGS_CURR (29 values)


def compute_expanded_features(i_raw, X_old, label=""):
    """
    i_raw : (n, N) float32/float64  raw intensity traces
    X_old : (n, 212) float64        existing features
    Returns (n, 71) float64 array of new features only.
    """
    n = len(i_raw)
    i_f32 = i_raw.astype(np.float32)
    i_f64 = i_raw.astype(np.float64)

    # Pull existing quantities we need
    means   = X_old[:, 0].astype(np.float64)       # (n,)
    G0      = X_old[:, COL_G0].astype(np.float64)  # (n,)
    acf     = X_old[:, COL_ACF_START:COL_ACF_START + len(LAGS_CURR)].astype(np.float64)  # (n,29)

    # ── 1. FFT correlation (needed for new short lags) ──────────────────────
    log(f"{label} FFT correlation for short lags...")
    fluct = i_f64 - means[:, None]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F) ** 2, n=n_fft, axis=1)   # (n, n_fft)

    acf_short = np.zeros((n, len(NEW_SHORT_LAGS)))
    for j, lag in enumerate(NEW_SHORT_LAGS):
        acf_short[:, j] = corr[:, lag] / (N - lag) / (means ** 2 + 1e-12)

    # ── 2. Exponential ACF fit  (vectorised OLS in log-space) ───────────────
    log(f"{label} Exponential ACF fit...")
    log_acf = np.log(np.maximum(acf, 1e-10))           # (n, 29)
    X_ols   = np.column_stack([np.ones(len(LAGS_ARR)), LAGS_ARR])  # (29, 2)
    XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
    coeffs  = log_acf @ X_ols @ XtX_inv                # (n, 2): [log_A, -1/tau_D]
    A_exp     = np.exp(np.clip(coeffs[:, 0], -10, 10))
    inv_tau   = -coeffs[:, 1]
    tau_D_exp = np.where(inv_tau > 1e-10,
                         1.0 / np.maximum(inv_tau, 1e-10),
                         LAGS_ARR[-1])
    tau_D_exp = np.clip(tau_D_exp, 0.1, N)
    pred_exp  = A_exp[:, None] * np.exp(-LAGS_ARR[None, :] / tau_D_exp[:, None])
    rmse_exp  = np.sqrt(((acf - pred_exp) ** 2).mean(axis=1))

    # ── 3. Stretched exponential fit  (vectorised log-log OLS) ──────────────
    log(f"{label} Stretched exponential fit...")
    # G(tau) = G0 * exp(-(tau/tau_D)^beta)
    # log(-log(G/G0)) = beta*log(tau) + c   where tau_D = exp(-c/beta)
    ratio_s  = acf / np.maximum(G0[:, None], 1e-10)
    ratio_s  = np.clip(ratio_s, 1e-10, 1.0 - 1e-10)
    y_ll     = np.log(-np.log(ratio_s))               # (n, 29)
    log_lags = np.log(np.maximum(LAGS_ARR, 1.0))      # (29,)
    X_ll     = np.column_stack([log_lags, np.ones(len(LAGS_ARR))])  # (29, 2)
    XtX_inv_ll = np.linalg.inv(X_ll.T @ X_ll)
    coeffs_ll  = y_ll @ X_ll @ XtX_inv_ll             # (n, 2): [beta, c]
    beta       = np.clip(coeffs_ll[:, 0], 0.05, 5.0)
    c_ll       = coeffs_ll[:, 1]
    tau_D_s    = np.exp(np.clip(-c_ll / np.maximum(beta, 0.05), 0, np.log(N)))
    pred_s     = G0[:, None] * np.exp(-(LAGS_ARR[None, :] / np.maximum(tau_D_s[:, None], 0.1)) ** beta[:, None])
    rmse_s     = np.sqrt(((acf - pred_s) ** 2).mean(axis=1))

    # ── 4. ACF decay percentiles (vectorised) ───────────────────────────────
    log(f"{label} ACF decay percentiles...")
    decay_lags = np.zeros((n, len(DECAY_FRACS)))
    for fi, frac in enumerate(DECAY_FRACS):
        thresh     = frac * G0                             # (n,)
        below_mask = acf < thresh[:, None]                 # (n, 29)
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

    # ── 5. Threshold crossing statistics ────────────────────────────────────
    log(f"{label} Threshold crossing statistics...")
    i_max  = i_f32.max(axis=1)          # (n,)
    cross_feats = np.zeros((n, 4 * len(THRESHOLDS)))

    for ti, thr in enumerate(THRESHOLDS):
        thresh_vals = (thr * i_max).astype(np.float32)          # (n,)
        above       = i_f32 >= thresh_vals[:, None]              # (n, N) bool
        # n_crossings: vectorised rising-edge count
        trans       = np.diff(above.astype(np.int8), axis=1)    # (n, N-1)
        n_cross     = (trans == 1).sum(axis=1)                   # (n,)
        cross_feats[:, 4 * ti] = n_cross / N

        # Burst stats: loop per trace (irregular run lengths)
        max_burst  = np.zeros(n)
        std_burst  = np.zeros(n)
        mean_inter = np.ones(n)    # normalised default = 1.0 (= N/N)
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

    # ── 6. Multi-scale segment statistics ───────────────────────────────────
    log(f"{label} Multi-scale segment statistics...")
    seg_feats = np.zeros((n, 3 * len(WIN_SIZES)))
    for wi, w in enumerate(WIN_SIZES):
        n_segs = N // w
        segs   = i_f64[:, :n_segs * w].reshape(n, n_segs, w)  # (n, n_segs, w)
        sv = segs.var(axis=2)    # (n, n_segs) within-segment variance
        sm = segs.mean(axis=2)   # (n, n_segs) within-segment mean
        mean_sv = sv.mean(axis=1)
        std_sm  = sm.std(axis=1)
        cv_sv   = np.where(mean_sv > 0, sv.std(axis=1) / np.maximum(mean_sv, 1e-10), 0.0)
        seg_feats[:, 3 * wi]     = mean_sv
        seg_feats[:, 3 * wi + 1] = std_sm
        seg_feats[:, 3 * wi + 2] = cv_sv

    # ── 7. Higher-order cumulants ────────────────────────────────────────────
    log(f"{label} Higher-order cumulants + multi-tau variance...")
    dev         = fluct                                          # already computed (n, N)
    kappa3_norm = (dev ** 3).mean(axis=1) / np.maximum(means ** 3, 1e-10)
    kappa4_norm = (dev ** 4).mean(axis=1) / np.maximum(means ** 4, 1e-10)

    multitau = np.zeros((n, len(BIN_TIMES)))
    for bi, T in enumerate(BIN_TIMES):
        n_bins   = N // T
        binned   = i_f64[:, :n_bins * T].reshape(n, n_bins, T).sum(axis=2)
        mu_T     = binned.mean(axis=1)
        var_T    = binned.var(axis=1)
        multitau[:, bi] = var_T / np.maximum(mu_T ** 2, 1e-10)

    # ── 8. Non-stationarity features ────────────────────────────────────────
    log(f"{label} Non-stationarity features...")
    half   = N // 2
    mean1  = i_f64[:, :half].mean(axis=1)
    mean2  = i_f64[:, half:].mean(axis=1)
    var1   = i_f64[:, :half].var(axis=1)
    var2   = i_f64[:, half:].var(axis=1)
    mean_ratio = mean1 / np.maximum(mean2, 1e-10)
    var_ratio  = var1  / np.maximum(var2,  1e-10)

    # Linear trend of 16-segment means (slope normalised by overall mean)
    segs16      = i_f64[:, :16 * 256].reshape(n, 16, 256)
    seg_means16 = segs16.mean(axis=2)                           # (n, 16)
    idx_c       = np.arange(16, dtype=np.float64) - 7.5
    slope       = (seg_means16 * idx_c).sum(axis=1) / (idx_c ** 2).sum()
    slope_norm  = slope / np.maximum(means, 1e-10)

    # Pearson correlation between first and second half of trace
    h1    = i_f64[:, :half] - mean1[:, None]
    h2    = i_f64[:, half:] - mean2[:, None]
    numer = (h1 * h2).mean(axis=1)
    denom = np.sqrt(var1 * var2)
    corr_halves = np.where(denom > 1e-10, numer / denom, 0.0)

    nonstationarity = np.column_stack([mean_ratio, var_ratio, slope_norm, corr_halves])

    # ── Assemble all new features ────────────────────────────────────────────
    return np.column_stack([
        A_exp, tau_D_exp, rmse_exp,          # 3   (212-214)
        tau_D_s, beta, rmse_s,               # 3   (215-217)
        decay_lags,                          # 4   (218-221)
        acf_short,                           # 10  (222-231)
        cross_feats,                         # 20  (232-251)
        seg_feats,                           # 15  (252-266)
        kappa3_norm[:, None], kappa4_norm[:, None],  # 2  (267-268)
        multitau,                            # 10  (269-278)
        nonstationarity,                     # 4   (279-282)
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
for split, i_file, x_file in [
    ('train_90pct', 'cache_i_train_90pct.npy', 'cache_X_train_90pct.npy'),
    ('test_90pct',  'cache_i_test_90pct.npy',  'cache_X_test_90pct.npy'),
]:
    log(f"=== Processing {split} ===")

    log(f"  Loading raw traces from {i_file}...")
    i_raw = np.load(i_file)          # float32, (n, 4096)
    log(f"  Traces shape: {i_raw.shape}")

    log(f"  Loading existing features from {x_file}...")
    X_old = np.load(x_file).astype(np.float64)
    log(f"  Existing features shape: {X_old.shape}")

    if X_old.shape[1] == 283:
        log("  Already 283 features — stripping old expanded cols first")
        X_old = X_old[:, :212]
    assert X_old.shape[1] == 212, f"Expected 212-col base, got {X_old.shape[1]}"

    t0 = time.time()
    X_new = compute_expanded_features(i_raw, X_old, label=f"[{split}]")
    log(f"  New features computed in {time.time()-t0:.1f}s  shape={X_new.shape}")

    X_out = np.hstack([X_old, X_new])
    np.save(x_file, X_out)
    log(f"  Saved {x_file}  shape={X_out.shape}")

log(f"Done.  Feature count: 212 -> 283")
log(f"New groups (cols 212-282):")
log(f"  212-214: exponential fit (A, tau_D, RMSE)")
log(f"  215-217: stretched-exp fit (tau_D, beta, RMSE)")
log(f"  218-221: ACF decay at 90/75/25/10% of G0")
log(f"  222-231: ACF at short linear lags {NEW_SHORT_LAGS.tolist()}")
log(f"  232-251: crossing stats (n,max,std,inter) × 5 thresholds")
log(f"  252-266: segment stats (var,std_mean,cv) × 5 window sizes")
log(f"  267-268: kappa3_norm, kappa4_norm")
log(f"  269-278: multi-tau kappa2_norm at T={BIN_TIMES}")
log(f"  279-282: mean_ratio, var_ratio, slope_norm, corr_halves")
