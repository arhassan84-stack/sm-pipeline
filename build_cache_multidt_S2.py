"""
Build train/test feature caches for the multi-dt S2/mCherry2 simulation dataset.

Loads sims_multidt_S2_noise0_i_{dt010,dt050,dt100}.npy and sims_multidt_S2_noise0_d.npy,
creates ONE consistent 90/10 train/test split (seed=42) and computes the full
302-feature pipeline independently for each dt.

Output files  (30 total):
  cache_i_{train,test}_multidt_S2_dt010.npy    (N × 40960) float32 raw traces
  cache_X_{train,test}_multidt_S2_dt010.npy    (N × 302)   float64 features
  cache_d_{train,test}_multidt_S2_dt010.npy    (N,)        float64 D values
  ... same for dt050  (N × 8192)
  ... same for dt100  (N × 4096)

Usage:
  python build_cache_multidt_S2.py --workers 8 --chunk-size 500
"""
import numpy as np
import argparse
import time
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Fixed pipeline constants (same as build_cache_dt.py) ─────────────────────
TRANSIT_THRESHOLDS = [0.80, 0.60, 0.40, 0.20, 0.10]
EXP_THRESHOLDS     = [0.80, 0.60, 0.40, 0.20, 0.10]
WIN_SIZES          = [32, 64, 128, 512, 1024]
BIN_TIMES          = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DECAY_FRACS        = [0.90, 0.75, 0.25, 0.10]
COL_G0             = 8
COL_ACF_START      = 10
LAG_COL = [
    ( 1, 10), ( 2, 11), ( 3, 12), ( 4, 13), ( 5,222),
    ( 6, 14), ( 7, 15), ( 8,223), ( 9, 16), (10,224),
    (11,225), (12, 17), (13,226), (14,227), (15, 18),
    (16,228), (17,229), (18,230), (19, 19), (20,231),
]
COLS_BY_LAG = {lag: col for lag, col in LAG_COL}


def make_lag_constants(N):
    LAGS_CURR      = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
    LAGS_ARR       = LAGS_CURR.astype(np.float64)
    NEW_SHORT_LAGS = np.setdiff1d(np.arange(1, 21), LAGS_CURR)
    return LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS


def extract_base_features(i, N, LAGS_CURR, LAGS_ARR):
    means     = i.mean(axis=1)
    variances = i.var(axis=1)
    q_mandel  = (variances - means) / means
    skewness  = skew(i, axis=1)
    kurt      = kurtosis(i, axis=1)
    kappa1    = means
    kappa2    = variances - means
    kappa2_norm = kappa2 / kappa1**2
    fluct = i - means[:, np.newaxis]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F)**2, n=n_fft, axis=1)
    acf   = np.zeros((len(i), len(LAGS_CURR)))
    for j, lag in enumerate(LAGS_CURR):
        acf[:, j] = corr[:, lag] / (N - lag) / means**2
    g0 = variances / means**2
    half_g0 = g0 / 2
    half_decay = np.zeros(len(i))
    for k in range(len(i)):
        crossed = np.where(acf[k] < half_g0[k])[0]
        half_decay[k] = LAGS_CURR[crossed[0]] if len(crossed) > 0 else LAGS_CURR[-1]
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
    S1 = []
    for psi in psi_bank:
        u1 = np.abs(np.fft.irfft(X_fft * psi, n=n_sc, axis=1)[:, :N])
        S1.append(u1.mean(axis=1))
    S1 = np.column_stack(S1)
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
    n_fft2 = 2 ** int(np.ceil(np.log2(N)))
    freqs2 = np.fft.rfftfreq(n_fft2)
    psd    = np.abs(np.fft.rfft(fluct, n=n_fft2, axis=1)) ** 2 / N
    bin_edges = np.logspace(np.log10(freqs2[1]), np.log10(freqs2[-1]), 33)
    psd_bins  = np.zeros((len(i), 32))
    for b in range(32):
        mask = (freqs2 >= bin_edges[b]) & (freqs2 < bin_edges[b+1])
        if mask.sum() > 0:
            psd_bins[:, b] = psd[:, mask].mean(axis=1)
    return np.column_stack([
        means, variances, q_mandel, skewness, kurt,
        kappa1, kappa2, kappa2_norm, g0, half_decay,
        acf, S1, S2, psd_bins
    ])


def compute_transit_features(i_raw, N):
    i_max = i_raw.max(axis=1, keepdims=True)
    result = np.zeros((len(i_raw), 5), dtype=np.float64)
    for j, thr in enumerate(TRANSIT_THRESHOLDS):
        result[:, j] = (i_raw >= thr * i_max).sum(axis=1) / N
    return result


def compute_expanded_features(i_raw, X_old, N, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS):
    n     = len(i_raw)
    i_f64 = i_raw.astype(np.float64)
    i_f32 = i_raw.astype(np.float32)
    means = X_old[:, 0].astype(np.float64)
    G0    = X_old[:, COL_G0].astype(np.float64)
    acf   = X_old[:, COL_ACF_START:COL_ACF_START + len(LAGS_CURR)].astype(np.float64)
    fluct = i_f64 - means[:, None]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F) ** 2, n=n_fft, axis=1)
    acf_short = np.zeros((n, len(NEW_SHORT_LAGS)))
    for j, lag in enumerate(NEW_SHORT_LAGS):
        acf_short[:, j] = corr[:, lag] / (N - lag) / (means ** 2 + 1e-12)
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
    seg_feats = np.zeros((n, 3 * len(WIN_SIZES)))
    for wi, w in enumerate(WIN_SIZES):
        n_segs = N // w
        segs   = i_f64[:, :n_segs * w].reshape(n, n_segs, w)
        sv = segs.var(axis=2); sm = segs.mean(axis=2)
        mean_sv = sv.mean(axis=1); std_sm = sm.std(axis=1)
        cv_sv   = np.where(mean_sv > 0, sv.std(axis=1) / np.maximum(mean_sv, 1e-10), 0.0)
        seg_feats[:, 3 * wi]     = mean_sv
        seg_feats[:, 3 * wi + 1] = std_sm
        seg_feats[:, 3 * wi + 2] = cv_sv
    dev         = fluct
    kappa3_norm = (dev ** 3).mean(axis=1) / np.maximum(means ** 3, 1e-10)
    kappa4_norm = (dev ** 4).mean(axis=1) / np.maximum(means ** 4, 1e-10)
    multitau = np.zeros((n, len(BIN_TIMES)))
    for bi, T in enumerate(BIN_TIMES):
        n_bins = N // T
        binned = i_f64[:, :n_bins * T].reshape(n, n_bins, T).sum(axis=2)
        mu_T = binned.mean(axis=1); var_T = binned.var(axis=1)
        multitau[:, bi] = var_T / np.maximum(mu_T ** 2, 1e-10)
    half   = N // 2
    mean1  = i_f64[:, :half].mean(axis=1); mean2 = i_f64[:, half:2*half].mean(axis=1)
    var1   = i_f64[:, :half].var(axis=1);  var2  = i_f64[:, half:2*half].var(axis=1)
    mean_ratio = mean1 / np.maximum(mean2, 1e-10)
    var_ratio  = var1  / np.maximum(var2,  1e-10)
    seg_w = N // 16
    segs16      = i_f64[:, :16 * seg_w].reshape(n, 16, seg_w)
    seg_means16 = segs16.mean(axis=2)
    idx_c       = np.arange(16, dtype=np.float64) - 7.5
    slope       = (seg_means16 * idx_c).sum(axis=1) / (idx_c ** 2).sum()
    slope_norm  = slope / np.maximum(means, 1e-10)
    h1 = i_f64[:, :half] - mean1[:, None]; h2 = i_f64[:, half:2*half] - mean2[:, None]
    numer = (h1 * h2).mean(axis=1); denom = np.sqrt(var1 * var2)
    corr_halves = np.where(denom > 1e-10, numer / denom, 0.0)
    nonstationarity = np.column_stack([mean_ratio, var_ratio, slope_norm, corr_halves])
    return np.column_stack([
        A_exp, tau_D_exp, rmse_exp, tau_D_s, beta, rmse_s,
        decay_lags, acf_short, cross_feats, seg_feats,
        kappa3_norm[:, None], kappa4_norm[:, None], multitau, nonstationarity,
    ])


def compute_acf_ratio_features(X_283):
    n = X_283.shape[0]
    out = np.zeros((n, 19), dtype=np.float64)
    for i in range(19):
        lag_lo, lag_hi = i + 1, i + 2
        col_lo = COLS_BY_LAG[lag_lo]; col_hi = COLS_BY_LAG[lag_hi]
        g_lo = np.maximum(X_283[:, col_lo], 1e-10)
        g_hi = np.maximum(X_283[:, col_hi], 1e-10)
        out[:, i] = np.log(g_hi / g_lo)
    return out


def _feature_chunk(args):
    i_chunk, N, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS = args
    i_f64     = i_chunk.astype(np.float64)
    X_base    = extract_base_features(i_f64, N, LAGS_CURR, LAGS_ARR)
    X_transit = compute_transit_features(i_chunk, N)
    X_212     = np.hstack([X_base, X_transit])
    X_exp     = compute_expanded_features(i_chunk, X_212, N, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS)
    X_283     = np.hstack([X_212, X_exp])
    X_302     = np.hstack([X_283, compute_acf_ratio_features(X_283)])
    return X_302


def build_features_for_split(i_split, N_BINS, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS,
                              n_workers, chunk_size, split_name):
    n_chunks  = max(n_workers, int(np.ceil(len(i_split) / chunk_size)))
    chunks    = np.array_split(i_split, n_chunks)
    args_list = [(c, N_BINS, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS) for c in chunks]
    log(f"  Computing {split_name} features ({len(i_split)} traces, "
        f"{n_workers} workers, {n_chunks} chunks)...")
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_feature_chunk, args_list)
    X = np.vstack(results)
    log(f"  {split_name} features: {X.shape}  ({time.time()-t0:.1f}s)")
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build multi-dt S2/mCherry2 feature caches')
    parser.add_argument('--workers',    type=int, default=8,
                        help='Parallel workers (default: 8)')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Traces per sub-task (default: 500)')
    parser.add_argument('--test-frac',  type=float, default=0.10,
                        help='Test fraction (default: 0.10)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for split (default: 42)')
    parser.add_argument('--tag',        type=str, default='multidt_S2',
                        help='Dataset tag prefix (default: multidt_S2)')
    args = parser.parse_args()

    t_total = time.time()
    DT_CONFIGS = [
        ('dt010', 'sims_multidt_S2_noise0_i_dt010.npy', 40960),
        ('dt050', 'sims_multidt_S2_noise0_i_dt050.npy',  8192),
        ('dt100', 'sims_multidt_S2_noise0_i_dt100.npy',  4096),
    ]

    # ── One shared train/test split ───────────────────────────────────────────
    log("Creating shared train/test split from D values...")
    d_all   = np.load('sims_multidt_S2_noise0_d.npy')
    N_total = len(d_all)
    rng     = np.random.default_rng(args.seed)
    N_test  = int(np.round(N_total * args.test_frac))
    test_idx  = rng.choice(N_total, size=N_test, replace=False)
    train_idx = np.setdiff1d(np.arange(N_total), test_idx)
    d_train   = d_all[train_idx]
    d_test    = d_all[test_idx]
    log(f"  Total: {N_total:,}  Train: {len(d_train):,}  Test: {len(d_test):,}")

    # Save D once (same for all dt)
    np.save(f'cache_d_train_{args.tag}.npy', d_train)
    np.save(f'cache_d_test_{args.tag}.npy',  d_test)
    log(f"  Saved cache_d_{{train,test}}_{args.tag}.npy")

    # ── Process each dt value ─────────────────────────────────────────────────
    for dt_tag, i_file, expected_bins in DT_CONFIGS:
        log(f"\n{'='*60}")
        log(f"Processing {dt_tag}  ({i_file})  expected {expected_bins} bins")
        log(f"{'='*60}")

        i_all  = np.load(i_file, mmap_mode='r')
        N_BINS = i_all.shape[1]
        log(f"  Shape: {i_all.shape}  N_BINS={N_BINS}")

        LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS = make_lag_constants(N_BINS)
        log(f"  Lags: {LAGS_CURR[0]}..{LAGS_CURR[-1]} ({len(LAGS_CURR)} lags)  "
            f"new_short: {len(NEW_SHORT_LAGS)}")

        # Extract split arrays (copies from mmap into RAM)
        log("  Extracting train/test intensity arrays...")
        i_train = np.array(i_all[train_idx], dtype=np.float32)
        i_test  = np.array(i_all[test_idx],  dtype=np.float32)
        del i_all
        log(f"  i_train: {i_train.shape}  i_test: {i_test.shape}")

        n_workers = min(args.workers, len(i_train))
        X_train = build_features_for_split(
            i_train, N_BINS, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS,
            n_workers, args.chunk_size, 'train')
        X_test = build_features_for_split(
            i_test, N_BINS, LAGS_CURR, LAGS_ARR, NEW_SHORT_LAGS,
            min(n_workers, len(i_test)), args.chunk_size, 'test')

        out_tag = f'{args.tag}_{dt_tag}'
        log(f"  Saving cache_{{i,X,d}}_{{train,test}}_{out_tag}.npy ...")
        np.save(f'cache_i_train_{out_tag}.npy', i_train)
        np.save(f'cache_X_train_{out_tag}.npy', X_train)
        np.save(f'cache_d_train_{out_tag}.npy', d_train)
        np.save(f'cache_i_test_{out_tag}.npy',  i_test)
        np.save(f'cache_X_test_{out_tag}.npy',  X_test)
        np.save(f'cache_d_test_{out_tag}.npy',  d_test)
        log(f"  Done {dt_tag}")
        del i_train, i_test, X_train, X_test

    log(f"\nAll done. Total: {time.time()-t_total:.1f}s")
