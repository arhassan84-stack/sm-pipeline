"""
Standalone feature computation for a single intensity time trace (or batch).

Usage
-----
  # Single trace (1D array, length N):
  from compute_features import compute_features
  feats = compute_features(trace)          # returns (212,) array

  # Batch of traces (2D array, shape (n, N)):
  feats = compute_features(traces)         # returns (n, 212) array

  # Command-line demo:
  python compute_features.py

Feature vector (212 total)
--------------------------
Index  Count  Description
-----  -----  -----------
 0- 4      5  Intensity statistics: mean, variance, Mandel-Q, skewness, kurtosis
 5- 9      5  Factorial cumulants & decay: κ₁, κ₂, κ₂_norm, G₀, half-decay lag
10-38     29  ACF at 29 log-spaced lags (lag 1 … 2048)
39-54     16  Scattering S1: J=8, Q=2 Morlet filterbank mean amplitudes
55-174   120  Scattering S2: second-order scattering (k₂ > k₁ pairs)
175-206   32  PSD in 32 log-spaced frequency bins
207-211    5  Transit time: fraction of trace ≥ {80,60,40,20,10}% of peak

Dependencies: numpy, scipy
"""
import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurtosis


# ── Constants ────────────────────────────────────────────────────────────────
N               = 4096          # expected trace length
J, Q            = 8, 2          # scattering transform octaves / wavelets per octave
N_ACF_LAGS      = 32            # log-spaced points → ~29 unique after rounding
N_PSD_BINS      = 32            # log-spaced PSD bins
TRANSIT_THRESH  = [0.80, 0.60, 0.40, 0.20, 0.10]   # fractions of max


def compute_features(trace):
    """
    Compute all 212 features for one or more intensity time traces.

    Parameters
    ----------
    trace : array-like, shape (N,) or (n_traces, N)
        Raw intensity time trace(s).  Dtype will be cast to float64.

    Returns
    -------
    feats : np.ndarray, shape (212,) or (n_traces, 212)
        Feature vector(s).  Single trace returns 1-D array.
    """
    single = (np.ndim(trace) == 1)
    i = np.atleast_2d(np.asarray(trace, dtype=np.float64))   # (n, N)
    n_traces, trace_len = i.shape
    assert trace_len == N, f"Expected trace length {N}, got {trace_len}"

    feats = _extract_features(i)

    return feats[0] if single else feats


# ── Internal computation ──────────────────────────────────────────────────────

def _extract_features(i):
    """i: (n_traces, N) float64"""
    n_traces, L = i.shape

    # ── 1. Intensity statistics (cols 0-4) ───────────────────────────────────
    means     = i.mean(axis=1)                            # (n,)
    variances = i.var(axis=1)
    q_mandel  = (variances - means) / means
    skewness  = skew(i, axis=1)
    kurt      = scipy_kurtosis(i, axis=1)

    # ── 2. Factorial cumulants + G₀ + half-decay lag (cols 5-9) ─────────────
    kappa1      = means.copy()
    kappa2      = variances - means
    kappa2_norm = kappa2 / (kappa1 ** 2)
    g0          = variances / (means ** 2)

    # ACF via FFT (needed for half-decay; reused below)
    lags  = np.unique(np.round(np.logspace(0, np.log10(L // 2), N_ACF_LAGS)).astype(int))
    fluct = i - means[:, np.newaxis]
    n_fft = 2 ** int(np.ceil(np.log2(2 * L)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F) ** 2, n=n_fft, axis=1)

    acf = np.zeros((n_traces, len(lags)))
    for j, lag in enumerate(lags):
        acf[:, j] = corr[:, lag] / (L - lag) / (means ** 2 + 1e-12)

    half_g0    = g0 / 2
    half_decay = np.zeros(n_traces)
    for k in range(n_traces):
        crossed = np.where(acf[k] < half_g0[k])[0]
        half_decay[k] = lags[crossed[0]] if len(crossed) > 0 else lags[-1]

    # ── 3. Scattering transform S1 + S2 (cols 39-54, 55-174) ────────────────
    n_sc  = 2 ** int(np.ceil(np.log2(L)))
    freqs = np.fft.rfftfreq(n_sc)

    psi_bank = []
    for j in range(J):
        for q in range(Q):
            xi    = 0.5 / (2 ** (j + q / Q))
            sigma = xi / (Q * 2 * np.sqrt(2 * np.log(2)))
            psi_bank.append(np.exp(-0.5 * ((freqs - xi) / sigma) ** 2))

    X_fft  = np.fft.rfft(i, n=n_sc, axis=1)
    n_filt = len(psi_bank)   # J * Q = 16

    U1, S1 = [], []
    for psi in psi_bank:
        u1 = np.abs(np.fft.irfft(X_fft * psi, n=n_sc, axis=1)[:, :L])
        S1.append(u1.mean(axis=1))
        U1.append(u1)
    S1 = np.column_stack(S1)   # (n_traces, 16)

    S2 = []
    for k1 in range(n_filt):
        U1_fft = np.fft.rfft(U1[k1], n=n_sc, axis=1)
        for k2 in range(k1 + 1, n_filt):
            u2 = np.abs(np.fft.irfft(U1_fft * psi_bank[k2], n=n_sc, axis=1)[:, :L])
            S2.append(u2.mean(axis=1))
    S2 = np.column_stack(S2)   # (n_traces, 120)

    # ── 4. PSD in log-spaced bins (cols 175-206) ─────────────────────────────
    n_fft2    = 2 ** int(np.ceil(np.log2(L)))
    freqs2    = np.fft.rfftfreq(n_fft2)
    psd       = np.abs(np.fft.rfft(fluct, n=n_fft2, axis=1)) ** 2 / L
    bin_edges = np.logspace(np.log10(freqs2[1]), np.log10(freqs2[-1]), N_PSD_BINS + 1)
    psd_bins  = np.zeros((n_traces, N_PSD_BINS))
    for b in range(N_PSD_BINS):
        mask = (freqs2 >= bin_edges[b]) & (freqs2 < bin_edges[b + 1])
        if mask.sum() > 0:
            psd_bins[:, b] = psd[:, mask].mean(axis=1)

    # ── 5. Transit time features (cols 207-211) ──────────────────────────────
    i_max     = i.max(axis=1, keepdims=True)   # (n, 1)
    tt_feats  = np.zeros((n_traces, len(TRANSIT_THRESH)))
    for j, thr in enumerate(TRANSIT_THRESH):
        tt_feats[:, j] = (i >= thr * i_max).sum(axis=1) / L

    # ── Assemble ─────────────────────────────────────────────────────────────
    return np.column_stack([
        means, variances, q_mandel, skewness, kurt,   # 0-4
        kappa1, kappa2, kappa2_norm, g0, half_decay,  # 5-9
        acf,                                           # 10-38  (29)
        S1,                                            # 39-54  (16)
        S2,                                            # 55-174 (120)
        psd_bins,                                      # 175-206 (32)
        tt_feats,                                      # 207-211 (5)
    ])


# ── Feature name helper ───────────────────────────────────────────────────────

def feature_names():
    """Return a list of 212 feature name strings."""
    lags = np.unique(np.round(np.logspace(0, np.log10(N // 2), N_ACF_LAGS)).astype(int))
    n_filt = J * Q   # 16

    names = (
        ['mean', 'variance', 'mandel_Q', 'skewness', 'kurtosis',
         'kappa1', 'kappa2', 'kappa2_norm', 'G0', 'half_decay_lag']
        + [f'acf_lag{lag}' for lag in lags]
        + [f'S1_{j}_{q}' for j in range(J) for q in range(Q)]
        + [f'S2_{k1}_{k2}' for k1 in range(n_filt) for k2 in range(k1+1, n_filt)]
        + [f'psd_bin{b}' for b in range(N_PSD_BINS)]
        + [f'transit_{int(thr*100)}pct' for thr in TRANSIT_THRESH]
    )
    return names


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    print("compute_features.py — feature extraction demo")
    print(f"Feature count: {len(feature_names())}")
    print()

    # Try to load a real trace from the cache; fall back to synthetic
    cache = 'cache_i_test_90pct.npy'
    if os.path.exists(cache):
        print(f"Loading example trace from {cache}...")
        i_all  = np.load(cache)
        trace  = i_all[0].astype(np.float64)
        print(f"  Trace shape: {trace.shape}  min={trace.min():.2f}  max={trace.max():.2f}")
    else:
        print("Cache not found — using synthetic Poisson trace (mean=100)...")
        rng   = np.random.default_rng(0)
        trace = rng.poisson(100, size=N).astype(np.float64)

    import time
    t0    = time.time()
    feats = compute_features(trace)
    print(f"Single trace: {feats.shape}  ({time.time()-t0:.3f}s)")
    print()

    # Print feature values
    names = feature_names()
    print("Feature values:")
    groups = [
        ("Intensity stats   (0-4)   ", slice(0, 5)),
        ("Factorial cumulants (5-9) ", slice(5, 10)),
        ("ACF (first 5 lags)  (10+) ", slice(10, 15)),
        ("Scattering S1       (39+) ", slice(39, 45)),
        ("PSD (first 5 bins)  (175+)", slice(175, 180)),
        ("Transit times       (207+)", slice(207, 212)),
    ]
    for gname, sl in groups:
        vals = feats[sl]
        nms  = names[sl]
        print(f"  {gname}:")
        for nm, v in zip(nms, vals):
            print(f"    {nm:<25s} = {v:.6g}")
    print()

    # Batch demo
    if os.path.exists(cache):
        print("Batch demo (first 100 traces)...")
        batch  = i_all[:100].astype(np.float64)
        t0     = time.time()
        bfeats = compute_features(batch)
        print(f"  Batch shape: {bfeats.shape}  ({time.time()-t0:.2f}s)")
        print(f"  Transit time means: " +
              "  ".join(f"{int(thr*100)}%→{bfeats[:,207+j].mean():.4f}"
                        for j, thr in enumerate(TRANSIT_THRESH)))
