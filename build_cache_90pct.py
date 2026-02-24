"""
Build 90% train / 10% test cache by:
  - Reusing existing 50k train features
  - Computing features for 40k additional rows from the old test pool
  - Using remaining 10k as the new test set
"""
import numpy as np
import h5py
import time
import os
from scipy.stats import skew, kurtosis

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Reconstruct original index split ─────────────────────────────────────────
log("Reconstructing original index split...")
f    = h5py.File('unrollIntensity.mat', 'r')
uInt = f['uInt']
n_all     = uInt[1, :]
valid_idx = np.where(n_all == 0)[0]                        # 100k rows

rng       = np.random.default_rng(42)
old_train = rng.choice(valid_idx, size=len(valid_idx) // 2, replace=False)
old_train_set = set(old_train.tolist())
old_train.sort()

remaining = np.array([i for i in valid_idx if i not in old_train_set])  # 50k rows

# New split: 40k more for train, 10k for test
rng2       = np.random.default_rng(7)
extra_train_idx = rng2.choice(remaining, size=40000, replace=False)
extra_train_set = set(extra_train_idx.tolist())
extra_train_idx.sort()

test_idx = np.array([i for i in remaining if i not in extra_train_set])  # ~10k
test_idx.sort()

log(f"Old train: {len(old_train)}  Extra train: {len(extra_train_idx)}  Test: {len(test_idx)}")
log(f"Total train: {len(old_train) + len(extra_train_idx)}")

# ── Feature extraction helper ─────────────────────────────────────────────────
def extract_features(i, label=""):
    N = i.shape[1]
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
    lags  = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
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
    U1, S1 = [], []
    for psi in psi_bank:
        u1 = np.abs(np.fft.irfft(X_fft * psi, n=n_sc, axis=1)[:, :N])
        S1.append(u1.mean(axis=1)); U1.append(u1)
    S1 = np.column_stack(S1)
    S2 = []
    for k1 in range(n_filt):
        U1_fft = np.fft.rfft(U1[k1], n=n_sc, axis=1)
        for k2 in range(k1 + 1, n_filt):
            u2 = np.abs(np.fft.irfft(U1_fft * psi_bank[k2], n=n_sc, axis=1)[:, :N])
            S2.append(u2.mean(axis=1))
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

    log(f"{label} Done — {int(time.time()-t)}s elapsed")
    return np.column_stack([
        means, variances, q_mandel, skewness, kurt,
        kappa1, kappa2, kappa2_norm, g0, half_decay,
        acf, S1, S2, psd_bins
    ])

# ── Compute features for extra train rows ────────────────────────────────────
log("Reading extra train i from HDF5...")
i_extra = uInt[2:, extra_train_idx].T.astype(np.float64)
d_extra = uInt[0, extra_train_idx]
log(f"  Shape: {i_extra.shape}")

log("Extracting features for extra train rows...")
X_extra = extract_features(i_extra, label="[ExtraTrain]")
log(f"  Features shape: {X_extra.shape}")

# ── Compute features for new test rows ───────────────────────────────────────
log("Reading test i from HDF5...")
i_test = uInt[2:, test_idx].T.astype(np.float64)
d_test = uInt[0, test_idx]
log(f"  Shape: {i_test.shape}")

log("Extracting features for test rows...")
X_test = extract_features(i_test, label="[Test]")
log(f"  Features shape: {X_test.shape}")

# ── Load existing 50k train features and combine ──────────────────────────────
log("Loading existing 50k train features...")
X_old   = np.load('cache_X_train_psd.npy')
d_old   = np.load('cache_d_train.npy')

X_train_90 = np.vstack([X_old, X_extra])
d_train_90  = np.concatenate([d_old, d_extra])
log(f"Combined train shape: {X_train_90.shape}")

# ── Save new caches ───────────────────────────────────────────────────────────
np.save('cache_X_train_90pct.npy', X_train_90)
np.save('cache_d_train_90pct.npy', d_train_90)
np.save('cache_X_test_90pct.npy',  X_test)
np.save('cache_d_test_90pct.npy',  d_test)

log("Saved:")
log("  cache_X_train_90pct.npy  cache_d_train_90pct.npy")
log("  cache_X_test_90pct.npy   cache_d_test_90pct.npy")
log("Done.")
