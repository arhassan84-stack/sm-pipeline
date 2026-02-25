"""
Extend existing 207-feature caches with:
  - 29 extra ACF lags (total 58, using 64 log-spaced → unique)
  - 16 segment variances (trace split into 16 chunks of 256)
Outputs:
  cache_X_train_extfeat.npy  (50k, 252)
  cache_X_test_extfeat.npy   (5k,  252)
"""
import numpy as np
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

N = 4096

# Lags used in original 207-feature cache (32 logspace → ~29 unique)
lags_old = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
# Extended set (64 logspace → ~58 unique)
lags_all = np.unique(np.round(np.logspace(0, np.log10(N // 2), 64)).astype(int))
# Only the NEW lags not already in cache
extra_lags = np.setdiff1d(lags_all, lags_old)
log(f"Original lags: {len(lags_old)}  All lags: {len(lags_all)}  Extra lags: {len(extra_lags)}")
log(f"Final feature count: 207 + {len(extra_lags)} ACF + 16 seg_var = {207 + len(extra_lags) + 16}")

def compute_extra_features(i_raw):
    """Compute extra ACF lags + segment variances from raw traces."""
    means  = i_raw.mean(axis=1)
    fluct  = i_raw - means[:, np.newaxis]
    n_fft  = 2 ** int(np.ceil(np.log2(2 * N)))
    F      = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr   = np.fft.irfft(np.abs(F)**2, n=n_fft, axis=1)

    # Extra ACF lags
    acf_extra = np.zeros((len(i_raw), len(extra_lags)))
    for j, lag in enumerate(extra_lags):
        acf_extra[:, j] = corr[:, lag] / (N - lag) / (means**2 + 1e-12)

    # 16 segment variances
    seg_size = N // 16   # 256
    seg_var  = np.zeros((len(i_raw), 16))
    for s in range(16):
        seg_var[:, s] = i_raw[:, s*seg_size:(s+1)*seg_size].var(axis=1)

    return np.column_stack([acf_extra, seg_var])

for split, i_file, x_file, out_file in [
    ('train', 'cache_i_train.npy', 'cache_X_train_psd.npy', 'cache_X_train_extfeat.npy'),
    ('test',  'cache_i_test.npy',  'cache_X_test_psd.npy',  'cache_X_test_extfeat.npy'),
]:
    log(f"Processing {split}...")
    i_raw = np.load(i_file).astype(np.float64)
    X_old = np.load(x_file)
    log(f"  Loaded: i={i_raw.shape}  X={X_old.shape}")

    t0 = time.time()
    X_extra = compute_extra_features(i_raw)
    log(f"  Extra features computed in {time.time()-t0:.1f}s  shape={X_extra.shape}")

    X_ext = np.hstack([X_old, X_extra])
    np.save(out_file, X_ext)
    log(f"  Saved {out_file}  shape={X_ext.shape}")

log("Done.")
