"""
Append 5 transit-time features to the 90pct feature caches.

Transit time at threshold p: the fraction of time points where
  intensity >= p * max(intensity)
computed for p = 0.80, 0.60, 0.40, 0.20, 0.10.

Updates in-place:
  cache_X_train_90pct.npy  (90k, 207) → (90k, 212)
  cache_X_test_90pct.npy   (10k, 207) → (10k, 212)
"""
import numpy as np
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

THRESHOLDS = [0.80, 0.60, 0.40, 0.20, 0.10]

def compute_transit_times(i, thresholds=THRESHOLDS):
    """
    i: (n_traces, N) float array of raw intensity traces.
    Returns (n_traces, len(thresholds)) array.
    Each column k is the fraction of time points where
      i[t] >= thresholds[k] * max(i)
    i.e. transit_time[k] = sum(i >= thr * max) / N
    """
    N = i.shape[1]
    i_max = i.max(axis=1, keepdims=True)   # (n_traces, 1)
    result = np.zeros((len(i), len(thresholds)), dtype=np.float64)
    for j, thr in enumerate(thresholds):
        result[:, j] = (i >= thr * i_max).sum(axis=1) / N
    return result

for split, i_file, x_file in [
    ('train_90pct', 'cache_i_train_90pct.npy', 'cache_X_train_90pct.npy'),
    ('test_90pct',  'cache_i_test_90pct.npy',  'cache_X_test_90pct.npy'),
]:
    log(f"Processing {split}...")

    log(f"  Loading raw traces from {i_file}...")
    i_raw = np.load(i_file).astype(np.float64)
    log(f"  Traces shape: {i_raw.shape}")

    t0 = time.time()
    tt = compute_transit_times(i_raw)
    log(f"  Transit times computed in {time.time()-t0:.1f}s  shape={tt.shape}")
    log(f"  Threshold fractions (mean over traces): "
        + "  ".join(f"thr={p:.0%}: {tt[:,j].mean():.4f}" for j, p in enumerate(THRESHOLDS)))

    log(f"  Loading existing features from {x_file}...")
    X_old = np.load(x_file)
    log(f"  Existing features shape: {X_old.shape}")

    if X_old.shape[1] == 212:
        log(f"  WARNING: cache already has 212 features — stripping old transit cols first")
        X_old = X_old[:, :207]

    X_new = np.hstack([X_old, tt])
    np.save(x_file, X_new)
    log(f"  Saved {x_file}  shape={X_new.shape}")
    log(f"  New feature layout: cols 0-206 unchanged, cols 207-211 = transit time at "
        + ", ".join(f"{p:.0%}" for p in THRESHOLDS))

log("Done. Feature count: 207 → 212")
log("New features (cols 207-211): transit_80, transit_60, transit_40, transit_20, transit_10")
