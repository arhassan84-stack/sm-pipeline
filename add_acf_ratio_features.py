"""
Append 19 log-ACF ratio features to the 283-feature caches.

Features: log(G[k+1] / G[k]) for k = 1..19  (consecutive lag pairs, lags 1-20)
These encode the local ACF decay rate and are ~4x more discriminative across
d sub-bins than raw ACF values — especially useful for d>=1.

Col mapping for lags 1-20 (all already in the 283-feature cache):
  lag  1 -> col 10    lag  5 -> col 222   lag  9 -> col 16
  lag  2 -> col 11    lag  6 -> col 14    lag 10 -> col 224
  lag  3 -> col 12    lag  7 -> col 15    lag 11 -> col 225
  lag  4 -> col 13    lag  8 -> col 223   lag 12 -> col 17
  lag 13 -> col 226   lag 16 -> col 228   lag 19 -> col 19
  lag 14 -> col 227   lag 17 -> col 229   lag 20 -> col 231
  lag 15 -> col 18    lag 18 -> col 230

Updates in-place:
  cache_X_train_90pct.npy  (n, 283) -> (n, 302)
  cache_X_test_90pct.npy   (n, 283) -> (n, 302)
"""
import numpy as np, time

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Sorted (lag, column) pairs for lags 1-20
LAG_COL = [
    ( 1, 10), ( 2, 11), ( 3, 12), ( 4, 13), ( 5,222),
    ( 6, 14), ( 7, 15), ( 8,223), ( 9, 16), (10,224),
    (11,225), (12, 17), (13,226), (14,227), (15, 18),
    (16,228), (17,229), (18,230), (19, 19), (20,231),
]
COLS_BY_LAG = {lag: col for lag, col in LAG_COL}   # lag -> col index in 283-feat cache


def compute_acf_ratios(X):
    """(n, 283) -> (n, 19) log-ratio features"""
    n = X.shape[0]
    out = np.zeros((n, 19), dtype=np.float64)
    for i in range(19):            # 19 consecutive pairs: (lag 1,2), ..., (lag 19,20)
        lag_lo, lag_hi = i + 1, i + 2
        col_lo = COLS_BY_LAG[lag_lo]
        col_hi = COLS_BY_LAG[lag_hi]
        g_lo = np.maximum(X[:, col_lo], 1e-10)
        g_hi = np.maximum(X[:, col_hi], 1e-10)
        out[:, i] = np.log(g_hi / g_lo)
    return out


for split in ('train', 'test'):
    log(f"=== Processing {split}_90pct ===")
    fname = f"cache_X_{split}_90pct.npy"
    X = np.load(fname)
    log(f"  Loaded {fname}  shape={X.shape}")
    assert X.shape[1] == 283, f"Expected 283 features, got {X.shape[1]}"

    t0 = time.time()
    ratios = compute_acf_ratios(X)
    log(f"  Computed log-ACF ratios in {time.time()-t0:.1f}s  shape={ratios.shape}")

    X_new = np.concatenate([X, ratios], axis=1)
    np.save(fname, X_new)
    log(f"  Saved {fname}  shape={X_new.shape}")

log("Done.  Feature count: 283 -> 302")
log("New cols 283-301: log(G[k+1]/G[k]) for k=1..19 (consecutive lag pairs, lags 1-20)")
