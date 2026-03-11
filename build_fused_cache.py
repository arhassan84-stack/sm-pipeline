"""
Build fused feature caches: raw features + smoothed features concatenated column-wise.

For each smoothing window W, combines raw and smoothed features into a single
604-dimensional feature vector per trace (302 raw + 302 smoothed):

  cache_X_train_aug.npy     (N, 302)  raw train features
  cache_X_train_smooth{W}.npy (N, 302)  smooth train features
  → cache_X_train_fused{W}.npy  (N, 604)  [raw_feats | smooth_feats]

  cache_X_test_90pct.npy    (M, 302)  raw test features
  cache_X_test_smooth{W}.npy  (M, 302)  smooth test features
  → cache_X_test_fused{W}.npy   (M, 604)  [raw_feats | smooth_feats]

Training traces (raw + smooth) are NOT duplicated here — the training script
loads both trace arrays from existing files and concatenates them along the
time axis to form 8192-point inputs (4096 raw + 4096 smooth).

Same number of samples as the raw-only caches (390k train, ~10k test).
Labels unchanged: cache_d_train_aug.npy / cache_d_test_90pct.npy.

Usage:
  python build_fused_cache.py --windows 3 5 10 20 50
"""
import numpy as np, argparse, time

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, nargs='+', default=[3, 5, 10, 20, 50])
args = parser.parse_args()

t0 = time.time()

# Load raw caches once (shared across all windows)
log("Loading raw feature caches...")
X_raw_tr = np.load('cache_X_train_aug.npy').astype(np.float32)   # (N, 302)
X_raw_te = np.load('cache_X_test_90pct.npy').astype(np.float32)  # (M, 302)
log(f"  Train raw features: {X_raw_tr.shape}  Test raw features: {X_raw_te.shape}")

for W in args.windows:
    log(f"\n=== Window W={W} ===")
    t_w = time.time()

    X_sm_tr = np.load(f'cache_X_train_smooth{W}.npy').astype(np.float32)  # (N, 302)
    X_sm_te = np.load(f'cache_X_test_smooth{W}.npy').astype(np.float32)   # (M, 302)
    log(f"  Smooth train: {X_sm_tr.shape}  Smooth test: {X_sm_te.shape}")

    # Sanity check: label alignment (just shape check here; training script verifies values)
    assert X_raw_tr.shape[0] == X_sm_tr.shape[0], "Train size mismatch!"
    assert X_raw_te.shape[0] == X_sm_te.shape[0], "Test size mismatch!"

    # Concatenate features column-wise: [raw_feats | smooth_feats] → (N, 604)
    X_fused_tr = np.hstack([X_raw_tr, X_sm_tr])
    X_fused_te = np.hstack([X_raw_te, X_sm_te])
    log(f"  Fused train features: {X_fused_tr.shape}  test: {X_fused_te.shape}")

    np.save(f'cache_X_train_fused{W}.npy', X_fused_tr)
    np.save(f'cache_X_test_fused{W}.npy',  X_fused_te)
    log(f"  Saved cache_X_{{train,test}}_fused{W}.npy  ({time.time()-t_w:.1f}s)")

log(f"\n=== Done. Total: {time.time()-t0:.1f}s ===")
log("Trace arrays: training script loads cache_i_train_aug.npy + cache_i_train_smooth{W}.npy")
log("Labels: cache_d_train_aug.npy (train) and cache_d_test_90pct.npy (test) — unchanged")
