"""
Build raw trace caches for the 90% train / 10% test split.
Outputs:
  cache_i_train_90pct.npy  — (90k, 4096) raw intensity traces for train
  cache_i_test_90pct.npy   — (~10k, 4096) raw intensity traces for test
Uses the exact same index logic as build_cache_90pct.py.
"""
import numpy as np
import h5py
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Opening unrollIntensity.mat...")
f    = h5py.File('unrollIntensity.mat', 'r')
uInt = f['uInt']

log("Reconstructing index split...")
n_all     = uInt[1, :]
valid_idx = np.where(n_all == 0)[0]          # 100k rows

rng           = np.random.default_rng(42)
old_train     = rng.choice(valid_idx, size=len(valid_idx) // 2, replace=False)
old_train_set = set(old_train.tolist())
old_train.sort()

remaining       = np.array([i for i in valid_idx if i not in old_train_set])
rng2            = np.random.default_rng(7)
extra_train_idx = rng2.choice(remaining, size=40000, replace=False)
extra_train_set = set(extra_train_idx.tolist())
extra_train_idx.sort()

test_idx = np.array([i for i in remaining if i not in extra_train_set])
test_idx.sort()

log(f"Old train: {len(old_train)}  Extra train: {len(extra_train_idx)}  Test: {len(test_idx)}")

log("Reading extra train raw traces from HDF5 (40k × 4096)...")
t0 = time.time()
i_extra = uInt[2:, extra_train_idx].T.astype(np.float32)
log(f"  Done in {time.time()-t0:.1f}s  shape={i_extra.shape}")

log("Reading test raw traces from HDF5 (~10k × 4096)...")
t0 = time.time()
i_test_new = uInt[2:, test_idx].T.astype(np.float32)
log(f"  Done in {time.time()-t0:.1f}s  shape={i_test_new.shape}")

f.close()

log("Loading existing 50k train traces...")
i_old = np.load('cache_i_train.npy')   # (50000, 4096)

log("Combining 50k + 40k train traces...")
i_train_90 = np.vstack([i_old, i_extra])   # (90000, 4096)
log(f"  Combined shape: {i_train_90.shape}")

log("Saving cache_i_train_90pct.npy...")
np.save('cache_i_train_90pct.npy', i_train_90)
log(f"  Saved  ({i_train_90.nbytes / 1e9:.2f} GB)")

log("Saving cache_i_test_90pct.npy...")
np.save('cache_i_test_90pct.npy', i_test_new)
log(f"  Saved  ({i_test_new.nbytes / 1e6:.0f} MB)")

log("Done.")
