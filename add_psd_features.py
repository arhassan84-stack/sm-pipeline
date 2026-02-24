"""
Step 1: Compute PSD features from raw traces and append to existing cache.
Also saves raw i matrices for CNN use later.
"""
import numpy as np
import h5py
import time
import os

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── PSD feature extraction ────────────────────────────────────────────────────
def compute_psd_features(i, n_bins=32, label=""):
    N = i.shape[1]
    n_fft = 2 ** int(np.ceil(np.log2(N)))
    log(f"{label} Computing PSD ({len(i)} traces, {n_bins} log-spaced bins)...")

    means = i.mean(axis=1)
    fluct = i - means[:, np.newaxis]
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    psd   = np.abs(F) ** 2 / N          # (n_traces, n_fft//2+1)

    freqs     = np.fft.rfftfreq(n_fft)
    bin_edges = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_bins + 1)

    psd_bins = np.zeros((len(i), n_bins))
    for b in range(n_bins):
        mask = (freqs >= bin_edges[b]) & (freqs < bin_edges[b + 1])
        if mask.sum() > 0:
            psd_bins[:, b] = psd[:, mask].mean(axis=1)

    log(f"{label} PSD done — shape {psd_bins.shape}")
    return psd_bins

# ── Recreate same train/test indices ─────────────────────────────────────────
log("Opening HDF5 file...")
f    = h5py.File('unrollIntensity.mat', 'r')
uInt = f['uInt']

n_all     = uInt[1, :]
valid_idx = np.where(n_all == 0)[0]

rng        = np.random.default_rng(42)
train_idx  = rng.choice(valid_idx, size=len(valid_idx) // 2, replace=False)
train_set  = set(train_idx.tolist())
train_idx.sort()

test_idx_all = np.array([idx for idx in valid_idx if idx not in train_set])
rng2     = np.random.default_rng(99)
test_idx = rng2.choice(test_idx_all, size=len(test_idx_all) // 10, replace=False)
test_idx.sort()

log(f"Train indices: {len(train_idx)}  Test indices: {len(test_idx)}")

# ── Load raw i (and cache it for CNN) ────────────────────────────────────────
if os.path.exists('cache_i_train.npy') and os.path.exists('cache_i_test.npy'):
    log("Raw i cache found — loading...")
    i_train = np.load('cache_i_train.npy')
    i_test  = np.load('cache_i_test.npy')
else:
    log("Reading train i from HDF5...")
    i_train = uInt[2:, train_idx].T.astype(np.float32)   # float32 to save space
    log(f"  Train i shape: {i_train.shape}  — saving cache...")
    np.save('cache_i_train.npy', i_train)

    log("Reading test i from HDF5...")
    i_test = uInt[2:, test_idx].T.astype(np.float32)
    log(f"  Test i shape: {i_test.shape}  — saving cache...")
    np.save('cache_i_test.npy', i_test)

# ── Compute PSD features ──────────────────────────────────────────────────────
psd_train = compute_psd_features(i_train.astype(np.float64), label="[Train]")
psd_test  = compute_psd_features(i_test.astype(np.float64),  label="[Test]")

# ── Append to existing cache and save ────────────────────────────────────────
log("Loading existing feature cache...")
X_train = np.load('cache_X_train.npy')
X_test  = np.load('cache_X_test.npy')

X_train_psd = np.hstack([X_train, psd_train])
X_test_psd  = np.hstack([X_test,  psd_test])

log(f"New feature shapes — Train: {X_train_psd.shape}  Test: {X_test_psd.shape}")

np.save('cache_X_train_psd.npy', X_train_psd)
np.save('cache_X_test_psd.npy',  X_test_psd)
log("Saved cache_X_train_psd.npy and cache_X_test_psd.npy")
log("Done.")
