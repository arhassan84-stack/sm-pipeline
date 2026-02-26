"""
Build peak-centered cropped trace caches.

For each 4096-point trace, find the intensity peak and extract CROP_W points
centered at that peak (zero-padding if the peak is within CROP_W/2 of either
boundary). This focuses the CNN input on the burst region:
  - d>=1 (fast):  burst ~33 pts wide → W=512 captures essentially 100% of signal
  - d<1  (slow):  burst ~918 pts wide → W=512 captures ~36% (the peak region)

Outputs:
  cache_i_train_peakcrop.npy  (90000, CROP_W)
  cache_i_test_peakcrop.npy   (10000, CROP_W)
"""
import numpy as np, time

CROP_W = 512   # width of peak-centered crop

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_crop(traces, W):
    """(n, N) -> (n, W)  peak-centered crop with zero-padding."""
    n, N = traces.shape
    half = W // 2
    out = np.zeros((n, W), dtype=traces.dtype)
    peaks = np.argmax(traces, axis=1)   # (n,)
    for i in range(n):
        pk = peaks[i]
        lo_src = max(0, pk - half)
        hi_src = min(N, pk + half)
        # destination offsets handle boundary zero-padding automatically
        lo_dst = lo_src - (pk - half)   # = 0 if pk>=half, else offset
        hi_dst = lo_dst + (hi_src - lo_src)
        out[i, lo_dst:hi_dst] = traces[i, lo_src:hi_src]
    return out


for split in ('train', 'test'):
    log(f"=== {split} ===")
    fname_in  = f"cache_i_{split}_90pct.npy"
    fname_out = f"cache_i_{split}_peakcrop.npy"
    t0 = time.time()
    traces = np.load(fname_in)
    log(f"  Loaded {fname_in}  shape={traces.shape}")
    cropped = peak_crop(traces, CROP_W)
    log(f"  Cropped in {time.time()-t0:.1f}s  shape={cropped.shape}")
    np.save(fname_out, cropped)
    log(f"  Saved {fname_out}")

    # Quick sanity check: signal capture fraction
    d = np.load(f"cache_d_{split}_90pct.npy")
    mask = d <= 10
    d, traces, cropped = d[mask], traces[mask], cropped[mask]
    mf = d >= 1.0
    ms = d < 1.0
    frac_fast = (cropped[mf].sum(axis=1) / (traces[mf].sum(axis=1) + 1e-10)).mean()
    frac_slow = (cropped[ms].sum(axis=1) / (traces[ms].sum(axis=1) + 1e-10)).mean()
    log(f"  Signal captured: d>=1={frac_fast*100:.1f}%  d<1={frac_slow*100:.1f}%")

log(f"Done. CROP_W={CROP_W}")
