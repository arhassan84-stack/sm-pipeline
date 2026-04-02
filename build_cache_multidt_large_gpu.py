"""
build_cache_multidt_large_gpu.py — GPU-accelerated cache builder for multidt_large.

Replaces the CPU numpy feature pipeline with features_gpu.py, making cached
features identical to the on-the-fly training pipeline (same precision, same
formulas).  Memory-maps the large dt010 intensity output to avoid 132 GB RAM
spike.

Output files (tag=multidt_large):
  cache_{i,X,d}_{train,test}_multidt_large_{dt010,dt050,dt100}.npy
  cache_d_{train,test}_multidt_large.npy
"""

import numpy as np
import torch
import time
import argparse

from features_gpu import make_constants, compute_features_gpu

# ── Config ────────────────────────────────────────────────────────────────────

FEAT_BATCH = 512      # traces per GPU feature-computation batch
S2_CHUNK   = 30       # S2 chunk size (memory vs. speed trade-off)
SAVE_CHUNK = 1_000    # rows per chunk when writing large mmap files

DT_CONFIGS = [
    ('dt010', 'sims_multidt_large_noise0_i_dt010.npy', 40960),
    ('dt050', 'sims_multidt_large_noise0_i_dt050.npy',  8192),
    ('dt100', 'sims_multidt_large_noise0_i_dt100.npy',  4096),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def compute_features_batched(i_mmap, indices, C, device, label=""):
    """Compute features for a set of trace indices using GPU batches."""
    feats = []
    n = len(indices)
    t0 = time.time()
    for start in range(0, n, FEAT_BATCH):
        end = min(start + FEAT_BATCH, n)
        idx = indices[start:end]
        I_gpu = torch.from_numpy(
            np.array(i_mmap[idx], dtype=np.float32)
        ).to(device)
        X_batch = compute_features_gpu(I_gpu, C, S2_CHUNK).cpu().numpy()
        feats.append(X_batch)
        if (start // FEAT_BATCH) % 100 == 0 and start > 0:
            elapsed = time.time() - t0
            log(f"    {label}  {end:,}/{n:,}  ({100*end/n:.1f}%)  {elapsed:.0f}s")
    log(f"    {label}  {n:,}/{n:,}  (100.0%)  {time.time()-t0:.0f}s — done")
    return np.concatenate(feats, axis=0).astype(np.float32)


def save_intensity_mmap(out_path, i_mmap, indices, n_bins):
    """Write intensity rows to a new memory-mapped .npy file in chunks."""
    n = len(indices)
    fp = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32, shape=(n, n_bins))
    for start in range(0, n, SAVE_CHUNK):
        end = min(start + SAVE_CHUNK, n)
        fp[start:end] = i_mmap[indices[start:end]]
    del fp   # flush to disk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated cache builder for multidt_large')
    parser.add_argument('--test-frac', type=float, default=0.10)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--tag',       type=str,   default='multidt_large')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available — aborting. Use --gres=gpu:1 in SLURM.")
    device = torch.device('cuda')
    log(f"Device: {device}  ({torch.cuda.get_device_name(0)})")

    # ── Shared train/test split ────────────────────────────────────────────
    log("Loading D values and creating shared train/test split...")
    d_all   = np.load(f'sims_{args.tag}_noise0_d.npy')
    N_total = len(d_all)
    N_test  = int(N_total * args.test_frac)
    rng     = np.random.default_rng(args.seed)
    test_idx  = rng.choice(N_total, size=N_test, replace=False)
    train_idx = np.setdiff1d(np.arange(N_total), test_idx)

    d_train = d_all[train_idx];  d_test = d_all[test_idx]
    np.save(f'cache_d_train_{args.tag}.npy', d_train)
    np.save(f'cache_d_test_{args.tag}.npy',  d_test)
    log(f"  N_total={N_total:,}  train={len(train_idx):,}  test={len(test_idx):,}")

    # ── Per-dt processing ──────────────────────────────────────────────────
    for dt_tag, i_file, n_bins in DT_CONFIGS:
        log(f"\n{'='*60}")
        log(f"Processing {dt_tag}  ({i_file})  {n_bins} bins/trace")
        t_dt = time.time()

        C = make_constants(n_bins, device)
        log(f"  n_lags={C['n_lags']}  n_short={C['n_short']}")

        log(f"  Memory-mapping {i_file}...")
        i_mmap = np.load(i_file, mmap_mode='r')
        assert i_mmap.shape == (N_total, n_bins), \
            f"Shape mismatch: {i_mmap.shape}"

        out_tag = f'{args.tag}_{dt_tag}'

        # Features (accumulate in RAM — ~1 GB per dt, manageable)
        log(f"  Computing train features ({len(train_idx):,} traces)...")
        X_train = compute_features_batched(
            i_mmap, train_idx, C, device, label="train")
        log(f"  Computing test features ({len(test_idx):,} traces)...")
        X_test  = compute_features_batched(
            i_mmap, test_idx, C, device, label="test")
        log(f"  Feature shape: train={X_train.shape}  test={X_test.shape}")

        # Intensity arrays — chunked mmap save for dt010 (avoids 132 GB spike)
        log(f"  Saving cache_i_train_{out_tag}.npy (chunked)...")
        save_intensity_mmap(f'cache_i_train_{out_tag}.npy',
                            i_mmap, train_idx, n_bins)
        log(f"  Saving cache_i_test_{out_tag}.npy (chunked)...")
        save_intensity_mmap(f'cache_i_test_{out_tag}.npy',
                            i_mmap, test_idx, n_bins)

        np.save(f'cache_X_train_{out_tag}.npy', X_train)
        np.save(f'cache_X_test_{out_tag}.npy',  X_test)
        np.save(f'cache_d_train_{out_tag}.npy', d_train)
        np.save(f'cache_d_test_{out_tag}.npy',  d_test)

        log(f"  Saved all cache_{{i,X,d}}_{{train,test}}_{out_tag}.npy")
        log(f"  {dt_tag} done in {time.time()-t_dt:.0f}s")

        del i_mmap, X_train, X_test

    log("\nAll done.")


if __name__ == '__main__':
    main()
