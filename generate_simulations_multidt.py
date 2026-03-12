"""
Multi-dt FCS simulation generator.

Computes a single molecular diffusion trajectory at the finest dt (dt_min),
then derives intensity traces at all requested dt values by subsampling the
shared trajectory.  All dt versions share the identical underlying molecular
motion — only the photon-counting integration window differs.

Physics:
  Trajectory at dt_min: x[t+1] = x[t] + sqrt(2*D*dt_min) * N(0,1)
  Coarser dt_i = n_i * dt_min: subsample every n_i steps.
  Intensity at coarse step j: counts ~ Poisson(dt_i * (noise + MAX_RATE * PSF(x_j, y_j)))
                               I[j] = counts / dt_i  (counts/s)

Benefits over independent simulations:
  - 1 trajectory simulation instead of N_dt simulations
  - Correlated dataset: same molecule, different measurement resolution
  - ~N_dt× faster total simulation time

Output files  (tag default = "multidt"):
  sims_{tag}_noise0_d.npy              (N_total,)            D values
  sims_{tag}_noise0_i_dt010.npy        (N_total, 40960)      dt=0.1ms
  sims_{tag}_noise0_i_dt020.npy        (N_total, 20480)      dt=0.2ms
  sims_{tag}_noise0_i_dt040.npy        (N_total, 10240)      dt=0.4ms
  sims_{tag}_noise0_i_dt060.npy        (N_total,  6826)      dt=0.6ms
  sims_{tag}_noise0_i_dt080.npy        (N_total,  5120)      dt=0.8ms
  sims_{tag}_noise0_i_dt100.npy        (N_total,  4096)      dt=1.0ms

  N_bins for dt_i = floor(N_max / round(dt_i / dt_min))
  (may differ by ±1 from standalone simulations for non-binary multiples of dt_min)

Usage:
  python generate_simulations_multidt.py \\
      --dts 0.1 0.2 0.4 0.6 0.8 1.0 \\
      --replicas 300 \\
      --workers 8 \\
      --seed 42
"""

import numpy as np
import argparse
import time
import os
from multiprocessing import Pool, cpu_count

# ── Fixed physical parameters (identical to generate_simulations_dt.py) ──────
EX_LAMBDA = 488.0
NA        = 1.2
WXY       = 1e-3 * 0.51 * EX_LAMBDA / NA   # 0.2074 µm
D0_MIN    = 0.01
D0_MAX    = 14.12537544622754
NUM_D     = 1000
MAX_RATE  = 50000.0                          # counts/s
T_MAX_MS  = 4096                             # total trace duration (ms)
DIM       = 2                                # 2D diffusion

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Worker ────────────────────────────────────────────────────────────────────

def _simulate_multidt_chunk(args):
    """
    Simulate a chunk of D values at dt_min; derive intensities for all dt_configs.

    Returns: (chunk_idx, {tag: intensity_array (n_traces, N_bins)}, d_array)
    """
    chunk_idx, D_chunk, num_replicas, back_rate, seed, \
        DT_MIN_S, N_MAX, MID, dt_configs = args

    rng   = np.random.default_rng(seed)
    n_D   = len(D_chunk)
    total = n_D * num_replicas

    # Pre-allocate output per dt tag
    out = {tag: np.zeros((total, N_bins_i), dtype=np.float32)
           for (tag, dt_s, n_sub, N_bins_i) in dt_configs}
    d_out = np.zeros(total, dtype=np.float64)

    for i, D in enumerate(D_chunk):
        # ── Fine trajectory at dt_min ─────────────────────────────────────
        k = np.sqrt(DIM * D * DT_MIN_S)
        steps_x = k * rng.standard_normal((N_MAX - 1, num_replicas))
        steps_y = k * rng.standard_normal((N_MAX - 1, num_replicas))
        x = np.zeros((N_MAX, num_replicas))
        y = np.zeros((N_MAX, num_replicas))
        x[1:] = np.cumsum(steps_x, axis=0)
        y[1:] = np.cumsum(steps_y, axis=0)
        # Centre on midpoint (same convention as standalone generator)
        x -= x[MID]
        y -= y[MID]

        # ── Derive intensity traces for each dt ───────────────────────────
        for (tag, dt_s, n_sub, N_bins_i) in dt_configs:
            x_s = x[::n_sub][:N_bins_i]       # (N_bins_i, num_replicas)
            y_s = y[::n_sub][:N_bins_i]
            em     = np.exp(-2.0 * ((x_s / WXY)**2 + (y_s / WXY)**2))
            rate   = dt_s * (back_rate + MAX_RATE * em)
            counts = rng.poisson(rate)
            row_s  = i * num_replicas
            row_e  = row_s + num_replicas
            out[tag][row_s:row_e] = (counts / dt_s).astype(np.float32).T

        d_out[i * num_replicas:(i + 1) * num_replicas] = D

    return chunk_idx, out, d_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-dt FCS traces from shared molecular trajectories')
    parser.add_argument('--dts', type=float, nargs='+',
                        default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help='dt values in ms (default: 0.1 0.2 0.4 0.6 0.8 1.0)')
    parser.add_argument('--replicas', type=int, required=True,
                        help='Replicas per D value (e.g. 300)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Background noise as %% of maxRate (default: 0)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Worker processes (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--tag', type=str, default='multidt',
                        help='Output filename prefix (default: multidt)')
    args = parser.parse_args()

    DT_MIN_MS = min(args.dts)
    DT_MIN_S  = DT_MIN_MS * 1e-3
    N_MAX     = round(T_MAX_MS / DT_MIN_MS)
    MID       = N_MAX // 2 - 1
    noise_tag = f"noise{args.noise:.0f}"
    back_rate = args.noise / 100.0 * MAX_RATE

    # Build dt config list (sorted finest → coarsest)
    dt_configs = []
    for dt_ms in sorted(args.dts):
        dt_s    = dt_ms * 1e-3
        n_sub   = round(dt_ms / DT_MIN_MS)
        N_bins_i = N_MAX // n_sub
        tag_i   = f"dt{round(dt_ms * 100):03d}"
        dt_configs.append((tag_i, dt_s, n_sub, N_bins_i))

    total_traces = NUM_D * args.replicas
    total_gb     = sum(total_traces * nb * 4 / 1e9 for (_, _, _, nb) in dt_configs)

    log("=" * 60)
    log(f"Multi-dt FCS Simulation  tag={args.tag}")
    log("=" * 60)
    log(f"  dt values  : {[(c[0], c[3]) for c in dt_configs]}")
    log(f"  dt_min     = {DT_MIN_MS} ms  →  N_MAX = {N_MAX} bins/trace")
    log(f"  D range    = [{D0_MIN}, {D0_MAX:.5f}] µm²/s  ({NUM_D} log-spaced)")
    log(f"  replicas   = {args.replicas}  →  {total_traces:,} traces total")
    log(f"  workers    = {args.workers}")
    log(f"  Output     ≈ {total_gb:.1f} GB total (memmap-backed)")
    log("=" * 60)

    D_values = np.logspace(np.log10(D0_MIN), np.log10(D0_MAX), NUM_D)

    # Pre-allocate memory-mapped output files on disk
    log("Creating output memmaps on disk...")
    out_memmaps = {}
    for (tag_i, _, _, N_bins_i) in dt_configs:
        fname = f'sims_{args.tag}_{noise_tag}_i_{tag_i}.npy'
        out_memmaps[tag_i] = np.lib.format.open_memmap(
            fname, mode='w+', dtype=np.float32, shape=(total_traces, N_bins_i))
        log(f"  {fname}  ({total_traces} × {N_bins_i})  "
            f"{total_traces * N_bins_i * 4 / 1e9:.1f} GB")
    out_d = np.zeros(total_traces, dtype=np.float64)

    # Chunk finely so each worker returns manageable data (< 1 GB per chunk)
    # Target: ~500 MB per chunk → n_D_per_chunk ≈ 500e6 / (replicas * sum_N_bins * 4)
    sum_N_bins = sum(nb for (_, _, _, nb) in dt_configs)
    target_bytes = 512 * 1024**2   # 512 MB per chunk
    n_D_per_chunk = max(1, int(target_bytes / (args.replicas * sum_N_bins * 4)))
    n_D_per_chunk = min(n_D_per_chunk, NUM_D // args.workers)   # at least workers chunks
    n_chunks = int(np.ceil(NUM_D / n_D_per_chunk))

    log(f"Chunking: {n_D_per_chunk} D values/chunk  →  {n_chunks} chunks  "
        f"(~{n_D_per_chunk * args.replicas * sum_N_bins * 4 / 1e6:.0f} MB/chunk)")

    D_chunks = [D_values[i * n_D_per_chunk: (i + 1) * n_D_per_chunk]
                for i in range(n_chunks)]
    chunk_starts = []
    pos = 0
    for dc in D_chunks:
        chunk_starts.append(pos)
        pos += len(dc) * args.replicas

    args_list = [
        (i, D_chunks[i], args.replicas, back_rate, args.seed + i,
         DT_MIN_S, N_MAX, MID, dt_configs)
        for i in range(n_chunks)
    ]

    log("Starting parallel simulation (imap_unordered — writes on completion)...")
    t0 = time.time()
    n_done = 0

    with Pool(args.workers) as pool:
        for chunk_idx, chunk_out, chunk_d in pool.imap_unordered(
                _simulate_multidt_chunk, args_list):
            row_s = chunk_starts[chunk_idx]
            row_e = row_s + len(chunk_d)
            # Write to memmaps immediately — chunk arrays freed after this loop body
            out_d[row_s:row_e] = chunk_d
            for tag_i, arr in chunk_out.items():
                out_memmaps[tag_i][row_s:row_e] = arr
            n_done += len(chunk_d)
            elapsed = time.time() - t0
            if n_done % max(1, total_traces // 20) < len(chunk_d):
                log(f"  {n_done:,}/{total_traces:,} traces  {elapsed:.1f}s  "
                    f"({n_done/elapsed:.0f} traces/s)")

    elapsed = time.time() - t0
    log(f"Simulation complete in {elapsed:.1f}s  ({total_traces/elapsed:.0f} traces/s)")

    # Save D values
    d_fname = f'sims_{args.tag}_{noise_tag}_d.npy'
    np.save(d_fname, out_d)
    log(f"Saved {d_fname}  ({os.path.getsize(d_fname)/1e3:.1f} KB)")

    # Flush and close memmaps
    for tag_i, mm in out_memmaps.items():
        mm.flush()
        fname = f'sims_{args.tag}_{noise_tag}_i_{tag_i}.npy'
        log(f"Saved {fname}  ({os.path.getsize(fname)/1e9:.2f} GB)")
        del mm

    log(f"Total time: {time.time() - t0:.1f}s")
    log("Done.")


if __name__ == '__main__':
    main()
