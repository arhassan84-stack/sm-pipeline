"""
Generate FCS intensity traces via Brownian diffusion simulation.

Translated from MATLAB:
  generateTrainingData.m → simulateDiffusionMaster_2.m
  → simulateDiffusion_2.m + generateTrace_3.m

Fixed parameters (from fixedParameters.mat):
  wxy     = 1e-3 * 0.51 * 488 / 1.2 = 0.2074 µm  (beam waist)
  D range = [0.01, 14.125] µm²/s  (1000 log-spaced values)
  tMax    = 4096 bins,  dt = 1 ms  →  4.096 s total
  maxRate = 50,000 counts/s
  dim     = 2  (2D diffusion)
  r0      = 0  (molecules start at beam center)

Usage:
  python generate_simulations.py --replicas 300
  python generate_simulations.py --replicas 300 --noise 10 --workers 8 --seed 42

Output files:
  {output}_noise{noise:.0f}_i.npy  —  (num_D * replicas, 4096) float32  intensity traces
  {output}_noise{noise:.0f}_d.npy  —  (num_D * replicas,)       float64  diffusion coefficients

Parallelism:
  The 1000 D values are split across --workers processes (default: all CPUs).
  Each worker gets an independent RNG, so results are not reproducible across
  different worker counts even with the same seed — but within a fixed worker
  count + seed, results are fully reproducible.

Memory note:
  replicas=300  → 300k traces → ~4.9 GB
  replicas=500  → 500k traces → ~8.2 GB
"""
import numpy as np
import argparse
import time
import os
from multiprocessing import Pool, cpu_count

# ── Fixed physical parameters (fixedParameters.mat + config) ──────────────────
EX_LAMBDA  = 488.0
NA         = 1.2
WXY        = 1e-3 * 0.51 * EX_LAMBDA / NA      # 0.2074 µm beam waist
D0         = 1.0
D0_MIN     = 0.01                               # µm²/s
D0_MAX     = 14.12537544622754                  # µm²/s
NUM_D      = 1000
MAX_RATE   = 50000.0                            # counts/s
NOISE_LEVELS = [0, 10, 20, 30, 40, 50]          # % of maxRate
T_MAX      = 4096
DT         = 1e-3                               # s per bin
DIM        = 2
MID        = T_MAX // 2 - 1   # MATLAB floor(tMax/2)=2048 (1-indexed) → 2047 (0-indexed)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Worker function (module-level for multiprocessing pickling) ───────────────
def _simulate_chunk(args):
    """
    Simulate one chunk of D values. Called by each worker process.

    args: (D_chunk, num_replicas, back_rate, seed)
      D_chunk      : 1-D array of diffusion coefficients for this worker
      num_replicas : replicas per D value
      back_rate    : background photon rate (counts/s)
      seed         : integer seed for this worker's RNG (None = random)
    """
    D_chunk, num_replicas, back_rate, seed = args

    rng = np.random.default_rng(seed)   # each worker has its own independent RNG

    n_D   = len(D_chunk)
    total = n_D * num_replicas
    intensities = np.zeros((total, T_MAX), dtype=np.float32)
    d_out       = np.zeros(total, dtype=np.float64)

    for i, D in enumerate(D_chunk):
        k = np.sqrt(DIM * D * DT)

        # Brownian walk: r0=0, all start at origin
        # x, y: shape (T_MAX, num_replicas)
        steps_x = k * rng.standard_normal((T_MAX - 1, num_replicas))
        steps_y = k * rng.standard_normal((T_MAX - 1, num_replicas))

        x = np.zeros((T_MAX, num_replicas))
        y = np.zeros((T_MAX, num_replicas))
        x[1:, :] = np.cumsum(steps_x, axis=0)
        y[1:, :] = np.cumsum(steps_y, axis=0)

        # Shift midpoint of trajectory to focus center
        x -= x[MID, :]
        y -= y[MID, :]

        # Gaussian PSF
        em = np.exp(-2.0 * ((x / WXY)**2 + (y / WXY)**2))

        # Poisson photon counts → count rate
        rate       = DT * (back_rate + MAX_RATE * em)
        counts     = rng.poisson(rate)
        int_traces = counts / DT                          # (T_MAX, num_replicas)

        row_s = i * num_replicas
        row_e = row_s + num_replicas
        intensities[row_s:row_e, :] = int_traces.T.astype(np.float32)
        d_out[row_s:row_e]          = D

    return intensities, d_out


def simulate_parallel(D_values, num_replicas, back_rate=0.0, n_workers=1, base_seed=None):
    """
    Distribute D values across n_workers processes, collect and assemble results.

    Each worker receives a contiguous chunk of D values and an independent seed
    derived as base_seed + worker_index  (or None for all workers if base_seed
    is None, giving OS-entropy randomness).
    """
    # Split D values into n_workers chunks (last chunk may be smaller)
    chunks = np.array_split(D_values, n_workers)

    # Build seed for each worker
    if base_seed is not None:
        seeds = [base_seed + i for i in range(n_workers)]
    else:
        seeds = [None] * n_workers

    args_list = [(chunk, num_replicas, back_rate, seed)
                 for chunk, seed in zip(chunks, seeds)]

    with Pool(n_workers) as pool:
        results = pool.map(_simulate_chunk, args_list)

    intensities = np.vstack([r[0] for r in results])
    d_out       = np.concatenate([r[1] for r in results])
    return intensities, d_out


def main():
    parser = argparse.ArgumentParser(
        description='Generate FCS simulation data (parallel Python translation of MATLAB)')
    parser.add_argument('--replicas', type=int, required=True,
                        help='Replicas per D value (e.g. 300 → 300k traces)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help=f'Background noise as %% of maxRate (default: 0). '
                             f'Standard levels: {NOISE_LEVELS}')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help='Number of parallel worker processes (default: all CPUs)')
    parser.add_argument('--output', type=str, default='new_sims',
                        help='Output filename prefix (default: new_sims)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (worker i uses seed+i). Default: no seed.')
    args = parser.parse_args()

    if args.noise < 0 or args.noise > 100:
        raise ValueError(f"--noise must be 0–100, got {args.noise}")
    back_rate = args.noise / 100.0 * MAX_RATE

    total_traces = NUM_D * args.replicas
    mem_gb = total_traces * T_MAX * 4 / 1e9

    log("=" * 60)
    log("FCS Simulation — parallel Python translation of MATLAB")
    log("=" * 60)
    log(f"  wxy        = {WXY:.4f} µm")
    log(f"  D range    = [{D0_MIN}, {D0_MAX:.5f}] µm²/s  ({NUM_D} log-spaced)")
    log(f"  tMax       = {T_MAX}  dt = {DT*1e3:.1f} ms  ({T_MAX*DT:.3f} s/trace)")
    log(f"  maxRate    = {MAX_RATE:.0f} counts/s   backRate = {back_rate:.0f} ({args.noise:.1f}%)")
    log(f"  replicas   = {args.replicas} per D value")
    log(f"  workers    = {args.workers} (of {cpu_count()} available CPUs)")
    log(f"  total      = {total_traces:,} traces")
    log(f"  memory est.= {mem_gb:.1f} GB")
    if args.seed is not None:
        log(f"  seeds      = {args.seed} .. {args.seed + args.workers - 1}")
    log("=" * 60)

    if mem_gb > 12:
        log(f"WARNING: estimated memory {mem_gb:.1f} GB — consider reducing replicas or running on cluster")

    D_values = D0 * np.logspace(np.log10(D0_MIN), np.log10(D0_MAX), NUM_D)

    t0 = time.time()
    log("Starting parallel simulation...")
    intensities, d_values = simulate_parallel(
        D_values, args.replicas,
        back_rate=back_rate,
        n_workers=args.workers,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0
    log(f"Simulation complete in {elapsed:.1f}s  ({elapsed/total_traces*1000:.2f} ms/trace)")
    log(f"  intensities: shape={intensities.shape}  dtype={intensities.dtype}")
    log(f"  d range: [{d_values.min():.5f}, {d_values.max():.5f}] µm²/s")
    log(f"  intensity range: [{intensities.min():.1f}, {intensities.max():.1f}] counts/s")
    log(f"  intensity mean:  {intensities.mean():.1f} counts/s")

    noise_tag = f"noise{args.noise:.0f}"
    i_file = f"{args.output}_{noise_tag}_i.npy"
    d_file = f"{args.output}_{noise_tag}_d.npy"
    np.save(i_file, intensities)
    np.save(d_file, d_values)
    log(f"Saved {i_file}  ({os.path.getsize(i_file)/1e6:.1f} MB)")
    log(f"Saved {d_file}  ({os.path.getsize(d_file)/1e6:.1f} MB)")
    log("Done.")


if __name__ == '__main__':
    main()
