"""
Generate FCS intensity traces via Brownian diffusion simulation — variable dt.

Identical physics to generate_simulations.py, but dt is a command-line argument
so we can produce training data at different sampling rates (0.25 ms, 0.5 ms, …).

Fixed parameters:
  wxy     = 1e-3 * 0.51 * 488 / 1.2 = 0.2074 µm  (beam waist)
  D range = [0.01, 14.125] µm²/s  (1000 log-spaced values)
  tMax    = 4096 bins
  maxRate = 50,000 counts/s
  dim     = 2  (2D diffusion)
  r0      = 0  (molecules start at beam center)

The only physics change vs generate_simulations.py is:
  k = sqrt(DIM * D * DT)  — smaller DT → smaller step size → finer trajectory
  Total trace duration = tMax * DT  (shorter for smaller DT)

Output tag convention (DT in ms, scaled ×100, zero-padded to 3 digits):
  dt=0.25 ms → tag "dt025"   files: sims_dt025_noise{n}_{i,d}.npy
  dt=0.50 ms → tag "dt050"   files: sims_dt050_noise{n}_{i,d}.npy
  dt=1.00 ms → tag "dt100"   files: sims_dt100_noise{n}_{i,d}.npy

Usage:
  python generate_simulations_dt.py --dt 0.25 --replicas 300
  python generate_simulations_dt.py --dt 0.5  --replicas 300 --workers 16 --seed 42
"""
import numpy as np
import argparse
import time
import os
from multiprocessing import Pool, cpu_count

# ── Fixed physical parameters ─────────────────────────────────────────────────
EX_LAMBDA  = 488.0
NA         = 1.2
WXY        = 1e-3 * 0.51 * EX_LAMBDA / NA   # 0.2074 µm
D0         = 1.0
D0_MIN     = 0.01
D0_MAX     = 14.12537544622754
NUM_D      = 1000
MAX_RATE   = 50000.0                          # counts/s
T_MAX_MS   = 4096                             # total trace duration in ms (fixed)
DIM        = 2
# N_BINS and MID are computed at runtime from --dt

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _simulate_chunk(args):
    """Simulate one chunk of D values. Called by each worker process."""
    D_chunk, num_replicas, back_rate, seed, DT, N_BINS, MID = args
    rng = np.random.default_rng(seed)
    n_D   = len(D_chunk)
    total = n_D * num_replicas
    intensities = np.zeros((total, N_BINS), dtype=np.float32)
    d_out       = np.zeros(total, dtype=np.float64)

    for i, D in enumerate(D_chunk):
        k = np.sqrt(DIM * D * DT)
        steps_x = k * rng.standard_normal((N_BINS - 1, num_replicas))
        steps_y = k * rng.standard_normal((N_BINS - 1, num_replicas))
        x = np.zeros((N_BINS, num_replicas))
        y = np.zeros((N_BINS, num_replicas))
        x[1:, :] = np.cumsum(steps_x, axis=0)
        y[1:, :] = np.cumsum(steps_y, axis=0)
        x -= x[MID, :]
        y -= y[MID, :]
        em         = np.exp(-2.0 * ((x / WXY)**2 + (y / WXY)**2))
        rate       = DT * (back_rate + MAX_RATE * em)
        counts     = rng.poisson(rate)
        int_traces = counts / DT
        row_s = i * num_replicas
        row_e = row_s + num_replicas
        intensities[row_s:row_e, :] = int_traces.T.astype(np.float32)
        d_out[row_s:row_e]          = D

    return intensities, d_out


def simulate_parallel(D_values, num_replicas, DT, N_BINS, MID,
                      back_rate=0.0, n_workers=1, base_seed=None):
    chunks = np.array_split(D_values, n_workers)
    seeds  = [base_seed + i if base_seed is not None else None for i in range(n_workers)]
    args_list = [(chunk, num_replicas, back_rate, seed, DT, N_BINS, MID)
                 for chunk, seed in zip(chunks, seeds)]
    with Pool(n_workers) as pool:
        results = pool.map(_simulate_chunk, args_list)
    intensities = np.vstack([r[0] for r in results])
    d_out       = np.concatenate([r[1] for r in results])
    return intensities, d_out


def main():
    parser = argparse.ArgumentParser(
        description='Generate FCS simulation data at arbitrary dt')
    parser.add_argument('--dt', type=float, required=True,
                        help='Time step in ms (e.g. 0.25 or 0.5)')
    parser.add_argument('--replicas', type=int, required=True,
                        help='Replicas per D value (e.g. 300 → 300k traces)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Background noise as %% of maxRate (default: 0)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help='Parallel worker processes (default: all CPUs)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (worker i uses seed+i)')
    args = parser.parse_args()

    DT_s   = args.dt * 1e-3                   # convert ms → seconds
    N_BINS = round(T_MAX_MS / args.dt)        # bins = 4096ms / dt_ms
    MID    = N_BINS // 2 - 1                  # midpoint index (0-based)
    # Tag: dt×100 zero-padded to 3 digits  (0.25ms → "025", 0.5ms → "050", 1ms → "100")
    tag    = f"dt{round(args.dt * 100):03d}"

    if args.noise < 0 or args.noise > 100:
        raise ValueError(f"--noise must be 0–100, got {args.noise}")
    back_rate     = args.noise / 100.0 * MAX_RATE
    total_traces  = NUM_D * args.replicas
    mem_gb        = total_traces * N_BINS * 4 / 1e9

    log("=" * 60)
    log(f"FCS Simulation  dt={args.dt} ms  tag={tag}")
    log("=" * 60)
    log(f"  wxy        = {WXY:.4f} µm")
    log(f"  D range    = [{D0_MIN}, {D0_MAX:.5f}] µm²/s  ({NUM_D} log-spaced)")
    log(f"  tMax       = {T_MAX_MS} ms  dt = {args.dt} ms  →  {N_BINS} bins/trace")
    log(f"  maxRate    = {MAX_RATE:.0f} counts/s   backRate = {back_rate:.0f} ({args.noise:.1f}%)")
    log(f"  replicas   = {args.replicas} per D value  →  {total_traces:,} traces total")
    log(f"  workers    = {args.workers}  memory est. = {mem_gb:.1f} GB")
    log("=" * 60)

    D_values = D0 * np.logspace(np.log10(D0_MIN), np.log10(D0_MAX), NUM_D)

    t0 = time.time()
    log("Starting parallel simulation...")
    intensities, d_values = simulate_parallel(
        D_values, args.replicas, DT_s, N_BINS, MID,
        back_rate=back_rate,
        n_workers=args.workers,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0
    log(f"Simulation complete in {elapsed:.1f}s")
    log(f"  shape={intensities.shape}  d range=[{d_values.min():.5f}, {d_values.max():.5f}]")

    noise_tag = f"noise{args.noise:.0f}"
    i_file = f"sims_{tag}_{noise_tag}_i.npy"
    d_file = f"sims_{tag}_{noise_tag}_d.npy"
    np.save(i_file, intensities)
    np.save(d_file, d_values)
    log(f"Saved {i_file}  ({os.path.getsize(i_file)/1e6:.1f} MB)")
    log(f"Saved {d_file}  ({os.path.getsize(d_file)/1e6:.1f} MB)")
    log("Done.")


if __name__ == '__main__':
    main()
