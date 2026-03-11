#!/bin/bash
# SLURM array job: one task per smoothing window, runs all 5 in parallel.
# Array indices 0-4 map to windows [3, 5, 10, 20, 50].
# Runs a single window per job to avoid loading all windows into memory at once.
# 8 workers × ~48k traces each keeps peak memory well within 128G.
#
#SBATCH --job-name=smooth_cache
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=1
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/smooth_cache_%A_%a.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/smooth_cache_%A_%a.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

# Map array index → window size
WINDOWS=(3 5 10 20 50)
W=${WINDOWS[$SLURM_ARRAY_TASK_ID]}

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID  Window: W=$W  CPUs: $SLURM_CPUS_PER_TASK"

# --chunk-size 10000: each worker handles ~10k traces at a time (~3GB peak per worker)
# 8 workers × 3GB = 24GB active + ~20GB parent/overhead → safe within 64G
python build_smoothed_cache.py --windows $W --workers $SLURM_CPUS_PER_TASK --chunk-size 10000

echo "Job finished: $(date)"
