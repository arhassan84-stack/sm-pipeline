#!/bin/bash
# Build fused training caches: raw aug + smoothed{W} stacked → 2N samples.
# Fast pure-numpy concatenation, no feature recomputation.
# One task covers all requested windows (takes <5 min total).
#
#SBATCH --job-name=fused_cache
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/fused_cache_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/fused_cache_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"

# Build fused caches for all 5 windows
python build_fused_cache.py --windows 3 5 10 20 50

echo "Job finished: $(date)"
