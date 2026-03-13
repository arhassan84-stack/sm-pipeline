#!/bin/bash
#SBATCH --job-name=cache_multidt
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_multidt_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_multidt_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Building multi-dt caches (dt010, dt050, dt100) with shared train/test split"

python build_cache_multidt.py \
    --workers 8 \
    --chunk-size 500 \
    --seed 42 \
    --tag multidt

echo "Job finished: $(date)"
