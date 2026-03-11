#!/bin/bash
# Build feature cache for dt050 simulation data.
# Runs independently — does not depend on dt025.
#
#SBATCH --job-name=cache_dt050
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt050_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt050_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"

python build_cache_dt.py \
    --dt_tag dt050 \
    --i_file sims_dt050_noise0_i.npy \
    --d_file sims_dt050_noise0_d.npy \
    --workers $SLURM_CPUS_PER_TASK \
    --chunk-size 5000

echo "Job finished: $(date)"
