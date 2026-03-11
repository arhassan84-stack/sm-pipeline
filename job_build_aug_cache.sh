#!/bin/bash
#SBATCH --job-name=aug_cache
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/aug_cache_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/aug_cache_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"

python build_augmented_cache.py \
    --i_file new_sims_noise0_i.npy \
    --d_file new_sims_noise0_d.npy \
    --workers $SLURM_CPUS_PER_TASK

echo "Job finished: $(date)"
