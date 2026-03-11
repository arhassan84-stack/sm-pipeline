#!/bin/bash
#SBATCH --job-name=cache_dt010
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt010_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt010_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "dt=0.1ms  N_BINS=40960  building feature cache"

python build_cache_dt.py \
    --dt_tag dt010 \
    --i_file sims_dt010_noise0_i.npy \
    --d_file sims_dt010_noise0_d.npy \
    --workers $SLURM_CPUS_PER_TASK \
    --chunk-size 500

echo "Job finished: $(date)"
