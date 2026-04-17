#!/bin/bash
#SBATCH --job-name=noise_levels
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/noise_levels_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/noise_levels_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"

python extract_noise_levels.py

echo "Job finished: $(date)"
