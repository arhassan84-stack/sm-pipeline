#!/bin/bash
#SBATCH --job-name=plot_noise_3dt
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/plot_noise_3dt_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/plot_noise_3dt_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
python plot_noise_3dt.py
echo "Job finished: $(date)"
