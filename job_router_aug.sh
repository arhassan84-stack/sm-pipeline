#!/bin/bash
#SBATCH --job-name=router_aug
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/router_aug_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/router_aug_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
python pt_router_aug.py
echo "Job finished: $(date)"
