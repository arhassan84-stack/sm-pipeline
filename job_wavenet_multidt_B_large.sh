#!/bin/bash
#SBATCH --job-name=wn_multidt_B_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_B_large_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_B_large_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "multidt_B_large: 3x dataset (900k samples), GPU-built cache"

python pt_wavenet_multidt_B_large_gpu.py

echo "Job finished: $(date)"
