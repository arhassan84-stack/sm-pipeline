#!/bin/bash
#SBATCH --job-name=wn_dt025
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_dt025_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_dt025_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "dt025 training: 16384 bins/trace, WaveNet wide + online aug"

python pt_wavenet_wide_aug_dt025_gpu.py

echo "Job finished: $(date)"
