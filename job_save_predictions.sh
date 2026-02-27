#!/bin/bash
#SBATCH --job-name=save_preds
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/save_preds_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/save_preds_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python save_predictions_gpu.py

echo "Job finished: $(date)"
