#!/bin/bash
#SBATCH --job-name=noise_w1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/noise_w1024_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/noise_w1024_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Window: 1024 ms"

python pt_wavenet_noise_window_gpu.py --window_ms 1024

echo "Job finished: $(date)"
