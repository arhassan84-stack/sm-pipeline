#!/bin/bash
#SBATCH --job-name=wn_B_large_v2
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/wn_B_large_v2_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/wn_B_large_v2_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "B_large_v2: channels=512, attn_pool, dilations=[1..512], AMP, large cache"

python pt_wavenet_multidt_B_large_v2_gpu.py

echo "Job finished: $(date)"
