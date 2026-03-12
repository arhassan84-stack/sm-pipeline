#!/bin/bash
#SBATCH --job-name=wn_multidt_A
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_A_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_A_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Option A: 3-channel multi-dt (dt010+dt050+dt100 downsampled to 4096)"

python pt_wavenet_multidt_A_gpu.py

echo "Job finished: $(date)"
