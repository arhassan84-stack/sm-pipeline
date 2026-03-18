#!/bin/bash
#SBATCH --job-name=tta_wavenet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/tta_wavenet_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/tta_wavenet_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "TTA for wavenet_wide_aug + dt020/025/040/050/060/080"

python pt_tta_wavenet_dt.py

echo "Job finished: $(date)"
