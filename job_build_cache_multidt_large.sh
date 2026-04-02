#!/bin/bash
#SBATCH --job-name=cache_multidt_large
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_multidt_large_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_multidt_large_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "GPU cache build: 900k traces x 3 dt, using features_gpu.py"

python build_cache_multidt_large_gpu.py \
    --seed 42 \
    --tag multidt_large

echo "Job finished: $(date)"
