#!/bin/bash
#SBATCH --job-name=cache_multidt_S2
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/cache_multidt_S2_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/cache_multidt_S2_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Building multi-dt S2/mCherry2 caches (dt010, dt050, dt100) with shared train/test split"

python build_cache_multidt_S2.py \
    --workers 8 \
    --chunk-size 500 \
    --seed 42 \
    --tag multidt_S2

echo "Job finished: $(date)"
