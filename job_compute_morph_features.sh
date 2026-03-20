#!/bin/bash
#SBATCH --job-name=morph_feats
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/morph_feats_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/morph_feats_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
python compute_morph_features.py
echo "Job finished: $(date)"
