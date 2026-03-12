#!/bin/bash
#SBATCH --job-name=wn_multidt_C
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_C_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_multidt_C_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Option C: shared-weight WaveNet backbone (3 passes, native resolution per dt)"

python pt_wavenet_multidt_C_gpu.py

echo "Job finished: $(date)"
