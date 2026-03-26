#!/bin/bash
#SBATCH --job-name=wn_otf_512ms
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=r818u09n09
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_otf_512ms_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/wn_otf_512ms_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Fixed window: 512ms  (N010=5120, N050=1024, N100=512)"

python pt_wavenet_multidt_otf_fixedlen_gpu.py --duration-ms 512

echo "Job finished: $(date)"
