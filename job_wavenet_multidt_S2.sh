#!/bin/bash
#SBATCH --job-name=wn_multidt_S2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/wn_multidt_S2_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/wn_multidt_S2_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "S2/mCherry2 WaveNet: 3 independent branches (dt010 AvgPool16, dt050, dt100)"

python pt_wavenet_multidt_S2_gpu.py

echo "Job finished: $(date)"
