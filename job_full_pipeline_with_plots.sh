#!/bin/bash
#SBATCH --job-name=pipeline_plots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/pipeline_plots_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/pipeline_plots_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"

python "${PIPELINE_SCRIPT:?PIPELINE_SCRIPT env var not set}" "${DATA_FOLDER:?DATA_FOLDER env var not set}"

echo "Job finished: $(date)"
