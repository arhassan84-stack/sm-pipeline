#!/bin/bash
# SLURM array job: build smoothed caches for both dt datasets.
# 2 dt values × 6 windows = 12 tasks (indices 0-11).
#
# Index mapping:
#   dt_idx  = SLURM_ARRAY_TASK_ID / 6   (0=dt025, 1=dt050)
#   win_idx = SLURM_ARRAY_TASK_ID % 6   (0..5 → W=2,4,6,12,20,40)
#
# Runs AFTER job_build_cache_dt.sh completes.
#
#SBATCH --job-name=smooth_dt
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-11
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/smooth_dt_%A_%a.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/smooth_dt_%A_%a.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

DT_TAGS=(dt025 dt050)
WINDOWS=(2 4 6 12 20 40)

dt_idx=$(( SLURM_ARRAY_TASK_ID / 6 ))
win_idx=$(( SLURM_ARRAY_TASK_ID % 6 ))

DT_TAG=${DT_TAGS[$dt_idx]}
W=${WINDOWS[$win_idx]}

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID  dt_tag=${DT_TAG}  W=${W}  CPUs: $SLURM_CPUS_PER_TASK"

python build_smoothed_cache_dt.py \
    --dt_tag $DT_TAG \
    --windows $W \
    --workers $SLURM_CPUS_PER_TASK \
    --chunk-size 5000

echo "Job finished: $(date)"
