#!/bin/bash
# SLURM array job: compute 302 features + 90/10 train/test split for dt simulations.
# Array index 0 → dt025 (dt=0.25ms, 16384 bins/trace)
# Array index 1 → dt050 (dt=0.50ms,  8192 bins/trace)
# Runs AFTER job_generate_sims_dt.sh completes.
#
#SBATCH --job-name=cache_dt
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-1
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt_%A_%a.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/cache_dt_%A_%a.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

# Map array index → dt tag and simulation files
DT_TAGS=(dt025 dt050)
DT_TAG=${DT_TAGS[$SLURM_ARRAY_TASK_ID]}
I_FILE="sims_${DT_TAG}_noise0_i.npy"
D_FILE="sims_${DT_TAG}_noise0_d.npy"

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID  dt_tag=${DT_TAG}  CPUs: $SLURM_CPUS_PER_TASK"
echo "Input: ${I_FILE}  ${D_FILE}"

python build_cache_dt.py \
    --dt_tag $DT_TAG \
    --i_file $I_FILE \
    --d_file $D_FILE \
    --workers $SLURM_CPUS_PER_TASK \
    --chunk-size 5000

echo "Job finished: $(date)"
