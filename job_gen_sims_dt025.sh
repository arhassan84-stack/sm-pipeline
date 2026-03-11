#!/bin/bash
# Resubmission of dt025 simulation (dt=0.25ms, 16384 bins/trace).
# Original job 55532119_0 failed: OOM (31.7/32G) caused swap thrashing → TIMEOUT.
# Fix: 64G RAM, 4h wall time.
#
#SBATCH --job-name=gen_sims_dt025
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/gen_sims_dt025_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/gen_sims_dt025_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "dt=0.25ms  CPUs: $SLURM_CPUS_PER_TASK  MEM: 64G"

python generate_simulations_dt.py \
    --dt 0.25 \
    --replicas 300 \
    --workers $SLURM_CPUS_PER_TASK \
    --seed 42

echo "Job finished: $(date)"
