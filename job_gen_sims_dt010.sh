#!/bin/bash
#SBATCH --job-name=gen_sims_dt010
#SBATCH --partition=day
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/gen_sims_dt010_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/gen_sims_dt010_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "dt=0.1ms  N_BINS=40960  replicas=300  CPUs: $SLURM_CPUS_PER_TASK  MEM: 128G"

python generate_simulations_dt.py \
    --dt 0.1 \
    --replicas 300 \
    --workers $SLURM_CPUS_PER_TASK \
    --seed 42

echo "Job finished: $(date)"
