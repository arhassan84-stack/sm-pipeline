#!/bin/bash
#SBATCH --job-name=sim_multidt4
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/gibbs/pi/holley/hassan/new_/logs/sim_multidt4_%j.out
#SBATCH --error=/gpfs/gibbs/pi/holley/hassan/new_/logs/sim_multidt4_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /gpfs/gibbs/pi/holley/hassan/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /gpfs/gibbs/pi/holley/hassan/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Multi-dt4 simulation: dts=0.05,0.1,0.5,1.0 ms  replicas=300"

python generate_simulations_multidt.py \
    --dts 0.05 0.1 0.5 1.0 \
    --replicas 300 \
    --workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --tag multidt4

echo "Job finished: $(date)"
