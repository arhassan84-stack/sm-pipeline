#!/bin/bash
#SBATCH --job-name=sim_multidt_S2
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/sim_multidt_S2_%j.out
#SBATCH --error=/nfs/roberts/project/pi_sah46/ah2286/new_/logs/sim_multidt_S2_%j.err

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source /nfs/roberts/project/pi_sah46/ah2286/new_/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /nfs/roberts/project/pi_sah46/ah2286/new_

echo "Job started: $(date)"; echo "Node: $(hostname)"
echo "Multi-dt S2/mCherry2 simulation: dts=0.1,0.5,1.0 ms  replicas=300  EX_LAMBDA=561nm"

python generate_simulations_multidt_S2.py \
    --dts 0.1 0.5 1.0 \
    --replicas 300 \
    --workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --tag multidt_S2

echo "Job finished: $(date)"
