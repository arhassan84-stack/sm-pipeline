#!/bin/bash
#SBATCH --job-name=fcs_inference
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

set -euo pipefail

WORKDIR=/nfs/roberts/project/pi_sah46/ah2286/new_
DATA_DIR=${WORKDIR}/01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF
OUT_DIR=${WORKDIR}/results_01082026

mkdir -p ${WORKDIR}/logs ${OUT_DIR}

cd ${WORKDIR}

# Load PyTorch module (Bouchet uses Lmod, not conda)
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

echo "=== FCS Inference ==="
echo "Data dir: ${DATA_DIR}"
echo "Output:   ${OUT_DIR}"
echo "Started:  $(date)"

python run_inference.py \
    --data_dir  "${DATA_DIR}" \
    --model_dir "${WORKDIR}" \
    --out_dir   "${OUT_DIR}"

echo "Finished: $(date)"
