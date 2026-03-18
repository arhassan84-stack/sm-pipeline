#!/bin/bash
# Submit larger architecture (channels=512) training for all single-dt WaveNet models.
# 7 GPU jobs total, one per dt value.

WORKDIR=/gpfs/gibbs/pi/holley/hassan/new_
LOGDIR=$WORKDIR/logs
COMMON="--partition=gpu --gpus=1 --cpus-per-task=4 --mem=64G"

submit_dt1_large() {
    local LABEL="wavenet_wide_aug_c512"
    sbatch $COMMON --time=12:00:00 \
        --job-name="wn_aug_c512" \
        --output=$LOGDIR/wn_aug_c512_%j.out \
        --error=$LOGDIR/wn_aug_c512_%j.err \
        --wrap="
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source $WORKDIR/venv/bin/activate
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
cd $WORKDIR
echo 'Job started: \$(date)'; echo 'Node: \$(hostname)'
python pt_wavenet_wide_aug_param_gpu.py --channels 512
echo 'Job finished: \$(date)'"
    echo "Submitted dt=1ms channels=512  label=$LABEL"
}

submit_dtX_large() {
    local DT_TAG=$1
    local DT=$2
    local TIME=$3
    local LABEL="pt_wavenet_wide_aug_${DT_TAG}_c512"
    sbatch $COMMON --time=$TIME \
        --job-name="wn_${DT_TAG}_c512" \
        --output=$LOGDIR/wn_${DT_TAG}_c512_%j.out \
        --error=$LOGDIR/wn_${DT_TAG}_c512_%j.err \
        --wrap="
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source $WORKDIR/venv/bin/activate
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
cd $WORKDIR
echo 'Job started: \$(date)'; echo 'Node: \$(hostname)'
python pt_wavenet_wide_aug_dtX_param_gpu.py --dt_tag $DT_TAG --dt $DT --channels 512
echo 'Job finished: \$(date)'"
    echo "Submitted $DT_TAG channels=512  label=$LABEL"
}

echo "=== Submitting large-arch (channels=512) jobs for all dt ==="

# dt=1ms  (~3h × 4 param multiplier → 12h)
submit_dt1_large

# dt=0.2ms  (N_BINS=20480, large model, 36h)
submit_dtX_large dt020 0.2 1-12:00:00

# dt=0.25ms (N_BINS=16384, 24h)
submit_dtX_large dt025 0.25 1-00:00:00

# dt=0.4ms  (~7h × 4 → 24h)
submit_dtX_large dt040 0.4 1-00:00:00

# dt=0.5ms  (~similar to dt040)
submit_dtX_large dt050 0.5 20:00:00

# dt=0.6ms  (~1.5h × 4 → 8h)
submit_dtX_large dt060 0.6 8:00:00

# dt=0.8ms  (~2h × 4 → 10h)
submit_dtX_large dt080 0.8 10:00:00

echo "=== All 7 large-arch jobs submitted ==="
