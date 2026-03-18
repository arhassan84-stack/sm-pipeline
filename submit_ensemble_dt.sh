#!/bin/bash
# Submit ensemble training (seeds 1 and 2) for all single-dt WaveNet models.
# Each dt gets 2 new training runs → 14 GPU jobs total.

WORKDIR=/gpfs/gibbs/pi/holley/hassan/new_
LOGDIR=$WORKDIR/logs
COMMON="--partition=gpu --gpus=1 --cpus-per-task=4 --mem=64G"

submit_dt1() {
    local SEED=$1
    local LABEL="wavenet_wide_aug_s${SEED}"
    sbatch $COMMON --time=8:00:00 \
        --job-name="wn_aug_s${SEED}" \
        --output=$LOGDIR/wn_aug_s${SEED}_%j.out \
        --error=$LOGDIR/wn_aug_s${SEED}_%j.err \
        --wrap="
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source $WORKDIR/venv/bin/activate
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
cd $WORKDIR
echo 'Job started: \$(date)'; echo 'Node: \$(hostname)'
python pt_wavenet_wide_aug_param_gpu.py --seed $SEED
echo 'Job finished: \$(date)'"
    echo "Submitted dt=1ms seed=$SEED  label=$LABEL"
}

submit_dtX() {
    local DT_TAG=$1
    local DT=$2
    local SEED=$3
    local TIME=$4
    local LABEL="pt_wavenet_wide_aug_${DT_TAG}_s${SEED}"
    sbatch $COMMON --time=$TIME \
        --job-name="wn_${DT_TAG}_s${SEED}" \
        --output=$LOGDIR/wn_${DT_TAG}_s${SEED}_%j.out \
        --error=$LOGDIR/wn_${DT_TAG}_s${SEED}_%j.err \
        --wrap="
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
source $WORKDIR/venv/bin/activate
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
cd $WORKDIR
echo 'Job started: \$(date)'; echo 'Node: \$(hostname)'
python pt_wavenet_wide_aug_dtX_param_gpu.py --dt_tag $DT_TAG --dt $DT --seed $SEED
echo 'Job finished: \$(date)'"
    echo "Submitted $DT_TAG seed=$SEED  label=$LABEL"
}

echo "=== Submitting ensemble jobs (seed=1 and seed=2 for all dt) ==="

# dt=1ms (uses aug cache, pre-augmented, ~3h each)
submit_dt1 1
submit_dt1 2

# dt=0.2ms  (N_BINS=20480, ~24h based on data volume)
submit_dtX dt020 0.2 1 1-00:00:00
submit_dtX dt020 0.2 2 1-00:00:00

# dt=0.25ms (N_BINS=16384)
submit_dtX dt025 0.25 1 20:00:00
submit_dtX dt025 0.25 2 20:00:00

# dt=0.4ms  (N_BINS=10240, ~8h)
submit_dtX dt040 0.4 1 12:00:00
submit_dtX dt040 0.4 2 12:00:00

# dt=0.5ms  (N_BINS=8192)
submit_dtX dt050 0.5 1 10:00:00
submit_dtX dt050 0.5 2 10:00:00

# dt=0.6ms  (N_BINS=6827, ~1.5h)
submit_dtX dt060 0.6 1 6:00:00
submit_dtX dt060 0.6 2 6:00:00

# dt=0.8ms  (N_BINS=5120, ~2h)
submit_dtX dt080 0.8 1 6:00:00
submit_dtX dt080 0.8 2 6:00:00

echo "=== All 14 ensemble jobs submitted ==="
