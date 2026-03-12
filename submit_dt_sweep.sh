#!/bin/bash
# Submit sim → cache → train chains for dt=0.2, 0.4, 0.6, 0.8 in parallel.
# Each chain is independent; all 4 simulations start immediately.
#
# Usage: bash submit_dt_sweep.sh

SB=/opt/slurm/current/bin/sbatch
BASE=/gpfs/gibbs/pi/holley/hassan/new_
LOGS=$BASE/logs
VENV="module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 && source $BASE/venv/bin/activate && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && cd $BASE"

echo "Submitting dt sweep: 0.2, 0.4, 0.6, 0.8 ms"
echo "============================================"

# Format: "dt_ms:dt_tag:sim_mem:sim_time:cache_mem:train_time"
for ENTRY in \
    "0.2:dt020:64G:8:00:00:64G:1-00:00:00" \
    "0.4:dt040:64G:6:00:00:64G:12:00:00"   \
    "0.6:dt060:64G:6:00:00:64G:10:00:00"   \
    "0.8:dt080:64G:6:00:00:64G:8:00:00";   \
do
    DT=$(echo $ENTRY | cut -d: -f1)
    TAG=$(echo $ENTRY | cut -d: -f2)
    SIM_MEM=$(echo $ENTRY | cut -d: -f3)
    SIM_TIME=$(echo $ENTRY | cut -d: -f4-6 | tr ':' ':' | cut -d: -f1-3)
    CACHE_MEM=$(echo $ENTRY | cut -d: -f7)
    TRAIN_TIME=$(echo $ENTRY | cut -d: -f8-10 | tr ':' ':')

    # ── Simulation ──────────────────────────────────────────────────────────
    JID_SIM=$($SB \
        --job-name=sim_${TAG} \
        --partition=day \
        --cpus-per-task=32 \
        --mem=${SIM_MEM} \
        --time=${SIM_TIME} \
        --output=${LOGS}/sim_${TAG}_%j.out \
        --error=${LOGS}/sim_${TAG}_%j.err \
        --wrap="$VENV && python generate_simulations_dt.py --dt $DT --replicas 300 --workers 32 --seed 42" \
        | awk '{print $4}')
    echo "SIM   ${TAG}: job ${JID_SIM}"

    # ── Cache build ──────────────────────────────────────────────────────────
    JID_CACHE=$($SB \
        --dependency=afterok:${JID_SIM} \
        --job-name=cache_${TAG} \
        --partition=day \
        --cpus-per-task=8 \
        --mem=${CACHE_MEM} \
        --time=8:00:00 \
        --output=${LOGS}/cache_${TAG}_%j.out \
        --error=${LOGS}/cache_${TAG}_%j.err \
        --wrap="$VENV && python build_cache_dt.py --dt_tag $TAG --i_file sims_${TAG}_noise0_i.npy --d_file sims_${TAG}_noise0_d.npy --workers 8 --chunk-size 500" \
        | awk '{print $4}')
    echo "CACHE ${TAG}: job ${JID_CACHE}  (after ${JID_SIM})"

    # ── Training ─────────────────────────────────────────────────────────────
    JID_TRAIN=$($SB \
        --dependency=afterok:${JID_CACHE} \
        --job-name=wn_${TAG} \
        --partition=gpu \
        --gpus=1 \
        --cpus-per-task=4 \
        --mem=128G \
        --time=${TRAIN_TIME} \
        --output=${LOGS}/wn_${TAG}_%j.out \
        --error=${LOGS}/wn_${TAG}_%j.err \
        --wrap="$VENV && python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag $TAG --dt $DT" \
        | awk '{print $4}')
    echo "TRAIN ${TAG}: job ${JID_TRAIN}  (after ${JID_CACHE})"
    echo ""
done

echo "Done. All chains submitted."
