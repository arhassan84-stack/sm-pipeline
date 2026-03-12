#!/bin/bash
# Submit multi-dt cache build → 4 training jobs (A, B, C, D) in parallel.
# All 4 training jobs start simultaneously after the shared cache completes.
#
# Usage: bash submit_multidt_training.sh

BASE=/gpfs/gibbs/pi/holley/hassan/new_
LOGS=$BASE/logs

echo "Submitting multi-dt pipeline (cache → A, B, C, D in parallel)"
echo "=============================================================="

# ── Cache build ───────────────────────────────────────────────────────────────
JID_CACHE=$(sbatch \
    --job-name=cache_multidt \
    --partition=day \
    --cpus-per-task=8 \
    --mem=128G \
    --time=16:00:00 \
    --output=${LOGS}/cache_multidt_%j.out \
    --error=${LOGS}/cache_multidt_%j.err \
    $BASE/job_build_cache_multidt.sh \
    | awk '{print $4}')
echo "CACHE: job ${JID_CACHE}"

# ── Training jobs (all 4 start in parallel after cache) ──────────────────────
for OPT in A B C D; do
    TIME="1-00:00:00"
    MEM="128G"
    if [ "$OPT" = "B" ] || [ "$OPT" = "C" ] || [ "$OPT" = "D" ]; then
        TIME="2-00:00:00"
        MEM="256G"
    fi
    JID=$(sbatch \
        --dependency=afterok:${JID_CACHE} \
        --job-name=wn_multidt_${OPT} \
        --partition=gpu \
        --gpus=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME} \
        --output=${LOGS}/wn_multidt_${OPT}_%j.out \
        --error=${LOGS}/wn_multidt_${OPT}_%j.err \
        $BASE/job_wavenet_multidt_${OPT}.sh \
        | awk '{print $4}')
    echo "TRAIN ${OPT}: job ${JID}  (after cache ${JID_CACHE})"
done

echo ""
echo "Done. All jobs submitted."
