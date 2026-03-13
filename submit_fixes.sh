#!/bin/bash
# Resubmit all failed jobs with fixes applied.

SB=/opt/slurm/current/bin/sbatch
BASE=/gpfs/gibbs/pi/holley/hassan/new_
LOGS=$BASE/logs
VENV="module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 && source $BASE/venv/bin/activate && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && cd $BASE"

echo "Submitting fixed jobs"
echo "====================="

# ── cache_dt060 (fixed: odd-N halving bug in build_cache_dt.py) ───────────────
JID_C060=$($SB \
    --job-name=cache_dt060 \
    --partition=day \
    --cpus-per-task=8 \
    --mem=64G \
    --time=8:00:00 \
    --output=${LOGS}/cache_dt060_%j.out \
    --error=${LOGS}/cache_dt060_%j.err \
    --wrap="$VENV && python build_cache_dt.py --dt_tag dt060 --i_file sims_dt060_noise0_i.npy --d_file sims_dt060_noise0_d.npy --workers 8 --chunk-size 500" \
    | awk '{print $4}')
echo "cache_dt060:   job ${JID_C060}"

# ── wn_dt060 (after cache; fixed: chunked eval in dtX script) ────────────────
JID_W060=$($SB \
    --dependency=afterok:${JID_C060} \
    --job-name=wn_dt060 \
    --partition=gpu \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=128G \
    --time=10:00:00 \
    --output=${LOGS}/wn_dt060_%j.out \
    --error=${LOGS}/wn_dt060_%j.err \
    --wrap="$VENV && python pt_wavenet_wide_aug_dtX_gpu.py --dt_tag dt060 --dt 0.6" \
    | awk '{print $4}')
echo "wn_dt060:      job ${JID_W060}  (after ${JID_C060})"

# ── cache_multidt (fixed: 256G memory, odd-N bug) ─────────────────────────────
JID_CMDT=$($SB $BASE/job_build_cache_multidt.sh | awk '{print $4}')
echo "cache_multidt: job ${JID_CMDT}"

# ── wn_multidt A/B/C/D (after cache_multidt) ──────────────────────────────────
for OPT in A B C D; do
    MEM=128G; TIME=1-00:00:00
    if [ "$OPT" != "A" ]; then MEM=256G; TIME=2-00:00:00; fi
    JID=$($SB \
        --dependency=afterok:${JID_CMDT} \
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
    echo "wn_multidt_${OPT}:  job ${JID}  (after ${JID_CMDT})"
done

# ── wn_dt010 (fixed: BATCH_SIZE=32, EVAL_BATCH=32) ────────────────────────────
JID_W010=$($SB \
    --job-name=wn_dt010 \
    --partition=gpu \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=128G \
    --time=2-00:00:00 \
    --output=${LOGS}/wn_dt010_%j.out \
    --error=${LOGS}/wn_dt010_%j.err \
    $BASE/job_wavenet_wide_aug_dt010.sh \
    | awk '{print $4}')
echo "wn_dt010:      job ${JID_W010}  (BATCH_SIZE=32)"

# ── eval_wn_dt020 (model already saved; just runs inference) ─────────────────
JID_E020=$($SB \
    --job-name=eval_dt020 \
    --partition=gpu \
    --gpus=1 \
    --cpus-per-task=2 \
    --mem=64G \
    --time=1:00:00 \
    --output=${LOGS}/eval_dt020_%j.out \
    --error=${LOGS}/eval_dt020_%j.err \
    --wrap="$VENV && python eval_wn_dt020.py" \
    | awk '{print $4}')
echo "eval_dt020:    job ${JID_E020}"

echo ""
echo "Done."
