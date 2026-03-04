#!/bin/bash
# Full augmented training pipeline:
#   Step 1: Generate 300k new simulation traces  (~30 min, CPU)
#   Step 2: Build augmented feature cache        (~4-6 h, CPU)
#   Step 3: Retrain best models on aug data      (parallel GPU jobs)
#   Step 4: Sync results + run meta-learner

REMOTE="ah2286@grace.ycrc.yale.edu"
WORKDIR="/gpfs/gibbs/pi/holley/hassan/new_"
LOCAL="/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_"

log() { echo "[$(date '+%H:%M:%S')] $1"; }
submit() { ssh "$REMOTE" "cd $WORKDIR && sbatch $1 2>&1" | awk '{print $NF}'; }
wait_jobs() {
    local jids=("$@"); log "  Waiting for jobs: ${jids[*]}"
    while true; do
        local any=0
        for jid in "${jids[@]}"; do
            if ssh "$REMOTE" "squeue -j $jid -h 2>/dev/null" | grep -q .; then any=1; break; fi
        done
        [ $any -eq 0 ] && break; sleep 60
    done
    log "  Done: ${jids[*]}"
}

log "=== Uploading scripts ==="
rsync -az \
    "$LOCAL/generate_simulations.py" \
    "$LOCAL/build_augmented_cache.py" \
    "$LOCAL/job_generate_sims.sh" \
    "$LOCAL/job_build_aug_cache.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Step 1: Generating 300k simulation traces ==="
JID1=$(submit job_generate_sims.sh)
log "  Submitted: gen_sims  job $JID1"
wait_jobs "$JID1"
log "  Simulation complete. Checking output..."
ssh "$REMOTE" "ls -lh $WORKDIR/new_sims_noise0_i.npy $WORKDIR/new_sims_noise0_d.npy 2>/dev/null || echo 'ERROR: output files not found'"

log "=== Step 2: Building augmented feature cache ==="
JID2=$(submit job_build_aug_cache.sh)
log "  Submitted: aug_cache  job $JID2"
wait_jobs "$JID2"
log "  Cache build complete. Checking output..."
ssh "$REMOTE" "ls -lh $WORKDIR/cache_i_train_aug.npy $WORKDIR/cache_X_train_aug.npy $WORKDIR/cache_d_train_aug.npy 2>/dev/null || echo 'ERROR: cache files not found'"

log "=== Done: aug caches ready for retraining ==="
log "  Next: submit GPU retraining jobs (pt_*_aug_gpu.py)"
