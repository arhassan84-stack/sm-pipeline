#!/bin/bash
# Build peak-centered trace cache locally, then train CNN on cropped traces
# and evaluate in ensemble with the existing 3-NN models.
#
# Wave 1: cnn_peakcrop  (peak-centered CNN, new model)
# Wave 2: ensemble_peakcrop  (compare 3-NN vs 3-NN+crop vs crop-only etc.)

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

log "=== Step 1: Build peak-crop cache locally ==="
python3 "$LOCAL/build_cache_peakcrop.py"
log "  Peakcrop caches built."

log "=== Step 2: Upload caches + scripts ==="
rsync -az --progress \
    "$LOCAL/cache_i_train_peakcrop.npy" \
    "$LOCAL/cache_i_test_peakcrop.npy" \
    "$LOCAL/pt_cnn_peakcrop_gpu.py" \
    "$LOCAL/pt_ensemble_peakcrop_gpu.py" \
    "$LOCAL/job_pt_cnn_peakcrop.sh" \
    "$LOCAL/job_pt_ensemble_peakcrop.sh" \
    "$REMOTE:$WORKDIR/"
log "  Upload done."

log "=== Wave 1: cnn_peakcrop ==="
JID1=$(submit job_pt_cnn_peakcrop.sh); log "  Submitted pt_cnn_peakcrop: job $JID1"
wait_jobs "$JID1"

log "=== Wave 2: ensemble_peakcrop ==="
JID2=$(submit job_pt_ensemble_peakcrop.sh); log "  Submitted pt_ensemble_peakcrop: job $JID2"
wait_jobs "$JID2"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
