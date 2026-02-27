#!/bin/bash
# Save all model predictions (train+test), sync back, run weight optimization locally.

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
        [ $any -eq 0 ] && break; sleep 30
    done
    log "  Done: ${jids[*]}"
}

log "=== Uploading save_predictions script ==="
rsync -az "$LOCAL/save_predictions_gpu.py" "$LOCAL/job_save_predictions.sh" "$REMOTE:$WORKDIR/"

log "=== Running save_predictions on cluster ==="
JID=$(submit job_save_predictions.sh); log "  Submitted: job $JID"
wait_jobs "$JID"

log "=== Syncing prediction files back ==="
rsync -az "$REMOTE:$WORKDIR/pred_*.npy" "$LOCAL/"
log "  Predictions synced."

log "=== Running weight optimization locally ==="
python3 "$LOCAL/optimize_weights.py"
log "=== Done! ==="
