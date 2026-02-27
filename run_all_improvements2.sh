#!/bin/bash
# Full improvement pipeline:
# Wave 1: cnn_s123 + cnn_multiscale_90k (parallel)
# Wave 2: cnn_s777
# Wave 3: save_all_predictions
# Local:  meta_learner.py (after sync)

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
sync_result() {
    local f="$1"
    rsync -az "$REMOTE:$WORKDIR/$f" "$LOCAL/" 2>/dev/null && cat "$LOCAL/$f" || log "  (not found: $f)"
}

log "=== Uploading scripts ==="
rsync -az \
    "$LOCAL/pt_cnn_s123_gpu.py" \
    "$LOCAL/pt_cnn_s777_gpu.py" \
    "$LOCAL/pt_cnn_multiscale_90k_gpu.py" \
    "$LOCAL/save_predictions_all_gpu.py" \
    "$LOCAL/job_cnn_s123.sh" \
    "$LOCAL/job_cnn_s777.sh" \
    "$LOCAL/job_cnn_multiscale_90k.sh" \
    "$LOCAL/job_save_all_predictions.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Wave 1: cnn_s123 + cnn_multiscale_90k (parallel) ==="
JID1=$(submit job_cnn_s123.sh);          log "  Submitted: cnn_s123       job $JID1"
JID2=$(submit job_cnn_multiscale_90k.sh);log "  Submitted: cnn_multiscale job $JID2"
wait_jobs "$JID1" "$JID2"
sync_result "results_pt_cnn_s123.txt"
sync_result "results_pt_cnn_multiscale_90k.txt"

log "=== Wave 2: cnn_s777 ==="
JID3=$(submit job_cnn_s777.sh); log "  Submitted: cnn_s777 job $JID3"
wait_jobs "$JID3"
sync_result "results_pt_cnn_s777.txt"

log "=== Wave 3: save all predictions ==="
JID4=$(submit job_save_all_predictions.sh); log "  Submitted: save_all_preds job $JID4"
wait_jobs "$JID4"

log "=== Syncing prediction files ==="
rsync -az "$REMOTE:$WORKDIR/pred_*.npy" "$LOCAL/"
log "  Predictions synced."

log "=== Running meta-learner locally ==="
python3 "$LOCAL/meta_learner.py"

log "=== Done! ==="
