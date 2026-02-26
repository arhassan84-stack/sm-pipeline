#!/bin/bash
# Retrain ensemble with 302-feature caches (283 + 19 log-ACF ratio features)
# and weighted MSE loss (w=2 for d>=1) for MLP + ResNet.
# CNN retrained on 302 features with standard loss.
#
# Wave 1: mlp_weighted + resnet_weighted  (parallel)
# Wave 2: cnn_90pct  (302 features, standard loss)
# Wave 3: ensemble

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

log "=== Uploading 302-feature caches + weighted scripts to cluster ==="
rsync -az --progress \
    "$LOCAL/cache_X_train_90pct.npy" \
    "$LOCAL/cache_X_test_90pct.npy" \
    "$LOCAL/pt_mlp_weighted_gpu.py" \
    "$LOCAL/pt_resnet_weighted_gpu.py" \
    "$LOCAL/job_pt_mlp_weighted.sh" \
    "$LOCAL/job_pt_resnet_weighted.sh" \
    "$REMOTE:$WORKDIR/"
log "  Upload done."

log "=== Wave 1: mlp_weighted + resnet_weighted (parallel) ==="
JID1=$(submit job_pt_mlp_weighted.sh);    log "  Submitted pt_mlp_weighted: job $JID1"
JID2=$(submit job_pt_resnet_weighted.sh); log "  Submitted pt_resnet_weighted: job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Wave 2: cnn_90pct (302 features) ==="
JID3=$(submit job_pt_cnn_90pct.sh); log "  Submitted pt_cnn_90pct: job $JID3"
wait_jobs "$JID3"

log "=== Wave 3: ensemble ==="
JID4=$(submit job_pt_ensemble.sh); log "  Submitted pt_ensemble: job $JID4"
wait_jobs "$JID4"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
