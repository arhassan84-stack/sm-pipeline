#!/bin/bash
# Retrain full ensemble with 283-feature caches (212 + 71 expanded features).
# All training scripts use X.shape[1] dynamically — no code changes needed.
#
# Wave 1: mlp_cosine + resnet_cosine   (parallel)
# Wave 2: cnn_90pct
# Wave 3: ensemble (3-NN) + ensemble_4model (HistGBR+3NN) in parallel

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

log "=== Uploading 283-feature caches + updated scripts to cluster ==="
rsync -az --progress \
    "$LOCAL/cache_X_train_90pct.npy" \
    "$LOCAL/cache_X_test_90pct.npy" \
    "$LOCAL/pt_ensemble_4model_gpu.py" \
    "$REMOTE:$WORKDIR/"
log "  Upload done."

log "=== Wave 1: mlp_cosine + resnet_cosine (parallel) ==="
JID1=$(submit job_pt_mlp_cosine.sh);    log "  Submitted pt_mlp_cosine: job $JID1"
JID2=$(submit job_pt_resnet_cosine.sh); log "  Submitted pt_resnet_cosine: job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Wave 2: cnn_90pct ==="
JID3=$(submit job_pt_cnn_90pct.sh); log "  Submitted pt_cnn_90pct: job $JID3"
wait_jobs "$JID3"

log "=== Wave 3: 3-NN ensemble + 4-model ensemble (parallel) ==="
JID4=$(submit job_pt_ensemble.sh);         log "  Submitted pt_ensemble: job $JID4"
JID5=$(submit job_pt_ensemble_4model.sh);  log "  Submitted pt_ensemble_4model: job $JID5"
wait_jobs "$JID4" "$JID5"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
