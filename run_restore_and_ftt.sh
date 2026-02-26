#!/bin/bash
# Restore best 283-feature 3-NN models, then train FT-Transformer and evaluate.
#
# Wave 1: mlp_cosine + resnet_cosine (parallel, 283 features)
# Wave 2: cnn_90pct
# Wave 3: ensemble (restore best 3-NN result)
# Wave 4: fttransformer (new architecture)
# Wave 5: ensemble_ftt (compare all combinations)

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

log "=== Uploading 283-feature caches + new scripts ==="
rsync -az --progress \
    "$LOCAL/cache_X_train_90pct.npy" \
    "$LOCAL/cache_X_test_90pct.npy" \
    "$LOCAL/pt_fttransformer_gpu.py" \
    "$LOCAL/pt_ensemble_ftt_gpu.py" \
    "$LOCAL/job_pt_fttransformer.sh" \
    "$LOCAL/job_pt_ensemble_ftt.sh" \
    "$REMOTE:$WORKDIR/"
log "  Upload done."

log "=== Wave 1: mlp_cosine + resnet_cosine (parallel, 283f) ==="
JID1=$(submit job_pt_mlp_cosine.sh);    log "  Submitted pt_mlp_cosine: job $JID1"
JID2=$(submit job_pt_resnet_cosine.sh); log "  Submitted pt_resnet_cosine: job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Wave 2: cnn_90pct (283 features) ==="
JID3=$(submit job_pt_cnn_90pct.sh); log "  Submitted pt_cnn_90pct: job $JID3"
wait_jobs "$JID3"

log "=== Wave 3: ensemble (restore 3-NN baseline) ==="
JID4=$(submit job_pt_ensemble.sh); log "  Submitted pt_ensemble: job $JID4"
wait_jobs "$JID4"

log "=== Wave 4: FT-Transformer (new architecture) ==="
JID5=$(submit job_pt_fttransformer.sh); log "  Submitted pt_fttransformer: job $JID5"
wait_jobs "$JID5"

log "=== Wave 5: ensemble_ftt (3-NN vs 4-model vs all swaps) ==="
JID6=$(submit job_pt_ensemble_ftt.sh); log "  Submitted pt_ensemble_ftt: job $JID6"
wait_jobs "$JID6"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
