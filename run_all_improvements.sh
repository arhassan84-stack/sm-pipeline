#!/bin/bash
# ============================================================================
# Orchestration script: submit all improvement experiments to Grace cluster.
# Runs locally. Manages 4 waves respecting gpu_devel 2-job limit.
#
# Wave 1: pt_mlp_cosine    + pt_resnet_cosine  (90k features, save weights)
# Wave 2: pt_cnn_multiscale+ pt_transformer    (50k traces, existing caches)
# Wave 3: pt_cnn_90pct     + pt_weighted_loss  (needs i_90pct cache)
# Wave 4: pt_mlp_extfeat   + pt_ensemble       (needs extfeat + wave1/3 weights)
# ============================================================================

REMOTE="ah2286@grace.ycrc.yale.edu"
WORKDIR="/gpfs/gibbs/pi/holley/hassan/new_"
LOCAL="/Users/abdel-rahmanhassan/Desktop/imaging/FCS/new_"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── submit one job, echo back its SLURM job ID ──────────────────────────────
submit() {
    ssh "$REMOTE" "cd $WORKDIR && sbatch $1 2>&1" | awk '{print $NF}'
}

# ── wait until all listed SLURM job IDs leave the queue ─────────────────────
wait_jobs() {
    local jids=("$@")
    log "  Waiting for jobs: ${jids[*]}"
    while true; do
        local any=0
        for jid in "${jids[@]}"; do
            if ssh "$REMOTE" "squeue -j $jid -h 2>/dev/null" | grep -q .; then
                any=1; break
            fi
        done
        [ $any -eq 0 ] && break
        sleep 60
    done
    log "  All jobs done: ${jids[*]}"
}

# ── wait for a local file to appear ─────────────────────────────────────────
wait_file() {
    local f="$1"
    while [ ! -f "$f" ]; do
        log "  Waiting for $f ..."
        sleep 30
    done
}

# ============================================================================
# Step 1: Transfer all new Python + SLURM scripts
# ============================================================================
log "=== Step 1: Transferring scripts to cluster ==="
rsync -az \
    "$LOCAL/pt_mlp_cosine_gpu.py" \
    "$LOCAL/pt_resnet_cosine_gpu.py" \
    "$LOCAL/pt_cnn_90pct_gpu.py" \
    "$LOCAL/pt_cnn_multiscale_gpu.py" \
    "$LOCAL/pt_transformer_gpu.py" \
    "$LOCAL/pt_weighted_loss_gpu.py" \
    "$LOCAL/pt_mlp_extfeat_gpu.py" \
    "$LOCAL/pt_ensemble_gpu.py" \
    "$LOCAL/job_pt_mlp_cosine.sh" \
    "$LOCAL/job_pt_resnet_cosine.sh" \
    "$LOCAL/job_pt_cnn_90pct.sh" \
    "$LOCAL/job_pt_cnn_multiscale.sh" \
    "$LOCAL/job_pt_transformer.sh" \
    "$LOCAL/job_pt_weighted_loss.sh" \
    "$LOCAL/job_pt_mlp_extfeat.sh" \
    "$LOCAL/job_pt_ensemble.sh" \
    "$REMOTE:$WORKDIR/"
log "  Scripts transferred."

# ============================================================================
# Step 2: Transfer extfeat cache (wait if still building)
# ============================================================================
log "=== Step 2: Waiting for extfeat cache build ==="
wait_file "$LOCAL/cache_X_train_extfeat.npy"
wait_file "$LOCAL/cache_X_test_extfeat.npy"
log "  Transferring extfeat caches..."
rsync -az \
    "$LOCAL/cache_X_train_extfeat.npy" \
    "$LOCAL/cache_X_test_extfeat.npy" \
    "$REMOTE:$WORKDIR/"
log "  Extfeat caches transferred."

# ============================================================================
# Step 3: Wave 1 — mlp_cosine + resnet_cosine
# ============================================================================
log "=== Wave 1: pt_mlp_cosine + pt_resnet_cosine ==="
JID1=$(submit job_pt_mlp_cosine.sh)
log "  Submitted pt_mlp_cosine:    job $JID1"
JID2=$(submit job_pt_resnet_cosine.sh)
log "  Submitted pt_resnet_cosine: job $JID2"
wait_jobs "$JID1" "$JID2"

# ============================================================================
# Step 4: Wave 2 — cnn_multiscale + transformer (50k, no new caches needed)
# ============================================================================
log "=== Wave 2: pt_cnn_multiscale + pt_transformer ==="
JID3=$(submit job_pt_cnn_multiscale.sh)
log "  Submitted pt_cnn_multiscale: job $JID3"
JID4=$(submit job_pt_transformer.sh)
log "  Submitted pt_transformer:    job $JID4"

# While wave 2 runs, wait for the i_90pct cache and transfer it
log "  (In parallel: waiting for cache_i_train_90pct.npy to finish building...)"
wait_file "$LOCAL/cache_i_train_90pct.npy"
wait_file "$LOCAL/cache_i_test_90pct.npy"
log "  Transferring i_90pct caches..."
rsync -az \
    "$LOCAL/cache_i_train_90pct.npy" \
    "$LOCAL/cache_i_test_90pct.npy" \
    "$REMOTE:$WORKDIR/"
log "  i_90pct caches transferred."

wait_jobs "$JID3" "$JID4"

# ============================================================================
# Step 5: Wave 3 — cnn_90pct + weighted_loss
# ============================================================================
log "=== Wave 3: pt_cnn_90pct + pt_weighted_loss ==="
JID5=$(submit job_pt_cnn_90pct.sh)
log "  Submitted pt_cnn_90pct:     job $JID5"
JID6=$(submit job_pt_weighted_loss.sh)
log "  Submitted pt_weighted_loss: job $JID6"
wait_jobs "$JID5" "$JID6"

# ============================================================================
# Step 6: Wave 4 — mlp_extfeat + ensemble (needs extfeat cache + prior weights)
# ============================================================================
log "=== Wave 4: pt_mlp_extfeat + pt_ensemble ==="
JID7=$(submit job_pt_mlp_extfeat.sh)
log "  Submitted pt_mlp_extfeat: job $JID7"
JID8=$(submit job_pt_ensemble.sh)
log "  Submitted pt_ensemble:    job $JID8"
wait_jobs "$JID7" "$JID8"

# ============================================================================
# Step 7: Sync all results back
# ============================================================================
log "=== All jobs done! Syncing results back to local... ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! All results in $LOCAL ==="
