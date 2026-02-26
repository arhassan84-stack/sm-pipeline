#!/bin/bash
# Orchestration: balanced training experiment
# Wave 1: balanced_mlp + balanced_resnet (parallel, both within 2-job limit)
# Wave 2: balanced_ensemble (needs both models from wave 1)

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

log "=== Transferring scripts ==="
rsync -az \
    "$LOCAL/pt_balanced_mlp_gpu.py" \
    "$LOCAL/pt_balanced_resnet_gpu.py" \
    "$LOCAL/pt_balanced_ensemble_gpu.py" \
    "$LOCAL/job_pt_balanced_mlp.sh" \
    "$LOCAL/job_pt_balanced_resnet.sh" \
    "$LOCAL/job_pt_balanced_ensemble.sh" \
    "$REMOTE:$WORKDIR/"
log "  Scripts transferred."

log "=== Wave 1: balanced_mlp + balanced_resnet (parallel) ==="
JID1=$(submit job_pt_balanced_mlp.sh);    log "  Submitted pt_balanced_mlp: job $JID1"
JID2=$(submit job_pt_balanced_resnet.sh); log "  Submitted pt_balanced_resnet: job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Wave 2: balanced_ensemble ==="
JID3=$(submit job_pt_balanced_ensemble.sh); log "  Submitted pt_balanced_ensemble: job $JID3"
wait_jobs "$JID3"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
