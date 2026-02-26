#!/bin/bash
# Orchestration: specialist + routing experiment
# Wave 1: slow specialist + fast specialist (parallel, save weights)
# Wave 2: router_all (trains router, evaluates options A+D, B, C)

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
    "$LOCAL/pt_specialist_slow_gpu.py" \
    "$LOCAL/pt_specialist_fast_gpu.py" \
    "$LOCAL/pt_router_all_gpu.py" \
    "$LOCAL/job_pt_specialist_slow.sh" \
    "$LOCAL/job_pt_specialist_fast.sh" \
    "$LOCAL/job_pt_router_all.sh" \
    "$REMOTE:$WORKDIR/"
log "  Scripts transferred."

log "=== Wave 1: specialist_slow + specialist_fast ==="
JID1=$(submit job_pt_specialist_slow.sh); log "  Submitted pt_specialist_slow: job $JID1"
JID2=$(submit job_pt_specialist_fast.sh); log "  Submitted pt_specialist_fast: job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Wave 2: router_all (trains router + evaluates A+D, B, C) ==="
JID3=$(submit job_pt_router_all.sh); log "  Submitted pt_router_all: job $JID3"
wait_jobs "$JID3"

log "=== Syncing results back ==="
rsync -az --include="results_*.txt" --include="*.png" \
    --exclude="cache_*" --exclude="model_*" --exclude="scaler_*" --exclude="*" \
    "$REMOTE:$WORKDIR/" "$LOCAL/"
log "=== Done! ==="
