#!/bin/bash
# Train FTT-large and FTT-v2 in parallel (Wave 1), then run ensemble eval (Wave 2).

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
    "$LOCAL/pt_fttransformer_large_gpu.py" \
    "$LOCAL/pt_fttransformer_v2_gpu.py" \
    "$LOCAL/pt_ensemble_multiftt_gpu.py" \
    "$LOCAL/job_ftt_large.sh" \
    "$LOCAL/job_ftt_v2.sh" \
    "$LOCAL/job_ensemble_multiftt.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Wave 1: FTT-large + FTT-v2 in parallel ==="
JID1=$(submit job_ftt_large.sh); log "  Submitted: ftt_large job $JID1"
JID2=$(submit job_ftt_v2.sh);    log "  Submitted: ftt_v2    job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Syncing individual results ==="
rsync -az "$REMOTE:$WORKDIR/results_pt_fttransformer_large.txt" "$LOCAL/" 2>/dev/null && \
    cat "$LOCAL/results_pt_fttransformer_large.txt" || log "  (large result not found)"
rsync -az "$REMOTE:$WORKDIR/results_pt_fttransformer_v2.txt" "$LOCAL/" 2>/dev/null && \
    cat "$LOCAL/results_pt_fttransformer_v2.txt" || log "  (v2 result not found)"

log "=== Wave 2: Ensemble eval ==="
JID3=$(submit job_ensemble_multiftt.sh); log "  Submitted: ensemble job $JID3"
wait_jobs "$JID3"

log "=== Syncing results ==="
rsync -az "$REMOTE:$WORKDIR/results_ensemble_multiftt.txt" "$LOCAL/"
cat "$LOCAL/results_ensemble_multiftt.txt"

log "=== Done! ==="
