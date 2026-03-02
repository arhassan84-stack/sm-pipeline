#!/bin/bash
# Train Hybrid WaveNet+FTT, sync predictions, run updated meta-learner.
#
# The training script saves pred_logd_wavenet_ftt_{train,test}.npy directly.
# meta_learner_5fold.py auto-detects wavenet_ftt if its pred files are present.

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

log "=== Uploading scripts ==="
rsync -az \
    "$LOCAL/pt_wavenet_ftt_gpu.py" \
    "$LOCAL/job_wavenet_ftt.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Submitting WaveNet+FTT job ==="
JID=$(submit job_wavenet_ftt.sh); log "  Submitted: wn_ftt  job $JID"
wait_jobs "$JID"

log "=== Syncing results + predictions ==="
rsync -az "$REMOTE:$WORKDIR/results_pt_wavenet_ftt.txt" "$LOCAL/" 2>/dev/null
cat "$LOCAL/results_pt_wavenet_ftt.txt"
rsync -az \
    "$REMOTE:$WORKDIR/pred_logd_wavenet_ftt_train.npy" \
    "$REMOTE:$WORKDIR/pred_logd_wavenet_ftt_test.npy" \
    "$LOCAL/"
log "  Synced predictions."

log "=== Running updated meta-learner ==="
python3 "$LOCAL/meta_learner_5fold.py"

log "=== Done! ==="
