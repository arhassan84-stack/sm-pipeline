#!/bin/bash
# Improvements 2, 3, 4:
#   #2 stride=2 WaveNet (4096→2048, 10 blocks, RF=4092 lags)
#   #3 wide WaveNet (channels=256, ~1.8M params)
#   #4 seeds s4 + s5
#
# Wave 1 (parallel): wavenet_stride2 + wavenet_wide  (~40-60 min each)
# Wave 2 (parallel): wavenet_s4 + wavenet_s5         (~20 min each)
# Local:             meta_learner_5fold.py

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
sync_model() {
    local m=$1
    rsync -az "$REMOTE:$WORKDIR/results_pt_${m}.txt" "$LOCAL/" 2>/dev/null
    cat "$LOCAL/results_pt_${m}.txt" 2>/dev/null
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_train.npy" "$LOCAL/"
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_test.npy"  "$LOCAL/"
}

log "=== Uploading scripts ==="
rsync -az \
    "$LOCAL/pt_wavenet_stride2_gpu.py" \
    "$LOCAL/pt_wavenet_wide_gpu.py" \
    "$LOCAL/pt_wavenet_s4_gpu.py" \
    "$LOCAL/pt_wavenet_s5_gpu.py" \
    "$LOCAL/job_wavenet_stride2.sh" \
    "$LOCAL/job_wavenet_wide.sh" \
    "$LOCAL/job_wavenet_s4.sh" \
    "$LOCAL/job_wavenet_s5.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Wave 1: wavenet_stride2 + wavenet_wide (parallel) ==="
JID1=$(submit job_wavenet_stride2.sh); log "  Submitted: wavenet_stride2  job $JID1"
JID2=$(submit job_wavenet_wide.sh);    log "  Submitted: wavenet_wide     job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Syncing Wave 1 ==="
sync_model wavenet_stride2
sync_model wavenet_wide
log "  Synced Wave 1 predictions."

log "=== Wave 2: wavenet_s4 + wavenet_s5 (parallel) ==="
JID3=$(submit job_wavenet_s4.sh); log "  Submitted: wavenet_s4  job $JID3"
JID4=$(submit job_wavenet_s5.sh); log "  Submitted: wavenet_s5  job $JID4"
wait_jobs "$JID3" "$JID4"

log "=== Syncing Wave 2 ==="
sync_model wavenet_s4
sync_model wavenet_s5
log "  Synced Wave 2 predictions."

log "=== Running updated meta-learner ==="
python3 "$LOCAL/meta_learner_5fold.py"

log "=== Done! ==="
