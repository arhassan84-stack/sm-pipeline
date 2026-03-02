#!/bin/bash
# Train 4 new WaveNet variants in 2 parallel waves, sync predictions, run meta-learner.
#
# Wave 1 (parallel): wavenet_s2 + wavenet_s3          (~18 min each)
# Wave 2 (parallel): wavenet_ftt_v2 + wavenet_alt     (~60 min each)
# Local:             meta_learner_5fold.py             (~1 min)
#
# New models:
#   wavenet_s2:      WaveNet seed=2 (same arch, different init)
#   wavenet_s3:      WaveNet seed=3 (same arch, different init)
#   wavenet_ftt_v2:  Hybrid WaveNet+FTT with per-branch LR (wave=5e-4, ftt=1e-3)
#   wavenet_alt:     WaveNet dilations=[1,2,4,8,16,32,64,128,256,512], RF=8184 lags

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
    "$LOCAL/pt_wavenet_s2_gpu.py" \
    "$LOCAL/pt_wavenet_s3_gpu.py" \
    "$LOCAL/pt_wavenet_ftt_v2_gpu.py" \
    "$LOCAL/pt_wavenet_alt_gpu.py" \
    "$LOCAL/job_wavenet_s2.sh" \
    "$LOCAL/job_wavenet_s3.sh" \
    "$LOCAL/job_wavenet_ftt_v2.sh" \
    "$LOCAL/job_wavenet_alt.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Wave 1: wavenet_s2 + wavenet_s3 (parallel) ==="
JID1=$(submit job_wavenet_s2.sh);  log "  Submitted: wavenet_s2   job $JID1"
JID2=$(submit job_wavenet_s3.sh);  log "  Submitted: wavenet_s3   job $JID2"
wait_jobs "$JID1" "$JID2"

log "=== Syncing Wave 1 results ==="
for m in wavenet_s2 wavenet_s3; do
    rsync -az "$REMOTE:$WORKDIR/results_pt_${m}.txt" "$LOCAL/" 2>/dev/null
    cat "$LOCAL/results_pt_${m}.txt" 2>/dev/null
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_train.npy" "$LOCAL/"
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_test.npy"  "$LOCAL/"
done
log "  Synced Wave 1 predictions."

log "=== Wave 2: wavenet_ftt_v2 + wavenet_alt (parallel) ==="
JID3=$(submit job_wavenet_ftt_v2.sh); log "  Submitted: wavenet_ftt_v2  job $JID3"
JID4=$(submit job_wavenet_alt.sh);    log "  Submitted: wavenet_alt     job $JID4"
wait_jobs "$JID3" "$JID4"

log "=== Syncing Wave 2 results ==="
for m in wavenet_ftt_v2 wavenet_alt; do
    rsync -az "$REMOTE:$WORKDIR/results_pt_${m}.txt" "$LOCAL/" 2>/dev/null
    cat "$LOCAL/results_pt_${m}.txt" 2>/dev/null
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_train.npy" "$LOCAL/"
    rsync -az "$REMOTE:$WORKDIR/pred_logd_${m}_test.npy"  "$LOCAL/"
done
log "  Synced Wave 2 predictions."

log "=== Running updated meta-learner ==="
python3 "$LOCAL/meta_learner_5fold.py"

log "=== Done! ==="
