#!/bin/bash
# Full improvement pipeline (Round 11):
#
# Wave 1 (parallel): cnn_aug + wavenet
# Wave 2 (parallel): cnn_wloss + ftt_wloss
# Wave 3:            save_predictions_v3  (all 13 models)
# Local pre-step:    compute_acf_nlfit.py (run before meta-learner, no GPU needed)
# Local final:       meta_learner_5fold.py
#
# Total cluster time: ~3-4 hours (2 hours training + 45 min saving)
# Total local time:   ~5 min (ACF fit) + ~2 min (meta-learner)

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
sync_result() {
    local f="$1"
    rsync -az "$REMOTE:$WORKDIR/$f" "$LOCAL/" 2>/dev/null && cat "$LOCAL/$f" || log "  (not found: $f)"
}

log "=== Uploading scripts ==="
rsync -az \
    "$LOCAL/pt_cnn_aug_gpu.py" \
    "$LOCAL/pt_cnn_wloss_gpu.py" \
    "$LOCAL/pt_ftt_wloss_gpu.py" \
    "$LOCAL/pt_wavenet_gpu.py" \
    "$LOCAL/save_predictions_v3_gpu.py" \
    "$LOCAL/job_cnn_aug.sh" \
    "$LOCAL/job_cnn_wloss.sh" \
    "$LOCAL/job_ftt_wloss.sh" \
    "$LOCAL/job_wavenet.sh" \
    "$LOCAL/job_save_preds_v3.sh" \
    "$REMOTE:$WORKDIR/"

log "=== Wave 1: cnn_aug + wavenet (parallel) ==="
JID1=$(submit job_cnn_aug.sh);   log "  Submitted: cnn_aug  job $JID1"
JID2=$(submit job_wavenet.sh);   log "  Submitted: wavenet  job $JID2"
wait_jobs "$JID1" "$JID2"
sync_result "results_pt_cnn_aug.txt"
sync_result "results_pt_wavenet.txt"

log "=== Wave 2: cnn_wloss + ftt_wloss (parallel) ==="
JID3=$(submit job_cnn_wloss.sh); log "  Submitted: cnn_wloss  job $JID3"
JID4=$(submit job_ftt_wloss.sh); log "  Submitted: ftt_wloss  job $JID4"
wait_jobs "$JID3" "$JID4"
sync_result "results_pt_cnn_wloss.txt"
sync_result "results_pt_ftt_wloss.txt"

log "=== Wave 3: save predictions (all 13 models) ==="
JID5=$(submit job_save_preds_v3.sh); log "  Submitted: save_preds_v3  job $JID5"
wait_jobs "$JID5"

log "=== Syncing prediction files ==="
rsync -az "$REMOTE:$WORKDIR/pred_*.npy" "$LOCAL/"
log "  Predictions synced."

log "=== Running local ACF nonlinear fit ==="
python3 "$LOCAL/compute_acf_nlfit.py"
log "  ACF nlfit features computed."

log "=== Running local meta-learner (5-fold NNLS) ==="
python3 "$LOCAL/meta_learner_5fold.py"

log "=== Done! ==="
