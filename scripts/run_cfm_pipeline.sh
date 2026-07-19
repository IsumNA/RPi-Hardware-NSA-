#!/usr/bin/env bash
# FAST CFM path on AI GPU: Teacher → Consistency distill → ONNX (Pi student).
# Aggressive shorter schedule for quicker results; keep consistency distill.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/cfm_teacher_panels outputs/cfm_student_panels

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/cfm_pipeline.log; }

log "FAST CFM pipeline start (Teacher 6k → Consistency distill 3k → ONNX)."

# Skip long wait if nothing is holding the GPU for Old-Way regression.
if pgrep -f 'python -u train_stream_to_gt.py' >/dev/null 2>&1; then
  log "Waiting for train_stream_to_gt.py to finish (free GPU)..."
  while pgrep -f 'python -u train_stream_to_gt.py' >/dev/null 2>&1; do
    sleep 30
  done
  log "GPU free."
fi

log "Starting CFM Teacher (rectified flow noisy→GT: 6000 steps, sample×8, 128ch×8)."

.venv/bin/python -u train_cfm_teacher.py \
  --gains 128,256,512 \
  --steps 6000 \
  --channels 128 \
  --depth 8 \
  --stride 2 \
  --temporal 4 \
  --batch 4 \
  --crop 192 \
  --lr 6e-4 \
  --sample-steps 8 \
  --panel-every 500 \
  2>&1 | tee -a logs/cfm_teacher_train.log

log "Teacher done. Consistency Flow Matching distillation (3000 steps, 64ch×6)."

.venv/bin/python -u train_cfm_distill.py \
  --teacher outputs/cfm_teacher.pt \
  --method consistency \
  --gains 128,256,512 \
  --steps 3000 \
  --channels 64 \
  --depth 6 \
  --stride 2 \
  --temporal 4 \
  --batch 2 \
  --crop 256 \
  --lr 6e-4 \
  --integrate-steps 4 \
  --cd-weight 0.5 \
  --teacher-steps 8 \
  --gt-weight 0.15 \
  --panel-every 500 \
  2>&1 | tee -a logs/cfm_distill_train.log

log "CFM FAST pipeline complete. Artifacts: outputs/cfm_teacher.pt outputs/cfm_student.pt outputs/cfm_student.onnx"
