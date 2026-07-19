#!/usr/bin/env bash
# Overnight sharp CFM distill on cloud GPU using cloud_pack (no DNGs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
command -v "$PY" >/dev/null || PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

mkdir -p outputs/cloud_sharp outputs/cloud_panels logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

TEACHER="${TEACHER:-cloud_pack/checkpoints/cfm_edm_teacher.pt}"
INIT="${INIT:-cloud_pack/checkpoints/perfect_r1_student_best.pt}"
PACK="${PACK:-cloud_pack/data}"
STEPS="${STEPS:-4000}"
BATCH="${BATCH:-4}"
CROP="${CROP:-256}"

echo "[$(date '+%F %T')] cloud overnight distill  steps=$STEPS batch=$BATCH crop=$CROP"
$PY -u train_cfm_distill.py \
  --teacher "$TEACHER" \
  --init-student "$INIT" \
  --pack-dir "$PACK" \
  --method consistency \
  --sample-loss l1_hf \
  --gt-weight 0 \
  --gt-hf-weight 0.45 \
  --gt-grad-energy-weight 0 \
  --gt-grad-weight 0 \
  --cd-weight 0 \
  --restore-best \
  --no-heun \
  --steps "$STEPS" \
  --channels 64 --depth 6 \
  --temporal 4 --batch "$BATCH" --crop "$CROP" \
  --lr 2e-4 \
  --integrate-steps 4 --teacher-steps 4 \
  --panel-every 200 \
  --panel-dir outputs/cloud_panels \
  --out outputs/cloud_sharp \
  2>&1 | tee -a logs/cloud_overnight.log

echo "[$(date '+%F %T')] done → outputs/cloud_sharp/"
