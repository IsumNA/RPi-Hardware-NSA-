#!/usr/bin/env bash
# Stronger CFM student (L1+high-freq, wider net) → QAT/INT8 → multi-target export.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/cfm_prod outputs/cfm_prod_int8 outputs/cfm_deploy \
         outputs/cfm_prod_panels outputs/cfm_qat_panels

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/cfm_prod_pipeline.log; }

TEACHER="${TEACHER:-outputs/cfm_teacher.pt}"
if [[ ! -f "$TEACHER" ]]; then
  log "ERROR: teacher missing: $TEACHER"
  exit 1
fi

log "=== 1/3 Stronger student distill (96ch×6, l1_hf, 4500 steps) ==="
.venv/bin/python -u train_cfm_distill.py \
  --teacher "$TEACHER" \
  --method consistency \
  --sample-loss l1_hf \
  --gt-weight 0 \
  --steps 4500 \
  --channels 96 \
  --depth 6 \
  --temporal 4 \
  --batch 2 \
  --crop 256 \
  --lr 5e-4 \
  --integrate-steps 6 \
  --teacher-steps 8 \
  --cd-weight 0.5 \
  --panel-every 500 \
  --panel-dir outputs/cfm_prod_panels \
  --out outputs/cfm_prod \
  2>&1 | tee -a logs/cfm_prod_distill.log

# train_cfm_distill writes cfm_student.pt into --out when out is a dir… check
if [[ -f outputs/cfm_prod/cfm_student.pt ]]; then
  STUDENT=outputs/cfm_prod/cfm_student.pt
elif [[ -f outputs/cfm_student.pt ]]; then
  # legacy: script may write to outputs/ root — move into prod dir
  mkdir -p outputs/cfm_prod
  cp -f outputs/cfm_student.pt outputs/cfm_prod/
  cp -f outputs/cfm_student.onnx outputs/cfm_prod/ 2>/dev/null || true
  STUDENT=outputs/cfm_prod/cfm_student.pt
else
  log "ERROR: student checkpoint not found after distill"
  exit 1
fi
log "Student: $STUDENT"

log "=== 2/3 QAT fine-tune (1500 steps) → INT8-ready weights ==="
.venv/bin/python -u train_cfm_qat.py \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --steps 1500 \
  --sample-loss l1_hf \
  --gt-weight 0 \
  --lr 1e-4 \
  --panel-dir outputs/cfm_qat_panels \
  --out outputs/cfm_prod_int8 \
  2>&1 | tee -a logs/cfm_prod_qat.log

# Prefer QAT fp32 weights for multi-export (best accuracy); also keep int8 eval ckpt
EXPORT_CKPT=outputs/cfm_prod_int8/cfm_student_qat_fp32.pt
if [[ ! -f "$EXPORT_CKPT" ]]; then
  EXPORT_CKPT=outputs/cfm_prod_int8/cfm_student_int8.pt
fi
if [[ ! -f "$EXPORT_CKPT" ]]; then
  EXPORT_CKPT="$STUDENT"
fi

log "=== 3/3 Multi-target export (cpu, brainstorm, hailo10h, deepx, intel_npu) ==="
.venv/bin/python -u export_cfm_targets.py \
  --checkpoint "$EXPORT_CKPT" \
  --out outputs/cfm_deploy \
  --int8 \
  --patch 256 \
  2>&1 | tee -a logs/cfm_prod_export.log

log "DONE. Artifacts:"
log "  FP student:  $STUDENT"
log "  QAT/INT8:    outputs/cfm_prod_int8/"
log "  Deploy pack: outputs/cfm_deploy/ (see deploy_manifest.json)"
