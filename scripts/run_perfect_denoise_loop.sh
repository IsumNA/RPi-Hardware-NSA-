#!/usr/bin/env bash
# Continuous laptop-CPU sharp-denoise search. Resumes across reboots.
# Memory-safe: alpha-trim GT, few pairs, no LPIPS-in-loop, small crops.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PY:-.venv/bin/python}"
LOG=logs/perfect_denoise_loop.log
STATE=outputs/perfect_run/loop_state.env
LOCK=/tmp/nsa_perfect_denoise.lock
mkdir -p logs outputs/perfect_run outputs/perfect_panels
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Single instance (survives systemd Restart=)
exec 9>"$LOCK"
if ! flock -n 9; then
  log "Another perfect-denoise loop holds $LOCK — exiting"
  exit 0
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-6}"

SCENES="cabinet_H_2,cabinet_D50_100"
GAINS="128"
DEFAULT_INIT="outputs/cfm_l1/cfm_student_best.pt"
TEACHER="outputs/cfm_edm/cfm_teacher.pt"

pick_init() {
  local best=""
  # Prefer newest restore-best student under perfect_run
  best=$(ls -t outputs/perfect_run/*/cfm_student_best.pt 2>/dev/null | head -1 || true)
  if [[ -z "$best" ]]; then
    best=$(ls -t outputs/perfect_run/*/cfm_student.pt 2>/dev/null | head -1 || true)
  fi
  if [[ -n "$best" ]]; then
    echo "$best"
  else
    echo "$DEFAULT_INIT"
  fi
}

next_round() {
  local max=0 r
  for d in outputs/perfect_run/r*_cfm_hf outputs/perfect_run/r*_stageb outputs/perfect_run/r*_cfm_grad; do
    [[ -e "$d" ]] || continue
    r=$(basename "$d" | sed -n 's/^r\([0-9]*\)_.*/\1/p')
    [[ -n "$r" ]] || continue
    if (( r > max )); then max=$r; fi
  done
  # If current max round's stage C finished, start max+1; else resume max
  if [[ -f "outputs/perfect_run/r${max}_cfm_grad/cfm_student.pt" ]] \
     || [[ -f "outputs/perfect_run/r${max}_cfm_grad/cfm_student_best.pt" ]]; then
    echo $((max + 1))
  elif (( max == 0 )); then
    echo 1
  else
    echo "$max"
  fi
}

phase_done() {
  # $1 = out dir — only skip if the *final* artifacts exist (not mid-run best)
  local out="$1"
  [[ -f "$out/cfm_student_summary.json" || -f "$out/cfm_stage_b_summary.json" ]]
}

INIT="$(pick_init)"
ROUND="$(next_round)"
log "Boot/resume: INIT=$INIT  next_round=$ROUND  host=$(hostname) pid=$$"
echo "INIT=$INIT" > "$STATE"
echo "ROUND=$ROUND" >> "$STATE"
echo "UPDATED=$(date -Iseconds)" >> "$STATE"

# optional synth (non-fatal)
if [[ ! -f datasets/synth/bursts/cabinet_H_2__ag128.npy ]]; then
  log "Building synth clean frames from local bursts..."
  $PY -u build_synth_dataset.py --skip-srgb >> logs/build_synth.log 2>&1 || \
    log "WARN: synth build failed (continuing)"
fi

while true; do
  log "======== ROUND $ROUND ========"
  echo "INIT=$INIT" > "$STATE"
  echo "ROUND=$ROUND" >> "$STATE"
  echo "UPDATED=$(date -Iseconds)" >> "$STATE"

  # A) Lean CFM distill — structure match (HF)
  OUT="outputs/perfect_run/r${ROUND}_cfm_hf"
  PAN="outputs/perfect_panels/r${ROUND}_cfm_hf"
  mkdir -p "$OUT" "$PAN"
  if phase_done "$OUT"; then
    log "A) skip (already have checkpoint in $OUT)"
  else
    log "A) CFM distill l1_hf → $OUT (init=$INIT)"
    $PY -u train_cfm_distill.py \
      --teacher "$TEACHER" \
      --method consistency \
      --sample-loss l1_hf \
      --init-student "$INIT" \
      --gt-mode alpha_trim --gt-frames 16 \
      --scenes "$SCENES" --gains "$GAINS" \
      --stride 4 \
      --gt-weight 0 --gt-hf-weight 0.45 \
      --gt-grad-energy-weight 0 --gt-grad-weight 0 \
      --cd-weight 0 --restore-best --no-heun \
      --steps 800 --channels 64 --depth 6 \
      --temporal 4 --batch 1 --crop 128 \
      --lr 1.5e-4 \
      --integrate-steps 2 --teacher-steps 2 \
      --panel-every 100 --panel-dir "$PAN" --out "$OUT" \
      --no-onnx \
      >> "logs/perfect_r${ROUND}_cfm_hf.log" 2>&1 \
      && log "A done" \
      || log "A FAILED (see logs/perfect_r${ROUND}_cfm_hf.log)"
  fi
  if [[ -f "$OUT/cfm_student_best.pt" ]]; then INIT="$OUT/cfm_student_best.pt"
  elif [[ -f "$OUT/cfm_student.pt" ]]; then INIT="$OUT/cfm_student.pt"
  fi

  # B) Stronger Stage B
  OUTB="outputs/perfect_run/r${ROUND}_stageb"
  PANB="outputs/perfect_panels/r${ROUND}_stageb"
  mkdir -p "$OUTB" "$PANB"
  if phase_done "$OUTB"; then
    log "B) skip (already have checkpoint in $OUTB)"
  else
    log "B) Stage B residual=0.35 → $OUTB (stage-a=$INIT)"
    $PY -u train_cfm_stage_b.py \
      --stage-a "$INIT" \
      --scenes "$SCENES" --gains "$GAINS" \
      --gt-frames 16 --stride 4 \
      --steps 2000 --crop 160 --batch 1 --lr 2.5e-4 \
      --detail-channels 48 --detail-depth 4 \
      --residual-scale 0.35 \
      --loss charbonnier+perceptual+ffl \
      --panel-every 200 --panel-dir "$PANB" --out "$OUTB" \
      --no-onnx \
      >> "logs/perfect_r${ROUND}_stageb.log" 2>&1 \
      && log "B done" \
      || log "B FAILED (see logs/perfect_r${ROUND}_stageb.log)"
  fi

  # C) Alternate distill: l1_grad
  OUTC="outputs/perfect_run/r${ROUND}_cfm_grad"
  PANC="outputs/perfect_panels/r${ROUND}_cfm_grad"
  mkdir -p "$OUTC" "$PANC"
  if phase_done "$OUTC"; then
    log "C) skip (already have checkpoint in $OUTC)"
  else
    log "C) CFM distill l1_grad → $OUTC (init=$INIT)"
    $PY -u train_cfm_distill.py \
      --teacher "$TEACHER" \
      --method consistency \
      --sample-loss l1_grad \
      --init-student "$INIT" \
      --gt-mode alpha_trim --gt-frames 16 \
      --scenes "$SCENES" --gains "$GAINS" \
      --stride 4 \
      --gt-weight 0 --gt-hf-weight 0.25 \
      --gt-grad-energy-weight 0 --gt-grad-weight 0 \
      --cd-weight 0 --restore-best --no-heun \
      --steps 600 --channels 64 --depth 6 \
      --temporal 4 --batch 1 --crop 128 \
      --lr 1.0e-4 \
      --integrate-steps 2 --teacher-steps 2 \
      --panel-every 100 --panel-dir "$PANC" --out "$OUTC" \
      --no-onnx \
      >> "logs/perfect_r${ROUND}_cfm_grad.log" 2>&1 \
      && log "C done" \
      || log "C FAILED (see logs/perfect_r${ROUND}_cfm_grad.log)"
  fi
  if [[ -f "$OUTC/cfm_student_best.pt" ]]; then INIT="$OUTC/cfm_student_best.pt"
  elif [[ -f "$OUTC/cfm_student.pt" ]]; then INIT="$OUTC/cfm_student.pt"
  fi

  LATEST=$(ls -t outputs/perfect_panels/r${ROUND}_*/*.png 2>/dev/null | head -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    cp -f "$LATEST" outputs/perfect_panels/latest.png
    log "latest panel → outputs/perfect_panels/latest.png ($LATEST)"
  fi

  log "Round $ROUND complete. INIT=$INIT — continuing..."
  ROUND=$((ROUND + 1))
done
