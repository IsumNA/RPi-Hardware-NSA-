#!/usr/bin/env bash
# Continuous laptop-CPU sharp-denoise search. Resumes across reboots.
# v2: skip Stage B (makes soft mush); always warm-start from champion
# (best PSNR+grad_ratio≈1 among non-stageb students); balanced HF↔grad.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
LOG=logs/perfect_denoise_loop.log
STATE=outputs/perfect_run/loop_state.env
LOCK=/tmp/nsa_perfect_denoise.lock
mkdir -p logs outputs/perfect_run outputs/perfect_panels
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

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

# Champion = best non-stageb student by PSNR + |grad_ratio-1| penalty.
# Exclude soft-rejects / early-aborts (score alone promotes mush).
pick_champion() {
  $PY - <<'PY'
import json, glob
from pathlib import Path
rows = []
for p in glob.glob("outputs/perfect_run/*/cfm_student_summary.json"):
    d = Path(p).parent
    if "stageb" in d.name:
        continue
    if (d / "SOFT_REJECT").is_file():
        continue
    best = d / "cfm_student_best.pt"
    last = d / "cfm_student.pt"
    ckpt = best if best.is_file() else last
    if not ckpt.is_file():
        continue
    j = json.loads(Path(p).read_text())
    psnr = float(j.get("psnr_out", 0))
    gr = float(j.get("grad_ratio", 1))
    # Soft mush often still scores well with gr slightly under 1 — hard floor.
    if gr < 0.985 or gr > 1.04:
        continue
    score = 0.45 * psnr + 0.55 * (20.0 * max(0.0, 1.0 - abs(gr - 1.0)))
    rows.append((score, str(ckpt), psnr, gr, d.name))
rows.sort(reverse=True)
if not rows:
    # Prefer known sharp lineage over empty
    fallback = Path("outputs/perfect_run/r142_lpips_long/cfm_student_best.pt")
    print(str(fallback if fallback.is_file() else Path("outputs/cfm_l1/cfm_student_best.pt")))
else:
    s, ckpt, psnr, gr, name = rows[0]
    print(ckpt)
    import sys
    print(f"champion {name} psnr={psnr:.2f} gr={gr:.3f} score={s:.2f}", file=sys.stderr)
PY
}

next_round() {
  local max=0 r
  for d in outputs/perfect_run/r*_cfm_bal outputs/perfect_run/r*_cfm_hf outputs/perfect_run/r*_cfm_grad; do
    [[ -e "$d" ]] || continue
    r=$(basename "$d" | sed -n 's/^r\([0-9]*\)_.*/\1/p')
    [[ -n "$r" ]] || continue
    if (( r > max )); then max=$r; fi
  done
  # Prefer new balanced rounds starting at 100+
  if [[ -f "outputs/perfect_run/r${max}_cfm_bal/cfm_student_summary.json" ]]; then
    echo $((max + 1))
  elif (( max >= 100 )); then
    echo "$max"
  else
    echo 100
  fi
}

phase_done() {
  local out="$1"
  [[ -f "$out/cfm_student_summary.json" ]]
}

INIT="$(pick_champion 2> >(tee -a "$LOG" >&2))"
ROUND="$(next_round)"
log "Boot/resume v2: INIT=$INIT  next_round=$ROUND  host=$(hostname) pid=$$"
echo "INIT=$INIT" > "$STATE"
echo "ROUND=$ROUND" >> "$STATE"
echo "UPDATED=$(date -Iseconds)" >> "$STATE"
echo "RECIPE=v2_balanced_no_stageb" >> "$STATE"

while true; do
  INIT="$(pick_champion 2> >(tee -a "$LOG" >&2))"
  log "======== ROUND $ROUND (balanced, no Stage B) INIT=$INIT ========"
  echo "INIT=$INIT" > "$STATE"
  echo "ROUND=$ROUND" >> "$STATE"
  echo "UPDATED=$(date -Iseconds)" >> "$STATE"

  # Balanced distill: HF structure + mild grad match, short steps, restore-best
  OUT="outputs/perfect_run/r${ROUND}_cfm_bal"
  PAN="outputs/perfect_panels/r${ROUND}_cfm_bal"
  mkdir -p "$OUT" "$PAN"
  if phase_done "$OUT"; then
    log "skip $OUT (already complete)"
  else
    # Match champion architecture: RawDenoiser dump-distill vs CFM consistency.
    CHAMP_METHOD="consistency"
    if [[ -f "$(dirname "$INIT")/cfm_student_summary.json" ]]; then
      CHAMP_METHOD="$($PY -c "import json; print(json.load(open('$(dirname "$INIT")/cfm_student_summary.json')).get('method','consistency'))" 2>/dev/null || echo consistency)"
    fi
    DUMP_DIR=datasets/cfm_dumps_cpu
    if [[ -f datasets/cfm_dumps_mixed/index.json ]]; then
      DUMP_DIR=datasets/cfm_dumps_mixed
    fi
    if [[ "$CHAMP_METHOD" == "regression_match" && -f "$DUMP_DIR/index.json" ]]; then
      # Prefer longer LPIPS / texture recipes — short tiny-lr/grad runs only soft-reject.
      case $((ROUND % 4)) in
        0) D_LOSS=l1_lpips; D_HF=0.12; D_LR=1e-5;  D_STEPS=1200; D_TAG="lpips-mid" ;;
        1) D_LOSS=l1_lpips; D_HF=0.18; D_LR=6e-6;  D_STEPS=1500; D_TAG="lpips-slow" ;;
        2) D_LOSS=l1_hf;    D_HF=0.25; D_LR=1.2e-5; D_STEPS=1000; D_TAG="l1hf" ;;
        *) D_LOSS=l1;       D_HF=0.28; D_LR=1e-5;  D_STEPS=900;  D_TAG="hf-heavy" ;;
      esac
      log "micro-finetune regression_match dumps ($D_TAG loss=$D_LOSS lr=$D_LR hf=$D_HF dir=$DUMP_DIR) → $OUT"
      $PY -u train_cfm_distill.py \
        --method regression_match \
        --dump-dir "$DUMP_DIR" \
        --sample-loss "$D_LOSS" \
        --init-student "$INIT" \
        --gt-hf-weight "$D_HF" \
        --restore-best --early-abort-soft --early-abort-after 200 \
        --steps "$D_STEPS" \
        --channels 64 --depth 6 --batch 2 --crop 160 --lr "$D_LR" \
        --panel-every 50 --panel-dir "$PAN" --out "$OUT" \
        --no-onnx \
        >> "logs/perfect_r${ROUND}_cfm_bal.log" 2>&1 \
        && log "micro-finetune done" \
        || log "micro-finetune FAILED (see logs/perfect_r${ROUND}_cfm_bal.log)"
      if grep -q 'early-abort soft' "logs/perfect_r${ROUND}_cfm_bal.log" 2>/dev/null; then
        log "EARLY-ABORT $OUT (soft drift — skipped likely reject)"
        # Do not let score-only pick_champion promote mushy early peaks.
        touch "$OUT/SOFT_REJECT"
        if [[ -f "$OUT/cfm_student_summary.json" ]]; then
          mv -f "$OUT/cfm_student_summary.json" "$OUT/cfm_student_summary.soft_reject.json"
        fi
      fi
    else
      # Consistency path for non-dump champions; match init I/O via gain probe.
      log "micro-finetune l1_grad consistency (tiny lr) → $OUT"
      $PY -u train_cfm_distill.py \
        --teacher "$TEACHER" \
        --method consistency \
        --sample-loss l1_grad \
        --init-student "$INIT" \
        --gt-mode alpha_trim --gt-frames 16 \
        --scenes "$SCENES" --gains "$GAINS" \
        --stride 4 \
        --gt-weight 0 --gt-hf-weight 0.1 \
        --gt-grad-energy-weight 0 --gt-grad-weight 0 \
        --cd-weight 0 --restore-best --no-heun \
        --steps 300 --channels 64 --depth 6 \
        --temporal 4 --batch 1 --crop 160 \
        --lr 2e-5 \
        --integrate-steps 3 --teacher-steps 3 \
        --panel-every 50 --panel-dir "$PAN" --out "$OUT" \
        --no-onnx \
        >> "logs/perfect_r${ROUND}_cfm_bal.log" 2>&1 \
        && log "micro-finetune done" \
        || log "micro-finetune FAILED (see logs/perfect_r${ROUND}_cfm_bal.log)"
    fi
    # Reject if worse than champion (don't chain soft models)
    if [[ -f "$OUT/cfm_student_summary.json" && -f outputs/perfect_run/champion/cfm_student_summary.json ]]; then
      if ! OUT="$OUT" $PY - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
def score(p):
    j = json.loads(Path(p).read_text())
    psnr, gr = float(j["psnr_out"]), float(j["grad_ratio"])
    return 0.45 * psnr + 0.55 * (20.0 * max(0.0, 1.0 - abs(gr - 1.0)))
s_new = score(out / "cfm_student_summary.json")
s_ch = score("outputs/perfect_run/champion/cfm_student_summary.json")
print(f"score new={s_new:.2f} champ={s_ch:.2f}")
raise SystemExit(0 if s_new >= s_ch - 0.05 else 1)
PY
      then
        log "REJECT $OUT (worse than champion) — keeping champion as INIT"
        rm -f "$OUT/cfm_student_summary.json"
      fi
    fi
  fi

  LATEST=$(ls -t "$PAN"/*.png 2>/dev/null | head -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    cp -f "$LATEST" outputs/perfect_panels/latest.png
    log "latest panel → outputs/perfect_panels/latest.png"
  fi

  # Copy champion to a stable path for cloud/export
  CHAMP="$(pick_champion 2>/dev/null)"
  if [[ -f "$CHAMP" ]]; then
    mkdir -p outputs/perfect_run/champion
    cp -f "$CHAMP" outputs/perfect_run/champion/cfm_student_best.pt
    # copy matching summary if any
    SUM="$(dirname "$CHAMP")/cfm_student_summary.json"
    [[ -f "$SUM" ]] && cp -f "$SUM" outputs/perfect_run/champion/cfm_student_summary.json
    log "champion checkpoint → outputs/perfect_run/champion/ ($(basename "$(dirname "$CHAMP")"))"
  fi

  log "Round $ROUND complete — continuing..."
  ROUND=$((ROUND + 1))
done
