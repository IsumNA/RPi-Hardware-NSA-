#!/usr/bin/env bash
# Run NSA training on the AI server (full DNG dataset at /opt/datasets/PI_RAW).
set -euo pipefail
SSH_HOST="${NSA_AI_HOST:-ai}"
REPO='~/RPi-Hardware-NSA-'
LOG_DIR='outputs/train_logs'
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG_DIR}/run_demo_${STAMP}.log"
EXTRA_ARGS=("$@")

ssh -o BatchMode=yes "${SSH_HOST}" bash -s -- "${LOG}" "${EXTRA_ARGS[@]:-}" <<'REMOTE'
set -euo pipefail
LOG="$1"
shift
cd ~/RPi-Hardware-NSA-
mkdir -p outputs/train_logs
nohup .venv/bin/python -u run_demo.py --no-window "$@" >"$LOG" 2>&1 &
echo "PID $!"
echo "LOG $LOG"
sleep 3
tail -25 "$LOG"
REMOTE
