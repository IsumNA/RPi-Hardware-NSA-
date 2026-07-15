#!/usr/bin/env bash
# Run from laptop: push training code, start GPU training on AI, poll panels locally.
set -euo pipefail
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
AI="${NSA_AI_HOST:-ai}"
REMOTE_REPO='~/RPi-Hardware-NSA-'
PANEL_EVERY="${PANEL_EVERY:-100}"
LOCAL_PANELS="$REPO_LOCAL/outputs/panels_live"

echo "==> Syncing training files to $AI"
rsync -avz --progress \
  "$REPO_LOCAL/train_visual.py" \
  "$REPO_LOCAL/nsa/inference.py" \
  "$REPO_LOCAL/scripts/sync_pi_burst_cache.sh" \
  "$AI:$REMOTE_REPO/" 2>/dev/null || true
rsync -avz "$REPO_LOCAL/train_visual.py" "$AI:$REMOTE_REPO/"
rsync -avz "$REPO_LOCAL/nsa/inference.py" "$AI:$REMOTE_REPO/nsa/"
rsync -avz "$REPO_LOCAL/scripts/sync_pi_burst_cache.sh" "$AI:$REMOTE_REPO/scripts/"

echo "==> Stopping any prior training on AI"
ssh -o BatchMode=yes "$AI" 'pkill -f "run_demo.py|train_visual.py" 2>/dev/null || true; sleep 1'

echo "==> Starting Pi cache sync (background on AI)"
ssh -o BatchMode=yes "$AI" "chmod +x $REMOTE_REPO/scripts/sync_pi_burst_cache.sh; nohup $REMOTE_REPO/scripts/sync_pi_burst_cache.sh $REMOTE_REPO/outputs/sync_pi_cache.log >$REMOTE_REPO/outputs/sync_pi_cache.nohup 2>&1 &"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="outputs/train_logs/train_visual_${STAMP}.log"
echo "==> Starting visual training on AI (log: $LOG)"
ssh -o BatchMode=yes "$AI" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_REPO
mkdir -p outputs/train_logs outputs/panels
nohup .venv/bin/python -u train_visual.py \
  --panel-every ${PANEL_EVERY} \
  --panel-dir outputs/panels \
  >${LOG} 2>&1 &
echo TRAIN_PID=\$!
echo TRAIN_LOG=${LOG}
REMOTE

mkdir -p "$LOCAL_PANELS"
echo "==> Polling panels -> $LOCAL_PANELS (Ctrl+C to stop watching)"
echo "    ssh $AI 'tail -f $REMOTE_REPO/${LOG}'"

while true; do
  rsync -avz --delete \
    "$AI:$REMOTE_REPO/outputs/panels/" \
    "$LOCAL_PANELS/" 2>/dev/null || true
  if [[ -f "$LOCAL_PANELS/latest.png" ]]; then
    LATEST=$(readlink -f "$LOCAL_PANELS/latest.png" 2>/dev/null || echo "$LOCAL_PANELS/latest.png")
    echo "$(date +%H:%M:%S) panels synced ($(ls "$LOCAL_PANELS"/step_*.png 2>/dev/null | wc -l) snapshots) latest=$LATEST"
  fi
  sleep 30
done
