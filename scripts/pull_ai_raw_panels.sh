#!/usr/bin/env bash
# One-shot rsync of raw training panels from AI (train_raw_visual --panel-dir outputs/raw_panels).
set -euo pipefail
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
AI="${NSA_AI_HOST:-ai}"
LOCAL="$REPO_LOCAL/outputs/raw_panels_live"
mkdir -p "$LOCAL"
rsync -avz "$AI:~/RPi-Hardware-NSA-/outputs/raw_panels/" "$LOCAL/"
echo "Synced to $LOCAL"
ls -lt "$LOCAL"/step_*.png 2>/dev/null | head -8
readlink -f "$LOCAL/latest.png" 2>/dev/null || ls -l "$LOCAL/latest.png" 2>/dev/null || true
