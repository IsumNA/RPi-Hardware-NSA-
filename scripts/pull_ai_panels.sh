#!/usr/bin/env bash
# One-shot panel pull from AI server (run anytime from laptop).
set -euo pipefail
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
AI="${NSA_AI_HOST:-ai}"
LOCAL="$REPO_LOCAL/outputs/panels_live"
mkdir -p "$LOCAL"
rsync -avz "$AI:~/RPi-Hardware-NSA-/outputs/panels/" "$LOCAL/"
rsync -avz "$AI:~/RPi-Hardware-NSA-/outputs/validation_panel.png" "$REPO_LOCAL/outputs/" 2>/dev/null || true
echo "Synced to $LOCAL"
ls -lt "$LOCAL"/*.png 2>/dev/null | head -8
