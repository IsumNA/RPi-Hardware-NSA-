#!/usr/bin/env bash
# Pull 24 GB Pi burst cache onto the AI server (run ON ai or via ssh ai).
set -euo pipefail
DEST="${PI_UNIQUE_CACHE:-/opt/datasets/PI_RAW/Pi_Unique_Cache}"
PI="${PI_SSH:-pi@10.3.31.153}"
SRC="${PI}:/home/pi/ctt-server-workspace/imx662/"
LOG="${1:-outputs/sync_pi_cache.log}"
mkdir -p "$(dirname "$LOG")" "$DEST"
echo "Syncing $SRC -> $DEST" | tee -a "$LOG"
rsync -avz --partial --append-verify --info=progress2 \
  "$SRC" "$DEST/" 2>&1 | tee -a "$LOG"
echo "Done. DNG count: $(find "$DEST" -name '*.dng' 2>/dev/null | wc -l)" | tee -a "$LOG"
