#!/bin/bash
# One-shot: point NSA at real PI_RAW on the AI machine (no copy to Desktop).
set -euo pipefail
cd "$(dirname "$0")"
python3 setup_denoise_hw_data.py --use /opt/datasets/PI_RAW --write-config --no-clone "$@"
