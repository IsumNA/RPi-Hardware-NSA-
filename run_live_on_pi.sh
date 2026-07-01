#!/usr/bin/env bash
# Sync outputs/model.pt to the Pi and open SSH live testing on the CSI camera.
#
# One-time:  ssh-copy-id user@pi  and  Host rpi  in ~/.ssh/config
# Then:       ./run_live_on_pi.sh
#
# Or click LIVE TEST in the GUI (same flow on the AI server).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec python3 -m nsa.pi_remote
