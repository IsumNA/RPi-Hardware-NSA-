#!/usr/bin/env python3
"""DEPRECATED — use ``train_gt_match.py`` at the repo root instead.

This scratchpad script trained a single-frame RawDenoiser on every 6th frame
with a 256-frame GT. That skipped most of the burst and still used a pure
regression loss (blurry posterior mean).

The production path:

    python train_gt_match.py \\
        --bursts datasets/imx662_project/bursts \\
        --gt-frames 100 --max-frames 8 --steps 8000
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

print(
    "scratchpad/raw_train_ai.py is deprecated.\n"
    "Redirecting to train_gt_match.py (RAW burst fusion → ~100-frame GT).\n",
    flush=True,
)
sys.argv = [str(Path(__file__).resolve().parents[1] / "train_gt_match.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
