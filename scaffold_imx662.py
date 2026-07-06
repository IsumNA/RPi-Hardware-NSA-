#!/usr/bin/env python3
"""Create calibration/ + clean_scenes/ beside an existing PI_RAW tree.

Example
-------
  # Parent of the manager's PI_RAW:
  python scaffold_imx662.py -o /opt/datasets

  # Or point directly at PI_RAW (calibration goes in the parent):
  python scaffold_imx662.py -o /opt/datasets/PI_RAW
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.dataset_layout import IMX662_TARGET_AG_TAGS, MANAGER_SCENES, scaffold_imx662_project
from nsa.theme import banner, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", "-o", default="datasets",
                   help="PI_RAW root or its parent folder")
    p.add_argument("--gain", type=int, default=256,
                   help="analog gain for calibration folder name")
    p.add_argument("--imx662-tags", nargs="+", default=list(IMX662_TARGET_AG_TAGS),
                   help="IMX662 synthesis folder tags (default: ag12 ag24 ag48)")
    p.add_argument("--scenes", nargs="+", default=list(MANAGER_SCENES))
    p.add_argument("--flat-levels", type=int, default=12)
    args = p.parse_args()

    banner("IMX662 calibration scaffold (beside PI_RAW)")
    root = scaffold_imx662_project(
        args.output,
        gain=args.gain,
        imx662_ag_tags=tuple(args.imx662_tags),
        scenes=tuple(args.scenes),
        flat_levels=max(2, args.flat_levels),
    )
    log(f"Calibration + clean_scenes → {root}", "ok")
    log("Existing PI_RAW/Data is untouched. Open Dataset Studio to inspect.", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
