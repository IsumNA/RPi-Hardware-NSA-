#!/usr/bin/env python3
"""Create the IMX662 dataset folder template with capture guides.

Example
-------
  python scaffold_imx662.py --output ~/datasets/imx662_project
  python scaffold_imx662.py -o /opt/datasets/imx662_project --gain 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.dataset_layout import scaffold_imx662_project
from nsa.theme import banner, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", "-o", default="datasets/imx662_project",
                   help="where to create the template tree")
    p.add_argument("--gain", type=int, default=256)
    p.add_argument("--ag-tag", default="ag12")
    p.add_argument("--scenes", nargs="+",
                   default=["cabinet_D50_100", "colour_strips", "study"])
    p.add_argument("--flat-levels", type=int, default=12)
    args = p.parse_args()

    banner("IMX662 dataset scaffold")
    root = scaffold_imx662_project(
        args.output,
        gain=args.gain,
        ag_tag=args.ag_tag,
        scenes=tuple(args.scenes),
        flat_levels=max(2, args.flat_levels),
    )
    log(f"Template created → {root}", "ok")
    log("Open Dataset Studio in the GUI to track captures.", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
