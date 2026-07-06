#!/usr/bin/env python3
"""Build a clean ground-truth image by temporally averaging a RAW burst.

Example
-------
  # After saving 64 DNG frames to bursts/cabinet_D50_100/take01/:
  python capture_gt.py --burst bursts/cabinet_D50_100/take01 \\
      --output clean_scenes/cabinet_D50_100/gt_01.png --min-frames 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.gt_capture import burst_folder_to_gt
from nsa.theme import banner, console, kv_table, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--burst", "-b", required=True,
                   help="folder of sequential RAW/PNG burst frames")
    p.add_argument("--output", "-o", required=True,
                   help="output GT image path (e.g. clean_scenes/scene/gt_01.png)")
    p.add_argument("--min-frames", type=int, default=8,
                   help="minimum frames required in the burst folder")
    p.add_argument("--max-side", type=int, default=0,
                   help="downscale before averaging (0 = full resolution)")
    p.add_argument("--no-align", action="store_true",
                   help="skip ECC alignment between burst frames")
    args = p.parse_args()

    banner("Ground truth  ·  temporal average")
    try:
        manifest = burst_folder_to_gt(
            args.burst, args.output,
            min_frames=max(2, args.min_frames),
            max_side=max(0, args.max_side),
            align=not args.no_align,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        log(str(exc), "err")
        return 1

    console.print()
    console.print(kv_table([
        ("burst", manifest["burst"]),
        ("frames", str(manifest["frames_used"])),
        ("output", manifest["output"]),
        ("size", f"{manifest['width']}×{manifest['height']}"),
    ], title="GT ready"))
    log("Use this file in clean_scenes/ for simulate_dataset.py or the GUI wizard.",
        "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
