#!/usr/bin/env python3
"""Genuinely solve IMX662 denoise: burst-merge (= GT) or detail-preserving single.

This is the answer to "make it look like the 100-frame average without blur":

  * If you have a burst → average it in linear RGB/RAW. That **is** the GT.
  * If you have one frame → dual-domain preserve (not L1-NAFNet plastic blur).

Examples
--------
  # Perfect: merge 100 frames from a real burst (equals your GT method)
  python solve_denoise.py --burst datasets/imx662_project/bursts/cabinet_H_2/ag512 \\
      --max-frames 100 --out outputs/solved.png

  # Single frame fallback (keeps resolution bars)
  python solve_denoise.py --input path/to/noisy.dng --out outputs/solved_single.png

  # Proof on synthetic resolution chart (no camera needed)
  python solve_denoise.py --proof --out-dir outputs/solve_proof
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.solve import (  # noqa: E402
    denoise_single_preserve, merge_burst, proof_synthetic, solve, _save_rgb,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--burst", help="folder of burst frames (DNG/PNG/…)")
    p.add_argument("--input", help="single noisy frame")
    p.add_argument("--max-frames", type=int, default=100,
                   help="frames to average from --burst (default 100 = GT look)")
    p.add_argument("--no-align", action="store_true")
    p.add_argument("--out", default="outputs/solved.png")
    p.add_argument("--proof", action="store_true",
                   help="run synthetic chirp proof (clean|noisy|blurry|preserve|merge)")
    p.add_argument("--out-dir", default="outputs/solve_proof")
    args = p.parse_args()

    if args.proof:
        m = proof_synthetic(args.out_dir)
        print(json.dumps(m, indent=2), flush=True)
        print(
            f"\nChirp contrast recovery vs clean:\n"
            f"  blurry (what L1 nets do):  {m['recovery_blurry']:.3f}\n"
            f"  single preserve:           {m['recovery_preserve']:.3f}\n"
            f"  100-frame merge (= GT):    {m['recovery_merge100']:.3f}\n"
            f"panel -> {args.out_dir}/proof_panel.png",
            flush=True,
        )
        if m["recovery_merge100"] < 0.9:
            print("WARNING: merge recovery < 0.9 — unexpected", flush=True)
            return 1
        return 0

    if args.burst:
        rgb = merge_burst(args.burst, max_frames=args.max_frames,
                          align=not args.no_align)
        mode = f"burst_merge×{args.max_frames}"
    elif args.input:
        from nsa.raw_io import _load_any
        noisy = _load_any(Path(args.input))
        rgb = denoise_single_preserve(noisy)
        mode = "single_preserve"
    else:
        p.error("pass --burst, --input, or --proof")

    out = Path(args.out)
    _save_rgb(out, rgb)
    print(f"wrote {out}  mode={mode}  shape={rgb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
