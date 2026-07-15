#!/usr/bin/env python3
"""Denoise for real use: motion means you cannot just average.

Modes
-----
  --input PATH
      Single-frame path (the common case under motion). Uses --ckpt if given
      (sharp RAW net), else dual-domain preserve.

  --burst DIR --motion
      Short burst with camera/subject motion: optical-flow warp + photometric
      weights so moving pixels do not ghost (static bg still gets multi-frame SNR).

  --burst DIR
      Static / tripod only — plain average (lab GT method). Do NOT use on motion.

  --proof-sharp
      Train a tiny single-frame net on synthetic data; must beat blur baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.solve import (  # noqa: E402
    _save_rgb, denoise_single_preserve, merge_burst, merge_burst_motion, proof_synthetic,
)
from nsa.raw_domain import packed_to_rgb  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--burst")
    p.add_argument("--input")
    p.add_argument("--motion", action="store_true",
                   help="flow-weighted merge (use with --burst when there is motion)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="default: 8 with --motion, 100 for static merge")
    p.add_argument("--ckpt", help="sharp_single.pt for --input")
    p.add_argument("--out", default="outputs/solved.png")
    p.add_argument("--display-gain", type=float, default=1.0)
    p.add_argument("--proof", action="store_true", help="static merge chirp proof")
    p.add_argument("--proof-sharp", action="store_true",
                   help="single-frame net vs blur proof")
    p.add_argument("--out-dir", default="outputs/solve_proof")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.proof_sharp:
        from nsa.sharp_single import proof_sharp_synth
        m = proof_sharp_synth(args.out_dir, steps=args.steps, device=args.device)
        print(json.dumps(m, indent=2, default=str), flush=True)
        return 0 if m.get("pred_beats_blur_psnr") else 1

    if args.proof:
        m = proof_synthetic(args.out_dir)
        print(json.dumps(m, indent=2), flush=True)
        return 0

    if args.burst and args.motion:
        k = args.max_frames or 8
        rgb = merge_burst_motion(args.burst, max_frames=k)
        mode = f"motion_merge×{k}"
    elif args.burst:
        k = args.max_frames or 100
        rgb = merge_burst(args.burst, max_frames=k)
        mode = f"static_merge×{k}"
    elif args.input:
        if args.ckpt:
            import torch
            from nsa.sharp_single import infer_sharp, load_sharp_model
            device = args.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            model = load_sharp_model(args.ckpt, device=device)
            packed = infer_sharp(model, args.input, device=device)
            rgb = packed_to_rgb(packed, args.display_gain)
            mode = "sharp_single_net"
        else:
            from nsa.raw_io import _load_any
            rgb = denoise_single_preserve(_load_any(Path(args.input)))
            mode = "single_preserve"
    else:
        p.error("pass --input, --burst, --proof, or --proof-sharp")

    out = Path(args.out)
    _save_rgb(out, np.clip(rgb, 0, 1))
    print(f"wrote {out}  mode={mode}  shape={tuple(rgb.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
