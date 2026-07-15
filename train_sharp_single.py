#!/usr/bin/env python3
"""Train a sharp SINGLE-FRAME RAW denoiser (for real scenes with motion).

Why not just average?
  Averaging only works on static content. Under motion it ghosts. Deploy needs
  one-frame denoise. Lab GT is still a static multi-frame average — that teaches
  the mapping; at inference you pass a single noisy frame.

    python train_sharp_single.py \\
      --bursts datasets/imx662_project/bursts \\
      --gt-frames 100 --steps 10000 --w-gan 0.05 \\
      --out outputs/sharp_single

Smoke proof (synthetic, CPU)::

    python train_sharp_single.py --proof --steps 400 --out outputs/sharp_proof
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.sharp_single import SharpTrainConfig, proof_sharp_synth, train_sharp_single  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bursts", default="datasets/imx662_project/bursts")
    p.add_argument("--out", default="outputs/sharp_single")
    p.add_argument("--gains", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--gt-frames", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--w-edge", type=float, default=0.75)
    p.add_argument("--w-fft", type=float, default=0.1)
    p.add_argument("--w-gan", type=float, default=0.05,
                   help="PatchGAN weight (0=off). Fights plastic blur.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=662)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--proof", action="store_true",
                   help="synthetic single-frame proof vs blur baseline")
    args = p.parse_args()

    if args.proof:
        m = proof_sharp_synth(args.out, steps=args.steps, device="cpu")
        print(json.dumps(m, indent=2, default=str), flush=True)
        ok = m.get("pred_beats_blur_psnr", False)
        print(f"\nsingle-frame pred PSNR {m['psnr_pred']:.2f} vs blur "
              f"{m['psnr_blur']:.2f}  → {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1

    cfg = SharpTrainConfig(
        bursts_root=args.bursts, out_dir=args.out, gains=list(args.gains),
        scenes=args.scenes, gt_frames=args.gt_frames, steps=args.steps,
        crop=args.crop, batch=args.batch, lr=args.lr,
        base_channels=args.base_channels, w_edge=args.w_edge, w_fft=args.w_fft,
        w_gan=args.w_gan, device=args.device, seed=args.seed,
        eval_every=args.eval_every,
    )
    result = train_sharp_single(cfg)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
