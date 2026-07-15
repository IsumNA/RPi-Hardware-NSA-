#!/usr/bin/env python3
"""Train / run conditional rectified-flow RAW denoise (generative — not MMSE).

This is the modern fix for the soft middle-column blur: sample a sharp clean
from p(clean|noisy) instead of regressing to E[clean|noisy].

Train on AI server (uses full burst diversity → multi-frame GT)::

    python train_flow_raw.py \\
      --bursts datasets/imx662_project/bursts \\
      --gains 128 256 512 \\
      --gt-frames 100 \\
      --steps 12000 \\
      --out outputs/flow_raw

Infer one frame (motion-safe)::

    python train_flow_raw.py --infer path/to/noisy.dng \\
      --ckpt outputs/flow_raw/flow_raw.pt \\
      --out outputs/flow_denoised.png

Synth proof (CPU)::

    python train_flow_raw.py --proof --steps 600 --out outputs/flow_proof
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.flow_raw import (  # noqa: E402
    FlowTrainConfig, infer_flow, load_flow_model, proof_flow_synth, train_flow_raw,
)
from nsa.raw_domain import packed_to_rgb  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bursts", default="datasets/imx662_project/bursts")
    p.add_argument("--out", default="outputs/flow_raw")
    p.add_argument("--gains", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--gt-frames", type=int, default=100)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--sample-steps", type=int, default=10,
                   help="ODE steps at inference (8–20 is typical)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=662)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--proof", action="store_true")
    p.add_argument("--infer", metavar="PATH", help="denoise one noisy DNG/npy")
    p.add_argument("--ckpt", default="")
    p.add_argument("--display-gain", type=float, default=8.0)
    args = p.parse_args()

    if args.proof:
        m = proof_flow_synth(args.out, steps=args.steps, device="cpu")
        print(json.dumps(m, indent=2), flush=True)
        print(f"\nflow sharp={m['sharp_flow']:.3f} blur={m['sharp_blur']:.3f} | "
              f"PSNR flow={m['psnr_flow']:.2f} blur={m['psnr_blur']:.2f} → "
              f"{'PASS' if m['pass'] else 'FAIL'}", flush=True)
        return 0 if m["pass"] else 1

    if args.infer:
        if not args.ckpt:
            p.error("--infer needs --ckpt")
        import torch
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        net, meta = load_flow_model(args.ckpt, device=device)
        steps = args.sample_steps or meta["sample_steps"]
        packed = infer_flow(net, args.infer, device=device, steps=steps)
        rgb = packed_to_rgb(packed, args.display_gain)
        out = Path(args.out)
        if out.suffix.lower() not in {".png", ".jpg", ".tif", ".tiff"}:
            out = out / "flow_denoised.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
        np.save(str(out.with_suffix(".npy")), packed)
        print(f"wrote {out}  (+ packed {out.with_suffix('.npy')})  steps={steps}",
              flush=True)
        return 0

    cfg = FlowTrainConfig(
        bursts_root=args.bursts, out_dir=args.out, gains=list(args.gains),
        scenes=args.scenes, gt_frames=args.gt_frames, steps=args.steps,
        crop=args.crop, batch=args.batch, lr=args.lr,
        base_channels=args.base_channels, sample_steps=args.sample_steps,
        device=args.device, seed=args.seed, eval_every=args.eval_every,
    )
    result = train_flow_raw(cfg)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
