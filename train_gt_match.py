#!/usr/bin/env python3
"""Train the GT-matching RAW burst denoiser on real IMX662 / IMX662H bursts.

This is the production training entry point that replaces the scratchpad
``raw_train_ai.py`` experiment. It:

  * builds GT as the mean of ``--gt-frames`` (default **100**) packed RAW frames
  * trains on **every** held-out burst frame, randomly stacked as K=1..max frames
  * uses an anti-blur loss (Charbonnier + edge + HF FFT)
  * writes a checkpoint + held-out PSNR / sharpness panel

Run on the AI server (GPU + real DNG bursts)::

    python train_gt_match.py \\
        --bursts datasets/imx662_project/bursts \\
        --gains 128 256 512 \\
        --gt-frames 100 \\
        --max-frames 8 \\
        --steps 8000 \\
        --out outputs/gt_match

Smoke test (no DNGs, CPU)::

    python train_gt_match.py --synth --steps 200 --out outputs/gt_match_synth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nsa.gt_match import (  # noqa: E402
    TrainConfig, make_synthetic_burst_dir, train_gt_match,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bursts", default="datasets/imx662_project/bursts",
                   help="root of bursts/<scene>/ag<N>/*.dng")
    p.add_argument("--out", default="outputs/gt_match")
    p.add_argument("--gains", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--gt-frames", type=int, default=100,
                   help="frames averaged for the clean target (default 100)")
    p.add_argument("--max-frames", type=int, default=8,
                   help="max burst frames fused at train/infer (K)")
    p.add_argument("--min-k", type=int, default=1)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=662)
    p.add_argument("--device", default="cuda")
    p.add_argument("--w-edge", type=float, default=0.5)
    p.add_argument("--w-fft", type=float, default=0.05)
    p.add_argument("--synth", action="store_true",
                   help="generate a tiny synthetic burst and smoke-train")
    p.add_argument("--eval-every", type=int, default=500)
    args = p.parse_args()

    bursts = args.bursts
    if args.synth:
        synth_root = Path(args.out) / "_synth_bursts"
        make_synthetic_burst_dir(synth_root, n_frames=48, h=128, w=128)
        bursts = str(synth_root)
        print(f"synthetic bursts at {bursts}", flush=True)

    cfg = TrainConfig(
        bursts_root=bursts,
        out_dir=args.out,
        gains=list(args.gains),
        scenes=args.scenes,
        gt_frames=args.gt_frames if not args.synth else 24,
        max_frames=args.max_frames if not args.synth else 4,
        min_k=args.min_k,
        steps=args.steps,
        crop=min(args.crop, 96) if args.synth else args.crop,
        batch=args.batch if not args.synth else 2,
        lr=args.lr,
        base_channels=args.base_channels if not args.synth else 16,
        seed=args.seed,
        w_edge=args.w_edge,
        w_fft=args.w_fft,
        device=args.device,
        eval_every=args.eval_every if not args.synth else max(50, args.steps // 2),
    )
    result = train_gt_match(cfg)
    print(json_dumps(result), flush=True)


def json_dumps(obj):
    import json
    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    main()
