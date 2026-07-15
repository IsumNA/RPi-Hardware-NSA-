#!/usr/bin/env python3
"""Infer with a GT-matching burst denoiser (single frame or multi-frame).

Examples::

    # single noisy DNG (K=1 — sharp as information allows)
    python infer_gt_match.py --ckpt outputs/gt_match/gt_match_denoiser.pt \\
        --input path/to/noisy.dng --out outputs/denoised.png

    # fuse a burst (K=8..32 → approaches the ~100-frame GT look)
    python infer_gt_match.py --ckpt outputs/gt_match/gt_match_denoiser.pt \\
        --burst datasets/imx662_project/bursts/cabinet_H_2/ag512 \\
        --max-frames 16 --out outputs/fused.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nsa.gt_match import (  # noqa: E402
    _BURST_EXTS, infer_burst, load_model, packed_to_rgb,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", help="single noisy DNG / npy")
    p.add_argument("--burst", help="folder of burst frames")
    p.add_argument("--max-frames", type=int, default=0,
                   help="cap frames from --burst (0 = model max)")
    p.add_argument("--out", default="outputs/gt_match_infer.png")
    p.add_argument("--display-gain", type=float, default=8.0)
    p.add_argument("--save-packed", default="",
                   help="optional path to also save packed RAW .npy")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.ckpt, device=device)
    paths: list[Path] = []
    if args.burst:
        d = Path(args.burst)
        paths = sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in _BURST_EXTS
        )
        cap = args.max_frames or model.max_frames
        # take a spread across the burst (not just the first N — those built GT)
        if len(paths) > cap:
            idx = np.linspace(len(paths) // 2, len(paths) - 1, cap).astype(int)
            paths = [paths[i] for i in idx]
    elif args.input:
        paths = [Path(args.input)]
    else:
        p.error("pass --input or --burst")

    if not paths:
        p.error("no frames found")

    print(f"fusing K={len(paths)} on {device}: {[x.name for x in paths[:8]]}"
          f"{'…' if len(paths) > 8 else ''}", flush=True)
    packed = infer_burst(model, paths, device=device)
    rgb = packed_to_rgb(packed, args.display_gain)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
    print(f"wrote {out}", flush=True)
    if args.save_packed:
        np.save(args.save_packed, packed)
        print(f"wrote {args.save_packed}", flush=True)


if __name__ == "__main__":
    main()
