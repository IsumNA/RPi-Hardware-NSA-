#!/usr/bin/env python3
"""High-gain denoise via large-N temporal mean (the GT-quality path).

For static low-light / high-gain captures, averaging 256–512+ frames *is*
the clean image. Use this when you have a full burst; the neural net is only
for short bursts where you cannot wait.

Usage::

  python denoise_burst_mean.py \\
      --burst-dir datasets/imx662_project/bursts/cabinet_H_2/ag512 \\
      --frames 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.inference import psnr, ssim
from nsa.raw_domain import burst_clean, load_packed, packed_to_rgb
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed

DISPLAY_GAIN = 8.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burst-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=512,
                    help="frames to average (default 512)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "burst_mean_out")
    args = ap.parse_args()

    dngs = sorted(args.burst_dir.glob("*.dng"))
    n = min(args.frames, len(dngs))
    if n < 8:
        print(f"Need more DNGs in {args.burst_dir}", file=sys.stderr)
        return 1

    cfg = FusionConfig(n_frames=n, k_cap=float(n), mode="mean")
    fused, _ = fuse_burst_packed([load_packed(p) for p in dngs[:n]], cfg)
    gt = burst_clean(dngs, limit=n)  # same N → identity check
    single = load_packed(dngs[0])

    fr, gr, sr = (packed_to_rgb(x, DISPLAY_GAIN) for x in (fused, gt, single))
    metrics = {
        "burst_dir": str(args.burst_dir),
        "frames": n,
        "psnr_single_vs_mean": round(psnr(sr, fr), 2),
        "ssim_single_vs_mean": round(ssim(sr, fr), 4),
        "psnr_mean_vs_gt": round(psnr(fr, gr), 2),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    panel = np.concatenate([sr, fr], axis=1)
    Image.fromarray((np.clip(panel, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        args.out / f"{args.burst_dir.name}_mean{n}.png")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out / (args.burst_dir.name + f'_mean{n}.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
