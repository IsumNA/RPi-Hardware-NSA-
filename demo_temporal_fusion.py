#!/usr/bin/env python3
"""Phase 2A demo — motion-aware packed-RAW burst fusion (no training).

Usage:
  # imx662h pair + nearest real burst (or override):
  python demo_temporal_fusion.py \\
      --pair datasets/PI_RAW/Data/cabinet_H_2/imx662h_ag128_test

  # Explicit burst (LCG proxy while HCG sync pending):
  python demo_temporal_fusion.py \\
      --pair datasets/PI_RAW/Data/cabinet_H_2/imx662h_ag128_test \\
      --burst-dir datasets/imx662_project/bursts/cabinet_D50_100/ag128 \\
      --frames 12

  # Scan all local imx662h ag128 pairs:
  python demo_temporal_fusion.py --scan --gain 128
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.inference import psnr, ssim
from nsa.raw_domain import burst_clean, load_packed, packed_to_rgb
from nsa.raw_io import _load_any
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed, resolve_burst_dir

DISPLAY_GAIN = 8.0
OUT = ROOT / "outputs" / "temporal_fusion_demo"


def _parse_gain(folder: Path) -> int:
    m = re.search(r"ag(\d+)", folder.name)
    return int(m.group(1)) if m else -1


def _find_pairs(data_root: Path, gain: int | None) -> list[Path]:
    pairs = sorted(data_root.rglob("imx662h_ag*_test"))
    if gain is not None:
        pairs = [p for p in pairs if _parse_gain(p) == gain]
    return [p for p in pairs if (p / "noisy.dng").is_file() and (p / "gt.tif").is_file()]


def _load_gt_rgb(pair_dir: Path) -> np.ndarray:
    return _load_any(pair_dir / "gt.tif")


def _align_gt(gt_rgb: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Resize full-res demosaiced GT to packed half-res for fair metrics."""
    th, tw = packed.shape[0], packed.shape[1]
    if gt_rgb.shape[0] == th and gt_rgb.shape[1] == tw:
        return gt_rgb
    return cv2.resize(gt_rgb, (tw, th), interpolation=cv2.INTER_AREA)


def _panel(noisy, fused, naive, gt) -> np.ndarray:
    rows = [packed_to_rgb(x, DISPLAY_GAIN) if x.shape[-1] == 4 else x
            for x in (noisy, fused, naive, gt)]
    strip = np.concatenate(rows, axis=1)
    return np.clip(strip, 0, 1)


def run_one(pair_dir: Path, burst_dir: Path | None, n_frames: int, out: Path) -> dict:
    pair_dir = pair_dir.resolve()
    gain = _parse_gain(pair_dir)
    single = load_packed(pair_dir / "noisy.dng")
    gt_rgb = _align_gt(_load_gt_rgb(pair_dir), single)

    burst_dir = burst_dir or resolve_burst_dir(pair_dir)
    if burst_dir is None or not burst_dir.is_dir():
        raise FileNotFoundError(
            f"No burst dir for {pair_dir.name}; pass --burst-dir "
            f"(e.g. datasets/imx662_project/bursts/cabinet_D50_100/ag128)"
        )

    dngs = sorted(burst_dir.glob("*.dng"))
    if len(dngs) < 8:
        raise FileNotFoundError(f"Only {len(dngs)} DNGs in {burst_dir}")

    cfg = FusionConfig(n_frames=min(n_frames, len(dngs)))
    frames = [load_packed(p) for p in dngs[: cfg.n_frames]]
    fused, _w = fuse_burst_packed(frames, cfg)
    naive = burst_clean(dngs, limit=cfg.n_frames)

    def rgb(x):
        return packed_to_rgb(x, DISPLAY_GAIN)

    h = min(*(a.shape[0] for a in (rgb(single), rgb(fused), rgb(naive), gt_rgb)))
    w = min(*(a.shape[1] for a in (rgb(single), rgb(fused), rgb(naive), gt_rgb)))
    s, f, n, g = [a[:h, :w] for a in (rgb(single), rgb(fused), rgb(naive), gt_rgb)]

    metrics = {
        "pair": str(pair_dir),
        "burst_dir": str(burst_dir),
        "n_frames": cfg.n_frames,
        "gain": gain,
        "psnr_single": round(psnr(s, g), 2),
        "psnr_fused": round(psnr(f, g), 2),
        "psnr_naive": round(psnr(n, g), 2),
        "ssim_single": round(ssim(s, g), 4),
        "ssim_fused": round(ssim(f, g), 4),
        "ssim_naive": round(ssim(n, g), 4),
    }

    out.mkdir(parents=True, exist_ok=True)
    tag = pair_dir.parent.name + "_" + pair_dir.name
    panel = _panel(single, fused, naive, gt_rgb)
    img = (panel * 255 + 0.5).astype(np.uint8)
    png = out / f"{tag}_fusion_panel.png"
    Image.fromarray(img).save(png)
    metrics["panel"] = str(png)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Phase 2A temporal fusion demo")
    ap.add_argument("--pair", type=Path, help="imx662h pair folder")
    ap.add_argument("--burst-dir", type=Path, help="override burst DNG directory")
    ap.add_argument("--frames", type=int, default=12, help="burst length (8–16)")
    ap.add_argument("--scan", action="store_true", help="all imx662h pairs")
    ap.add_argument("--gain", type=int, default=128, help="filter for --scan")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    data = ROOT / "datasets" / "PI_RAW" / "Data"
    pairs = _find_pairs(data, args.gain) if args.scan else [args.pair]
    if not pairs or pairs[0] is None:
        ap.error("specify --pair or --scan")

    default_burst = ROOT / "datasets/imx662_project/bursts/cabinet_D50_100/ag128"
    results = []
    for p in pairs:
        try:
            r = run_one(p, args.burst_dir or default_burst, args.frames, args.out)
            results.append(r)
            print(json.dumps(r, indent=2))
        except Exception as exc:
            print(f"SKIP {p}: {exc}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / "metrics.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {summary}  ({len(results)} ok)")


if __name__ == "__main__":
    main()
