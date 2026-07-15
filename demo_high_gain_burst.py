#!/usr/bin/env python3
"""High-gain / large-N burst demo — prove 500+ frame averaging reaches GT.

Builds a comparison panel for a VERY noisy capture (default ag512)::

  single | mean-64 | mean-256 | mean-512 (=GT) | optional neural from single

Usage (on AI where full bursts live)::

  .venv/bin/python demo_high_gain_burst.py \\
      --burst-dir datasets/imx662_project/bursts/cabinet_H_2/ag512

  .venv/bin/python demo_high_gain_burst.py --burst-dir …/ag512 \\
      --checkpoint outputs/raw_denoiser_5ch.pt
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
from nsa.raw_domain import burst_clean, load_packed, packed_to_rgb, stack_fusion_input
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed

DISPLAY_GAIN = 8.0
OUT = ROOT / "outputs" / "high_gain_demo"


def _rgb(pk: np.ndarray) -> np.ndarray:
    return packed_to_rgb(pk, DISPLAY_GAIN)


def _panel(cols: list[np.ndarray], labels: list[str]) -> np.ndarray:
    h = min(c.shape[0] for c in cols)
    w = min(c.shape[1] for c in cols)
    strips = [np.clip(c[:h, :w], 0, 1) for c in cols]
    # label bar
    bar_h = 28
    labeled = []
    for rgb, lab in zip(strips, labels):
        canvas = np.zeros((h + bar_h, w, 3), np.float32)
        canvas[bar_h:] = rgb
        canvas[:bar_h] = 0.12
        labeled.append(canvas)
    return np.concatenate(labeled, axis=1)


def run(burst_dir: Path, out_dir: Path, ckpt: Path | None, ns: list[int]) -> dict:
    dngs = sorted(burst_dir.glob("*.dng"))
    if len(dngs) < max(ns):
        raise FileNotFoundError(f"Need ≥{max(ns)} DNGs, found {len(dngs)} in {burst_dir}")

    single = load_packed(dngs[0])
    gt_n = max(ns)
    gt = burst_clean(dngs, limit=gt_n)
    gt_rgb = _rgb(gt)
    single_rgb = _rgb(single)

    means = {}
    mean_rgbs = {}
    for n in ns:
        cfg = FusionConfig(n_frames=n, k_cap=float(n), mode="mean")
        fused, _ = fuse_burst_packed([load_packed(p) for p in dngs[:n]], cfg)
        means[n] = fused
        mean_rgbs[n] = _rgb(fused)

    metrics = {
        "burst_dir": str(burst_dir),
        "n_dngs": len(dngs),
        "gt_frames": gt_n,
        "psnr_single": round(psnr(single_rgb, gt_rgb), 2),
        "ssim_single": round(ssim(single_rgb, gt_rgb), 4),
    }
    for n in ns:
        metrics[f"psnr_mean_{n}"] = round(psnr(mean_rgbs[n], gt_rgb), 2)
        metrics[f"ssim_mean_{n}"] = round(ssim(mean_rgbs[n], gt_rgb), 4)

    cols = [single_rgb]
    labels = [f"SINGLE  {metrics['psnr_single']:.1f}dB"]
    for n in ns:
        cols.append(mean_rgbs[n])
        labels.append(f"MEAN-{n}  {metrics[f'psnr_mean_{n}']:.1f}dB")
    cols.append(gt_rgb)
    labels.append(f"GT ({gt_n}-frame mean)")

    if ckpt is not None and ckpt.is_file():
        import torch
        from nsa.config import ModelConfig
        from nsa.models import build_model
        from nsa.inference import to_image, to_tensor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        meta = blob.get("model", {})
        model = build_model(ModelConfig(
            model_family="raw_denoiser",
            base_channels=int(meta.get("base_channels", 16)),
            block_depth=int(meta.get("block_depth", 4)),
        ))
        state = blob["state_dict"]
        if not any(k.startswith("net.") for k in state) and hasattr(model, "net"):
            state = {f"net.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device).eval()

        # Neural from SINGLE high-gain frame → should approach GT.
        weight = np.ones(single.shape[:2] + (1,), np.float32)
        x5 = stack_fusion_input(single, weight, k_cap=1.0)
        with torch.no_grad():
            out = model(to_tensor(x5).to(device))
        den = to_image(out.cpu())
        den_rgb = _rgb(den)
        metrics["psnr_neural_single"] = round(psnr(den_rgb, gt_rgb), 2)
        metrics["ssim_neural_single"] = round(ssim(den_rgb, gt_rgb), 4)
        cols.append(den_rgb)
        labels.append(f"NEURAL←1  {metrics['psnr_neural_single']:.1f}dB")
        metrics["checkpoint"] = str(ckpt)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel = _panel(cols, labels)
    # Burn labels with PIL (simple)
    img = (panel * 255 + 0.5).astype(np.uint8)
    png = out_dir / f"{burst_dir.parent.name}_{burst_dir.name}_high_gain.png"
    Image.fromarray(img).save(png)
    metrics["panel"] = str(png)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burst-dir", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts/cabinet_H_2/ag512")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="optional raw_denoiser_5ch.pt — neural from single frame")
    ap.add_argument("--ns", default="64,256,512",
                    help="comma-separated mean depths to show")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    m = run(args.burst_dir, args.out, args.checkpoint, ns)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
