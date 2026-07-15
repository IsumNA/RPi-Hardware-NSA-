#!/usr/bin/env python3
"""Phase 3 hybrid demo — burst fusion + 5ch RawDenoiser (inference only).

End-to-end on laptop (no training):
  burst DNGs → temporal fusion → stack_fusion_input (5ch)
  → raw_denoiser_5ch.pt → packed RGB panel vs GT

Usage:
  python demo_hybrid_denoise.py

  python demo_hybrid_denoise.py \\
      --burst-dir datasets/imx662_project/bursts/cabinet_D50_100/ag128 \\
      --pair datasets/PI_RAW/Data/cabinet_D50_100/imx662_ag128_test \\
      --checkpoint outputs/raw_denoiser_5ch.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.config import ModelConfig
from nsa.inference import psnr, ssim, to_image, to_tensor
from nsa.models import build_model
from nsa.raw_domain import (
    burst_clean,
    load_packed,
    packed_to_rgb,
    stack_fusion_input,
)
from nsa.raw_io import _load_any
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed

DISPLAY_GAIN = 8.0
DEFAULT_BURST = ROOT / "datasets/imx662_project/bursts/cabinet_D50_100/ag128"
DEFAULT_PAIR = ROOT / "datasets/PI_RAW/Data/cabinet_D50_100/imx662_ag128_test"
DEFAULT_CKPT = ROOT / "outputs/raw_denoiser_5ch.pt"
OUT = ROOT / "outputs" / "hybrid_denoise_demo"
FUSION_FRAMES = 12
GT_BURST_FRAMES = 256


def _load_gt_rgb(pair_dir: Path | None, burst_files: list[Path]) -> tuple[np.ndarray, str]:
    """Prefer local gt.tif; fall back to burst temporal mean."""
    if pair_dir is not None:
        for name in ("gt.tif", "gt.png"):
            p = pair_dir / name
            if p.is_file():
                return _load_any(p), f"local:{p.name}"
    gt = burst_clean(burst_files, limit=GT_BURST_FRAMES)
    return gt, "burst_mean"


def _align_gt(gt: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Resize GT to packed half-res. Accepts RGB (H,W,3) or packed RAW (H,W,4)."""
    th, tw = packed.shape[0], packed.shape[1]
    if gt.ndim == 3 and gt.shape[-1] == 3:
        if gt.shape[0] == th and gt.shape[1] == tw:
            return gt
        return cv2.resize(gt, (tw, th), interpolation=cv2.INTER_AREA)
    if gt.shape[0] == th and gt.shape[1] == tw:
        return gt
    return cv2.resize(gt, (tw, th), interpolation=cv2.INTER_AREA)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("model", {})
    cfg = ModelConfig(
        model_family="raw_denoiser",
        base_channels=int(meta.get("base_channels", 16)),
        block_depth=int(meta.get("block_depth", 4)),
    )
    model = build_model(cfg)
    state = ckpt["state_dict"]
    # Checkpoints from train_raw_visual use RawDenoiserDenoiser (net.* keys).
    if not any(k.startswith("net.") for k in state) and hasattr(model, "net"):
        state = {f"net.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _panel(fused_rgb: np.ndarray, denoised_rgb: np.ndarray, gt_rgb: np.ndarray) -> np.ndarray:
    rows = [np.clip(x, 0, 1) for x in (fused_rgb, denoised_rgb, gt_rgb)]
    return np.concatenate(rows, axis=1)


def run(
    burst_dir: Path,
    pair_dir: Path | None,
    ckpt_path: Path,
    out_dir: Path,
    *,
    n_frames: int = FUSION_FRAMES,
    device_name: str = "auto",
) -> dict:
    burst_dir = burst_dir.resolve()
    ckpt_path = ckpt_path.resolve()
    if not burst_dir.is_dir():
        raise FileNotFoundError(f"Burst dir missing: {burst_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    dngs = sorted(burst_dir.glob("*.dng"))
    if len(dngs) < 8:
        raise FileNotFoundError(f"Only {len(dngs)} DNGs in {burst_dir}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    cfg = FusionConfig(n_frames=min(n_frames, len(dngs)), k_cap=16.0)
    frames = [load_packed(p) for p in dngs[: cfg.n_frames]]
    fused, weight = fuse_burst_packed(frames, cfg)
    x5 = stack_fusion_input(fused, weight, k_cap=cfg.k_cap)

    model, ckpt = _load_model(ckpt_path, device)
    model = model.to(device)
    with torch.no_grad():
        out = model(to_tensor(x5).to(device))
    denoised = to_image(out.cpu())

    gt_packed, gt_kind = _load_gt_rgb(pair_dir, dngs)
    if gt_packed.ndim == 3 and gt_packed.shape[-1] == 3:
        gt_rgb = _align_gt(gt_packed, fused)
    else:
        gt_rgb = packed_to_rgb(_align_gt(gt_packed, fused), DISPLAY_GAIN)

    fused_rgb = packed_to_rgb(fused, DISPLAY_GAIN)
    denoised_rgb = packed_to_rgb(denoised, DISPLAY_GAIN)

    # Burst-mean GT matches training; report alongside panel GT when they differ.
    burst_gt_rgb = packed_to_rgb(
        _align_gt(burst_clean(dngs, limit=GT_BURST_FRAMES), fused), DISPLAY_GAIN)

    h = min(*(a.shape[0] for a in (fused_rgb, denoised_rgb, gt_rgb, burst_gt_rgb)))
    w = min(*(a.shape[1] for a in (fused_rgb, denoised_rgb, gt_rgb, burst_gt_rgb)))
    fused_rgb = fused_rgb[:h, :w]
    denoised_rgb = denoised_rgb[:h, :w]
    gt_rgb = gt_rgb[:h, :w]
    burst_gt_rgb = burst_gt_rgb[:h, :w]

    metrics = {
        "burst_dir": str(burst_dir),
        "pair": str(pair_dir) if pair_dir else None,
        "checkpoint": str(ckpt_path),
        "n_frames": cfg.n_frames,
        "device": str(device),
        "gt_kind": gt_kind,
        "psnr_fused": round(psnr(fused_rgb, gt_rgb), 2),
        "psnr_denoised": round(psnr(denoised_rgb, gt_rgb), 2),
        "ssim_fused": round(ssim(fused_rgb, gt_rgb), 4),
        "ssim_denoised": round(ssim(denoised_rgb, gt_rgb), 4),
        "psnr_fused_burst_gt": round(psnr(fused_rgb, burst_gt_rgb), 2),
        "psnr_denoised_burst_gt": round(psnr(denoised_rgb, burst_gt_rgb), 2),
        "checkpoint_psnr_out": ckpt.get("psnr_out"),
        "checkpoint_hcg_sync_pct": ckpt.get("hcg_sync_pct"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    panel = _panel(fused_rgb, denoised_rgb, gt_rgb)
    png = out_dir / "hybrid_denoise_panel.png"
    Image.fromarray((panel * 255 + 0.5).astype(np.uint8)).save(png)
    metrics["panel"] = str(png)

    summary = out_dir / "metrics.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_json"] = str(summary)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 hybrid denoise demo (inference only)")
    ap.add_argument("--burst-dir", type=Path, default=DEFAULT_BURST,
                    help="LCG burst folder (default cabinet_D50_100/ag128)")
    ap.add_argument("--pair", type=Path, default=DEFAULT_PAIR,
                    help="PI_RAW pair for local GT (optional)")
    ap.add_argument("--no-pair", action="store_true",
                    help="skip local GT; use burst temporal mean only")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--frames", type=int, default=FUSION_FRAMES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="auto", help="cpu|cuda|auto")
    args = ap.parse_args()

    pair = None if args.no_pair else args.pair
    if pair is not None and not pair.is_dir():
        print(f"WARN: pair dir missing ({pair}) — using burst-mean GT", file=sys.stderr)
        pair = None

    try:
        metrics = run(
            args.burst_dir, pair, args.checkpoint, args.out,
            n_frames=args.frames, device_name=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2))
    print(f"\nPanel: {metrics['panel']}")
    print(f"vs {metrics['gt_kind']}:  Fused {metrics['psnr_fused']:.2f} dB → "
          f"Denoised {metrics['psnr_denoised']:.2f} dB  "
          f"(Δ {metrics['psnr_denoised'] - metrics['psnr_fused']:+.2f} dB)")
    if metrics.get("psnr_denoised_burst_gt") is not None:
        print(f"vs burst_mean GT:  Fused {metrics['psnr_fused_burst_gt']:.2f} dB → "
              f"Denoised {metrics['psnr_denoised_burst_gt']:.2f} dB  "
              f"(checkpoint ref {metrics.get('checkpoint_psnr_out', '?')} dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
