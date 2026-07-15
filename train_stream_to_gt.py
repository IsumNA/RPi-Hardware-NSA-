#!/usr/bin/env python3
"""Train: live noisy frame → burst-averaged GT. Deploy that model on the Pi.

This is NOT "average 500 frames at inference". GT was already built from your
bursts. Training teaches a network to map a *single stream frame* (or a short
live window) toward that GT so the Pi can denoise in real time.

Run on AI GPU::

  .venv/bin/python -u train_stream_to_gt.py --gains 256,512 --steps 8000
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.inference import build_loss, psnr, ssim, to_image, to_tensor
from nsa.raw_domain import RawDenoiser, burst_clean, load_packed, packed_to_rgb

DISPLAY_GAIN = 8.0
DEFAULT_SCENES = (
    "cabinet_H_2", "cabinet_H_10", "cabinet_F11_25",
    "cabinet_D50_100", "cabinet_D_10", "cabinet_F_5",
)
DEFAULT_GAINS = (256, 512)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pairs(
    bursts_root: Path,
    scenes: tuple[str, ...],
    gains: tuple[int, ...],
    *,
    gt_frames: int,
    stride: int,
    holdout_start: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict], dict]:
    """Each individual DNG → packed GT (temporal mean of first gt_frames)."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    evals: list[dict] = []
    stats: list[dict] = []

    for scene in scenes:
        for g in gains:
            bdir = bursts_root / scene / f"ag{g}"
            files = sorted(bdir.glob("*.dng"))
            if len(files) < 32:
                continue
            n = len(files)
            n_gt = min(gt_frames, n)
            gt = burst_clean(files, limit=n_gt)
            train_idx = list(range(0, n_gt, stride))
            for i in train_idx:
                pairs.append((load_packed(files[i]), gt))
            # Held-out frames outside the GT window when possible
            hold = [i for i in range(holdout_start, n, max(1, (n - holdout_start) // 3))
                    if i < n][:3]
            if not hold:
                hold = [n_gt - 1]
            evals.append({
                "scene": scene, "gain": g,
                "gt": gt,
                "noisy": [(i, load_packed(files[i])) for i in hold],
            })
            stats.append({
                "scene": scene, "gain": g, "frames": n,
                "train": len(train_idx), "holdout": hold,
            })
            print(f"  {scene}/ag{g}: {n} DNGs → {len(train_idx)} train, "
                  f"holdout {hold}", flush=True)
    return pairs, evals, {"scenes": stats, "total_pairs": len(pairs)}


def _train(
    model: nn.Module,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    loss_fn,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
) -> nn.Module:
    from nsa.inference import _sample_batch

    tensors = [(to_tensor(n), to_tensor(c)) for n, c in pairs]
    # Emphasise darker / higher-noise samples via mean intensity
    wts = torch.tensor(
        [1.0 / max(float(n.mean()), 1e-3) for n, _ in pairs], dtype=torch.float32)
    wts = (wts / wts.mean()).clamp(0.5, 8.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 10)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(662)
    model = model.to(device)
    model.train()
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for i in range(steps):
        xb, yb = _sample_batch(tensors, crop, batch, g, weights=wts)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % 50 == 0 or i == steps - 1:
            print(f"  step {i+1}/{steps}  loss={loss.item():.4f}  "
                  f"{(time.time()-t0)/max(i,1):.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            _save_eval_panel(model, evals[0], device, panel_dir, i + 1)
    model.eval()
    return model


@torch.no_grad()
def _save_eval_panel(model, ev: dict, device, panel_dir: Path, step: int) -> None:
    idx, noisy = ev["noisy"][0]
    gt = ev["gt"]
    out = to_image(model(to_tensor(noisy).to(device)).cpu())
    nr, gr, or_ = (_rgb(noisy), _rgb(gt), _rgb(out))
    pin, pout = psnr(nr, gr), psnr(or_, gr)
    strip = np.concatenate([nr, or_, gr], axis=1)
    img = (np.clip(strip, 0, 1) * 255 + 0.5).astype(np.uint8)
    path = panel_dir / f"step_{step:05d}.png"
    Image.fromarray(img).save(path)
    latest = panel_dir / "latest.png"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        shutil.copy2(path, latest)
    print(f"  panel step {step}: {ev['scene']}/ag{ev['gain']} "
          f"frame {idx}  {pin:.1f}→{pout:.1f} dB", flush=True)


def _rgb(pk: np.ndarray) -> np.ndarray:
    return packed_to_rgb(pk, DISPLAY_GAIN)


@torch.no_grad()
def evaluate(model, evals, device) -> list[dict]:
    rows = []
    model.eval()
    for ev in evals:
        gt = ev["gt"]
        gr = _rgb(gt)
        for idx, noisy in ev["noisy"]:
            out = to_image(model(to_tensor(noisy).to(device)).cpu())
            nr, or_ = _rgb(noisy), _rgb(out)
            rows.append({
                "scene": ev["scene"], "gain": ev["gain"], "frame": idx,
                "psnr_in": round(psnr(nr, gr), 2),
                "psnr_out": round(psnr(or_, gr), 2),
                "ssim_in": round(ssim(nr, gr), 4),
                "ssim_out": round(ssim(or_, gr), 4),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--panel-every", type=int, default=200)
    ap.add_argument("--panel-dir", type=Path, default=ROOT / "outputs/stream_gt_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    dev = _device()
    print(f"Device {dev}  recipe: STREAM FRAME → {args.gt_frames}-frame GT", flush=True)
    print(f"Scenes {scenes}  gains {gains}", flush=True)

    pairs, evals, meta = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=args.gt_frames, stride=args.stride,
        holdout_start=args.holdout_start)
    if not pairs:
        print("No training pairs — check bursts/", file=sys.stderr)
        return 1
    print(f"Total train pairs: {len(pairs)}", flush=True)

    # Single packed frame in → packed clean out (what the Pi stream will call)
    model = RawDenoiser(
        base_channels=args.channels, block_depth=args.depth, in_ch=4, out_ch=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RawDenoiser 4→4  {n_params:,} params  "
          f"({args.channels}ch × {args.depth})", flush=True)

    # Packed 4ch RAW: LPIPS perceptual expects RGB and fails on 4ch.
    # Use charbonnier+swt (A/B winner in config.yaml for this domain).
    loss_fn = build_loss(
        "charbonnier+swt",
        charbonnier_eps=5e-4,
        weights={"charbonnier": 1.0, "swt": 0.15},
    )
    try:
        loss_fn(torch.zeros(1, 4, 32, 32), torch.zeros(1, 4, 32, 32))
        print("Loss: charbonnier+swt", flush=True)
    except Exception as e:
        print(f"swt probe failed ({e}); falling back to charbonnier+edge",
              flush=True)
        loss_fn = build_loss(
            "charbonnier+edge",
            charbonnier_eps=5e-4,
            weights={"charbonnier": 1.0, "edge": 0.15},
        )
        loss_fn(torch.zeros(1, 4, 32, 32), torch.zeros(1, 4, 32, 32))
        print("Loss: charbonnier+edge", flush=True)

    model = _train(
        model, pairs, args.steps, crop=args.crop, batch=args.batch, lr=2e-3,
        loss_fn=loss_fn, device=dev, panel_every=args.panel_every,
        panel_dir=args.panel_dir, evals=evals)

    rows = evaluate(model, evals, dev)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    print(f"Held-out mean PSNR {mean_in:.2f} → {mean_out:.2f} dB "
          f"(Δ{mean_out-mean_in:+.2f})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "stream_to_gt.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": {
            "family": "raw_denoiser_stream",
            "base_channels": args.channels,
            "block_depth": args.depth,
            "in_ch": 4, "out_ch": 4,
        },
        "recipe": "stream_frame_to_burst_gt",
        "gt_frames": args.gt_frames,
        "gains": list(gains),
        "scenes": list(scenes),
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "eval": rows,
    }, ckpt)
    (args.out / "stream_to_gt_summary.json").write_text(
        json.dumps({"psnr_in": mean_in, "psnr_out": mean_out, "eval": rows,
                    "params": n_params, "pairs": len(pairs), **meta}, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)

    # Final panel strip for first eval scene
    if evals:
        _save_eval_panel(model, evals[0], dev, args.panel_dir, args.steps)
        shutil.copy2(args.panel_dir / f"step_{args.steps:05d}.png",
                     args.out / "stream_to_gt_panel.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
