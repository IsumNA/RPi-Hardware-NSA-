#!/usr/bin/env python3
"""Train a denoiser on CONSISTENT pairs: noisy = single burst frame,
GT = temporal mean of the burst, both via the identical _load_any pipeline.

This fixes the colour/exposure/WB mismatch of the PI_RAW gt.tif targets that was
forcing every model to blur. Trains on GPU via the (patched) calibrate_multi.

  .venv/bin/python train_burst.py --gains 128 256 512 --frames 48 --noisy-per 3 \
     --base-channels 32 --enc 1 2 2 --mid 4 --dec 2 2 1 \
     --loss l1+perceptual+edge+swtrel+ffl --steps 9000 \
     --out outputs/iter/burst32.pt --holdout-npz outputs/iter/burst32_holdout.npz
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsa.config import ModelConfig
from nsa.models import build_model, count_params
from nsa.inference import build_loss, calibrate_multi
from nsa.raw_io import _load_any

BURST_ROOT = Path("datasets/imx662_project/bursts")


def burst_dirs(gains):
    out = []
    for scene in sorted(BURST_ROOT.iterdir()):
        if not scene.is_dir():
            continue
        for g in gains:
            d = scene / f"ag{g}"
            n = len(glob.glob(str(d / "*.dng")))
            if n >= 8:
                out.append((scene.name, g, d, n))
    return out


def build_pair(d: Path, frames: int, noisy_per: int, seed: int):
    files = sorted(glob.glob(str(d / "*.dng")))
    rng = np.random.default_rng(seed)
    k = min(frames, len(files))
    idx = rng.choice(len(files), k, replace=False)
    acc = None
    loaded = {}
    for i in idx:
        a = _load_any(Path(files[i]))
        loaded[i] = a
        acc = a.astype(np.float64) if acc is None else acc + a
    gt = (acc / k).astype(np.float32)
    npick = min(noisy_per, len(idx))
    picks = rng.choice(idx, npick, replace=False)
    return [(loaded[i], gt) for i in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--frames", type=int, default=48, help="frames averaged for GT")
    ap.add_argument("--noisy-per", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--enc", type=int, nargs="+", default=[1, 2, 2])
    ap.add_argument("--mid", type=int, default=4)
    ap.add_argument("--dec", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--block-depth", type=int, default=4)
    ap.add_argument("--loss", default="l1+perceptual+edge+swtrel+ffl")
    ap.add_argument("--loss-weight", action="append", default=[], help="term=val")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-scenes", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-npz", required=True)
    ap.add_argument("--seed", type=int, default=662)
    ap.add_argument("--init", default=None, help="warm-start state_dict")
    args = ap.parse_args()

    dirs = burst_dirs(args.gains)
    if not dirs:
        print("NO BURSTS FOUND", file=sys.stderr); sys.exit(2)
    # hold out the last N distinct scenes for eval
    scenes = sorted({s for s, *_ in dirs})
    hold = set(scenes[-args.holdout_scenes:]) if args.holdout_scenes else set()
    train_dirs = [t for t in dirs if t[0] not in hold]
    hold_dirs = [t for t in dirs if t[0] in hold]
    print(f"{len(dirs)} bursts | train {len(train_dirs)} | holdout {len(hold_dirs)} "
          f"(scenes {sorted(hold)})", flush=True)

    print("Building consistent train pairs (noisy frame, burst mean)...", flush=True)
    pairs = []
    for i, (s, g, d, n) in enumerate(train_dirs):
        pairs += build_pair(d, args.frames, args.noisy_per, args.seed + i)
        print(f"  [{i+1}/{len(train_dirs)}] {s}/ag{g} ({n} frames) -> {len(pairs)} pairs",
              flush=True)
    print(f"total train pairs: {len(pairs)}", flush=True)

    weights = {}
    for w in args.loss_weight:
        k, v = w.split("=")
        weights[k] = float(v)
    loss_fn = build_loss(args.loss, weights=weights or None)

    cfg = ModelConfig(model_family="nafnet", base_channels=args.base_channels,
                      block_depth=args.block_depth, conv_type="depthwise",
                      activation="relu", nafnet_enc_blocks=args.enc,
                      nafnet_middle_blocks=args.mid, nafnet_dec_blocks=args.dec)
    model = build_model(cfg)
    if args.init and Path(args.init).is_file():
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        try:
            model.load_state_dict(sd, strict=True); print("warm-started from", args.init)
        except Exception as e:
            print("warm-start skipped:", e)
    print(f"model: nafnet base{args.base_channels} enc{args.enc} mid{args.mid} "
          f"dec{args.dec} -> {count_params(model)/1000:.0f}K params", flush=True)

    last = [0.0]
    def prog(i, total, loss):
        if i % 200 == 0 or i == total:
            print(f"  step {i}/{total}  loss={loss:.4f}", flush=True)
        last[0] = loss
    calibrate_multi(model, pairs, args.steps, args.seed, prog,
                    crop=args.crop, batch=args.batch, lr=args.lr, loss_fn=loss_fn)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model": {"family": "nafnet", "base_channels": args.base_channels,
                          "block_depth": args.block_depth, "conv_type": "depthwise",
                          "activation": "relu", "nafnet_enc": args.enc,
                          "nafnet_middle": args.mid, "nafnet_dec": args.dec},
                "sensor": "imx662", "gain": max(args.gains),
                "train": {"loss": args.loss, "weights": weights, "steps": args.steps,
                          "frames": args.frames, "pairs": len(pairs)}}, args.out)
    print("SAVED", args.out, flush=True)

    # holdout pairs for visual eval (against the CORRECT burst-mean GT)
    hd = hold_dirs or train_dirs[-2:]
    npz = {}
    for j, (s, g, d, n) in enumerate(hd[:4]):
        pr = build_pair(d, args.frames, 1, args.seed + 9999 + j)[0]
        npz[f"noisy_{j}"] = pr[0]; npz[f"gt_{j}"] = pr[1]
        npz[f"name_{j}"] = f"{s}/ag{g}"
    np.savez_compressed(args.holdout_npz, **npz)
    print("SAVED holdout", args.holdout_npz, "n=", len(hd[:4]), flush=True)


if __name__ == "__main__":
    main()
