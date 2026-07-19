#!/usr/bin/env python3
"""Offline teacher ODE straighten dumps for Stage A distillation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsa.flow_matching import euler_sample
from nsa.inference import to_tensor
from train_cfm_distill import _load_teacher
from train_stream_to_gt import DEFAULT_GAINS, DEFAULT_SCENES, build_pairs


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "datasets/cfm_straighten_alpha_edm")
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-mode", choices=("mean", "alpha_trim"), default="alpha_trim")
    ap.add_argument("--gt-cache", type=Path,
                    default=ROOT / "datasets/imx662_project/gt_alpha16")
    ap.add_argument("--gt-frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--temporal", type=int, default=4)
    ap.add_argument("--target-samples", type=int, default=25000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--teacher-steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=662)
    args = ap.parse_args()

    if not args.teacher.is_file():
        print(f"Teacher missing: {args.teacher}", file=sys.stderr)
        return 1

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    gt_frames = args.gt_frames
    if args.gt_mode == "alpha_trim" and gt_frames == 512:
        gt_frames = 16

    dev = _device()
    teacher, tblob = _load_teacher(args.teacher, dev)
    tmeta = tblob.get("model", {})
    teacher_gain = bool(tmeta.get("gain_film", tblob.get("gain_film", False)))

    pairs, _, meta = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=gt_frames, stride=args.stride,
        holdout_start=args.holdout_start, temporal=temporal,
        gt_mode=args.gt_mode, gt_cache_root=args.gt_cache)
    if not pairs:
        print("No pairs — check bursts/", file=sys.stderr)
        return 1
    pair_gains = meta.get("pair_gains") or [128] * len(pairs)

    shards = args.out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    n_pairs = len(pairs)
    crops_per = max(1, int(np.ceil(args.target_samples / max(n_pairs, 1))))
    total_target = min(args.target_samples, n_pairs * crops_per)
    print(f"Dump {total_target} crops  ({n_pairs} pairs × up to {crops_per} crops)  "
          f"teacher_steps={args.teacher_steps}  device={dev}", flush=True)

    samples: list[dict] = []
    written = 0
    t0 = time.time()
    crop = args.crop
    batch = max(1, args.batch)

    for pi, ((cond_np, _), gain) in enumerate(zip(pairs, pair_gains)):
        cond_t = to_tensor(cond_np)
        h, w = cond_t.shape[-2], cond_t.shape[-1]
        c = min(crop, h, w)
        if c < 32:
            continue
        for _ in range(crops_per):
            if written >= total_target:
                break
            iy = int(torch.randint(0, h - c + 1, (1,), generator=g))
            ix = int(torch.randint(0, w - c + 1, (1,), generator=g))
            cond_crop = cond_t[..., iy:iy + c, ix:ix + c]
            gain_t = None
            if teacher_gain:
                gain_t = torch.tensor([float(gain)], device=dev, dtype=torch.float32)
            with torch.no_grad():
                out = euler_sample(
                    teacher, cond_crop.to(dev), gain=gain_t,
                    steps=args.teacher_steps)
            cond_hwc = cond_crop.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            teacher_hwc = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            rel = f"shards/{written:06d}.npz"
            path = args.out / rel
            np.savez_compressed(
                path,
                cond=cond_hwc.astype(np.float32),
                teacher=teacher_hwc.astype(np.float32),
                gain=np.int32(gain),
                pair_index=np.int32(pi),
            )
            samples.append({
                "file": rel, "gain": int(gain), "pair_index": pi,
                "crop": c, "y": int(iy), "x": int(ix),
            })
            written += 1
            if written % 200 == 0:
                rate = written / max(time.time() - t0, 1e-3)
                print(f"  {written}/{total_target}  {rate:.1f} samples/s", flush=True)
        if written >= total_target:
            break

    index = {
        "teacher": str(args.teacher),
        "teacher_steps": args.teacher_steps,
        "gt_mode": args.gt_mode,
        "temporal": temporal,
        "n_samples": written,
        "crop": crop,
        "samples": samples,
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {written} dumps → {args.out / 'index.json'}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
