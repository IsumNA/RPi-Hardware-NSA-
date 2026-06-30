#!/usr/bin/env python3
"""NSA Patch-Cache Builder
=========================
Pre-scans a dataset into detail-scored training patches (the denoise-hw
``dataset.py`` idea) so that full training runs can sample from a cache of the
sharpest, most informative crops instead of re-decoding RAWs every step.

For every frame found under ``--dataset`` it extracts the top ``--per-image``
high-detail (Laplacian-variance) crops, builds a matching ``(noisy, clean)``
pair, and writes them to ``outputs/patch_cache/`` as compressed ``.npz`` files
plus an ``index.json`` manifest.

Ground truth per crop:
  * paired ``noisy``/``gt`` folder  -> real gt crop (denoise-hw convention)
  * loose frame + ``--simulate-noise`` -> file treated as clean, sensor noise added
  * loose frame                      -> NL-means/edge-preserving reference

Examples
--------
  python cache.py --dataset ./datasets/imx219_raws --per-image 8
  python cache.py --dataset ./data --filter imx219 ag12 --patch 128 --limit 20
  python cache.py --sensor imxng --simulate-noise --dataset ./clean_scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from nsa.raw_io import (_classical_reference, _detail_score, _load_any,
                        _synth_noisy_gt, list_frames)
from nsa.sensors import get_sensor
from nsa.theme import banner, console, log


def _fit_for_cache(img: np.ndarray, patch: int) -> np.ndarray:
    """Keep the full frame (so many crops fit); only upscale if smaller than patch."""
    h, w = img.shape[:2]
    if min(h, w) < patch:
        s = patch / float(min(h, w))
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_LINEAR)
    return img


def _top_windows(img: np.ndarray, patch: int, k: int) -> list[tuple[int, int, float]]:
    """Return up to k high-detail, mostly non-overlapping (y, x, score) windows."""
    h, w = img.shape[:2]
    if h < patch or w < patch:
        return [(0, 0, 0.0)]
    gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    step = max(1, patch // 2)
    cands = []
    for y0 in range(0, h - patch + 1, step):
        for x0 in range(0, w - patch + 1, step):
            cands.append((y0, x0, _detail_score(gray[y0:y0 + patch, x0:x0 + patch])))
    cands.sort(key=lambda t: t[2], reverse=True)

    picked: list[tuple[int, int, float]] = []
    for y0, x0, s in cands:
        if all(abs(y0 - py) >= step or abs(x0 - px) >= step for py, px, _ in picked):
            picked.append((y0, x0, s))
        if len(picked) >= k:
            break
    return picked or [(0, 0, 0.0)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cache.py",
        description="NSA patch-cache builder — detail-scored crops for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dataset", dest="dataset_path", required=True,
                   help="folder or file of captures (paired noisy/gt supported)")
    p.add_argument("--filter", nargs="*", default=[],
                   help="keyword filter for dataset folders (e.g. imx219 ag12)")
    p.add_argument("--patch", type=int, default=128, help="patch size (default 128)")
    p.add_argument("--per-image", dest="per_image", type=int, default=6,
                   help="high-detail crops per frame (default 6)")
    p.add_argument("--limit", type=int, default=0,
                   help="max number of frames to scan (0 = all)")
    p.add_argument("--simulate-noise", dest="simulate_noise", action="store_true",
                   help="treat loose frames as clean and inject sensor noise")
    p.add_argument("--sensor", choices=["imx219", "imx662", "imxng"], default="imx662",
                   help="sensor profile for simulated noise (default imx662)")
    p.add_argument("--gain", type=int, choices=[256, 512], default=512)
    p.add_argument("--out", default="outputs/patch_cache", help="output cache directory")
    p.add_argument("--seed", type=int, default=662)
    return p


def main() -> int:
    args = build_parser().parse_args()
    banner("NSA Patch-Cache Builder")

    sensor = get_sensor(args.sensor)
    sources = list_frames(args.dataset_path, args.filter or [], limit=args.limit)
    if not sources:
        log(f"No usable frames found under {args.dataset_path!r}", "warn")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Scanning {len(sources)} frame(s) -> {out}  "
        f"(patch {args.patch}, up to {args.per_image}/frame)", "step")

    index = []
    n_patches = 0
    for fi, src in enumerate(sources):
        name = src.get("name", str(src["noisy"]))
        gt_path = src.get("gt")
        try:
            noisy_full = _fit_for_cache(_load_any(Path(src["noisy"])), args.patch)
            gt_full = (_fit_for_cache(_load_any(Path(gt_path)), args.patch)
                       if gt_path else None)
        except Exception as exc:
            log(f"Skipped {name}: {exc}", "warn")
            continue

        for (y0, x0, score) in _top_windows(noisy_full, args.patch, args.per_image):
            p = args.patch
            ncrop = noisy_full[y0:y0 + p, x0:x0 + p]
            if gt_full is not None:                      # real paired gt
                ccrop = gt_full[y0:y0 + p, x0:x0 + p]
                kind = "paired"
            elif args.simulate_noise:                    # clean source -> simulate
                ncrop, ccrop = _synth_noisy_gt(ncrop, args.gain, sensor, 64,
                                               args.seed + n_patches)
                kind = "clean+sim"
            else:                                        # derive a reference
                ccrop = _classical_reference(ncrop)
                kind = "reference"

            fname = f"patch_{n_patches:05d}.npz"
            np.savez_compressed(out / fname,
                                noisy=ncrop.astype(np.float32),
                                clean=ccrop.astype(np.float32))
            index.append({"file": fname, "source": name, "kind": kind,
                          "detail": round(float(score), 5),
                          "xy": [int(x0), int(y0)]})
            n_patches += 1

        if (fi + 1) % 10 == 0 or fi == len(sources) - 1:
            log(f"  {fi + 1}/{len(sources)} frames  ·  {n_patches} patches", "info")

    manifest = {
        "dataset": args.dataset_path,
        "filter": args.filter or [],
        "patch": args.patch,
        "per_image": args.per_image,
        "sensor": args.sensor,
        "gain": args.gain,
        "simulate_noise": args.simulate_noise,
        "n_frames": len(sources),
        "n_patches": n_patches,
        "patches": index,
    }
    (out / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Cache built: {n_patches} patches from {len(sources)} frame(s) -> "
        f"{out/'index.json'}", "ok")
    console.print(f"\n  Use it for a full training run by sampling pairs from "
                  f"[bold]{out}[/].")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
