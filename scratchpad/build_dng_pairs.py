#!/usr/bin/env python3
"""Rebuild PI_RAW pairs from real DNG bursts (noisy.dng + ~100-frame GT).

Prefer the production trainer ``train_gt_match.py`` which consumes bursts
directly. This script only rebuilds the denoise-hw-style PI_RAW folders for
the older RGB pipeline / GUI.

GT uses the first ``GT_FRAMES`` Bayer frames (default 100), demosaiced once.
``noisy.dng`` is a held-out frame outside that window.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rawpy
import cv2

ROOT = Path(__file__).resolve().parents[1]
BURSTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/imx662_project/bursts"
PI_RAW = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "datasets/PI_RAW/Data"

GT_FRAMES = 100
NOISY_PICK = 120  # outside the GT window when the burst is long enough


def demosaic_mean(files, limit):
    acc = None
    n = 0
    black = white = None
    for f in files[:limit]:
        with rawpy.imread(str(f)) as r:
            raw = r.raw_image_visible.astype(np.float32)
            if black is None:
                black = float(np.mean(r.black_level_per_channel))
                white = float(r.white_level)
        acc = raw if acc is None else acc + raw
        n += 1
    mean_bayer = acc / max(n, 1)
    norm = np.clip((mean_bayer - black) / max(white - black, 1.0), 0.0, 1.0)
    b16 = (norm * 65535.0).astype(np.uint16)
    return cv2.cvtColor(b16, cv2.COLOR_BAYER_RG2RGB)


def convert_one(scene, gain_num, sensor_prefix):
    burst_dir = BURSTS / scene / f"ag{gain_num}"
    files = sorted(burst_dir.glob("*.dng"))
    if len(files) < 16:
        return None
    dest = PI_RAW / scene / f"{sensor_prefix}_ag{gain_num}_test"
    dest.mkdir(parents=True, exist_ok=True)

    noisy_src = files[min(NOISY_PICK, len(files) - 1)]
    shutil.copyfile(noisy_src, dest / "noisy.dng")

    rgb16 = demosaic_mean(files, min(GT_FRAMES, len(files)))
    cv2.imwrite(str(dest / "gt.tif"), cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))

    for stale in ("noisy.png", "gt.png"):
        p = dest / stale
        if p.exists():
            p.unlink()

    gj = dest / "gain.json"
    if not gj.exists():
        gj.write_text(json.dumps({"requested_gain": gain_num}, indent=2))
    return len(files)


def main():
    if not BURSTS.is_dir():
        print(f"bursts root not found: {BURSTS}", file=sys.stderr)
        sys.exit(1)
    scenes = [d.name for d in BURSTS.iterdir() if d.is_dir()]
    gains = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    done = 0
    for scene in sorted(scenes):
        for g in gains:
            for prefix in ("imx662", "imx662h"):
                # only write the prefix that matches available data; both map
                # to the same agN burst folder when HCG shares the capture.
                n = convert_one(scene, g, prefix)
                if n and prefix == "imx662":
                    print(f"{scene}/ag{g}: {n} frames -> noisy.dng + gt.tif "
                          f"(GT={min(GT_FRAMES, n)} frames)", flush=True)
                    done += 1
    print(f"\nconverted {done} scene/gain folders (GT_FRAMES={GT_FRAMES})")
    print("For sharp no-blur training prefer: python train_gt_match.py --bursts ...")


if __name__ == "__main__":
    main()
