#!/usr/bin/env python3
"""Rebuild PI_RAW paired folders from real DNG bursts instead of pre-rendered PNGs.

For every scene/gain burst we have:
  * noisy.dng = one real raw frame, copied AS-IS (the app already decodes .dng
    via rawpy in nsa/raw_io.py's _load_any).
  * gt.tif    = temporal-average ground truth (mean of the burst, demosaiced the
    same way as a single DNG so it's pixel-compatible), written 16-bit lossless
    so no precision is lost versus the old 8-bit gt.png.
  * gain.json = copied over unchanged (actual_gain used by validate_gain sort).

This only touches scenes/gains that have a matching real burst; folders with no
burst (e.g. HCG-only imx662h_* or scenes never captured to burst) are left as-is.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rawpy
import cv2

ROOT = Path("/home/isum.nanomi-arachchige/RPi-Hardware-NSA-")
BURSTS = ROOT / "datasets/imx662_project/bursts"
PI_RAW = ROOT / "datasets/PI_RAW/Data"   # repo-local copy — NOT the shared /opt/datasets

GT_FRAMES = 256          # frames averaged for ground truth
NOISY_PICK = 300         # index of the single frame used as noisy.dng (outside GT window)


def demosaic_mean(files, limit):
    """Temporal-average raw Bayer over `files[:limit]`, then demosaic once."""
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
    rgb16 = cv2.cvtColor(b16, cv2.COLOR_BAYER_RG2RGB)
    return rgb16  # uint16 RGB, demosaiced, linear (no gamma) — matches _load_any's dng path


def convert_one(scene, gain_label, gain_num, sensor_prefix):
    burst_dir = BURSTS / scene / f"ag{gain_num}"
    files = sorted(burst_dir.glob("*.dng"))
    if len(files) < 10:
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
    scenes = [d.name for d in BURSTS.iterdir() if d.is_dir()]
    gains = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    done = 0
    for scene in sorted(scenes):
        for g in gains:
            n = convert_one(scene, f"ag{g}", g, "imx662")
            if n:
                print(f"{scene}/ag{g}: {n} frames -> noisy.dng + gt.tif", flush=True)
                done += 1
    print(f"\nconverted {done} scene/gain folders to real DNG noisy + multi-frame GT")


if __name__ == "__main__":
    main()
