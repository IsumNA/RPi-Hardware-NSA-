#!/usr/bin/env python3
"""Build a CLEAN denoise-hw-style dataset: noisy = one burst frame,
gt.tif = burst temporal mean, both via the identical _load_any pipeline so the
pair differs only by noise (fixes the mismatched PI_RAW gt.tif targets).

Writes datasets/PI_RAW_clean/Data/<scene>/imx662_ag<N>_test/{noisy.tif, gt.tif}
as 16-bit linear RGB — the format run_demo's loader reads back consistently.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsa.raw_io import _load_any

BURST = Path("datasets/imx662_project/bursts")
OUT = Path("datasets/PI_RAW_clean/Data")
GAINS = [128, 256, 512]
FRAMES = 48

def save16(path: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)

n = 0
for scene in sorted(BURST.iterdir()):
    if not scene.is_dir():
        continue
    for g in GAINS:
        files = sorted(glob.glob(str(scene / f"ag{g}" / "*.dng")))
        if len(files) < 8:
            continue
        k = min(FRAMES, len(files))
        rng = np.random.default_rng(1000 + g)
        idx = rng.choice(len(files), k, replace=False)
        acc = None; first = None
        for j, i in enumerate(idx):
            a = _load_any(Path(files[i]))
            if first is None:
                first = a
            acc = a.astype(np.float64) if acc is None else acc + a
        gt = (acc / k).astype(np.float32)
        d = OUT / scene.name / f"imx662_ag{g}_test"
        save16(d / "noisy.tif", first)
        save16(d / "gt.tif", gt)
        n += 1
        print(f"  {scene.name}/ag{g}: noisy+gt ({k}-frame mean) -> {d}", flush=True)
print(f"DONE: {n} clean paired folders under {OUT}", flush=True)
