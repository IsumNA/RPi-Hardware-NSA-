#!/usr/bin/env python3
"""Sort the synced HCG raw frames (.hcg_raw_cache/) into proper bursts/, then
build real noisy.dng + gt.tif pairs for the 3 recovered HCG scenes — the same
DNG+multi-frame-GT treatment already applied to LCG, using build_dng_pairs.py's
demosaic_mean.

Source: datasets/imx662_project/.hcg_raw_cache/<flat files>, sorted by
hcg_sort_manifest.json (scene -> ag_tag -> [filenames]), built from the Pi's
own project.json capture metadata (real gain/timestamp per frame — see
scratchpad/build_hcg_manifest.py). ag1 is not reachable in HCG mode on this
sensor (confirmed absent from every clean gain sweep), so only ag2..ag512
exist for HCG.
"""
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path("/home/isum.nanomi-arachchige/RPi-Hardware-NSA-")
CACHE = ROOT / "datasets/imx662_project/.hcg_raw_cache"
BURSTS = ROOT / "datasets/imx662_project/bursts"
PI_RAW = ROOT / "datasets/PI_RAW/Data"
MANIFEST = ROOT / "datasets/imx662_project/hcg_sort_manifest.json"

GT_FRAMES = 256
NOISY_PICK = 60  # HCG bursts are shorter (96-231 frames) than LCG's 512

from build_dng_pairs import demosaic_mean  # reuse the exact same GT logic


def sort_into_bursts(manifest):
    for scene, tags in manifest.items():
        for ag_tag, filenames in tags.items():
            dest = BURSTS / scene / ag_tag
            dest.mkdir(parents=True, exist_ok=True)
            moved = 0
            for fn in filenames:
                src = CACHE / fn
                if not src.exists():
                    continue
                dst = dest / fn
                if not dst.exists():
                    shutil.copyfile(src, dst)
                moved += 1
            print(f"{scene}/{ag_tag}: {moved} frames sorted into bursts/", flush=True)


def convert_one(scene, ag_tag):
    gain_num = int(ag_tag.replace("ag", ""))
    burst_dir = BURSTS / scene / ag_tag
    files = sorted(burst_dir.glob("*.dng"))
    if len(files) < 10:
        print(f"  {scene}/{ag_tag}: only {len(files)} frames, skipping")
        return None
    dest = PI_RAW / scene / f"imx662h_{ag_tag}_test"
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
    existing = {}
    if gj.exists():
        try:
            existing = json.loads(gj.read_text())
        except Exception:
            existing = {}
    existing.setdefault("requested_gain", gain_num)
    existing["hcg_enabled"] = True
    existing["source"] = "recovered_from_pi_ctt_cache"
    gj.write_text(json.dumps(existing, indent=2))
    return len(files)


def main():
    manifest = json.loads(MANIFEST.read_text())
    print("=== sorting into bursts/ ===")
    sort_into_bursts(manifest)
    print("\n=== building noisy.dng + gt.tif pairs ===")
    done = 0
    for scene, tags in manifest.items():
        for ag_tag in sorted(tags, key=lambda t: int(t.replace("ag", ""))):
            n = convert_one(scene, ag_tag)
            if n:
                print(f"{scene}/imx662h_{ag_tag}_test: {n} frames -> noisy.dng + gt.tif",
                     flush=True)
                done += 1
    print(f"\nconverted {done} HCG scene/gain folders to real DNG + multi-frame GT")


if __name__ == "__main__":
    main()
