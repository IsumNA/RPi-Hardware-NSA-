"""Fix: cabinet_H_2 is shared between LCG and HCG bursts. The HCG sort step
dumped imx662_5000k_5l_*.dng frames into the SAME bursts/cabinet_H_2/ag<N>/
folders that already held the real LCG burst_*.dng frames, contaminating the
HCG noisy.dng/gt.tif. Split them into a dedicated bursts/cabinet_H_2_hcg/
tree and rebuild HCG pairs from the clean set only.
"""
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path("/home/isum.nanomi-arachchige/RPi-Hardware-NSA-")
BURSTS = ROOT / "datasets/imx662_project/bursts"
PI_RAW = ROOT / "datasets/PI_RAW/Data"
GT_FRAMES = 256
NOISY_PICK = 60

from build_dng_pairs import demosaic_mean

GAINS = [2, 4, 8, 16, 32, 64, 128, 256, 512]

for g in GAINS:
    mixed = BURSTS / "cabinet_H_2" / f"ag{g}"
    clean_hcg = BURSTS / "cabinet_H_2_hcg" / f"ag{g}"
    clean_hcg.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in list(mixed.glob("imx662_5000k_*.dng")):
        dst = clean_hcg / f.name
        if not dst.exists():
            shutil.move(str(f), str(dst))
        else:
            f.unlink()
        moved += 1
    remaining_lcg = len(list(mixed.glob("*.dng")))
    print(f"ag{g}: moved {moved} HCG frames out -> cabinet_H_2_hcg/ag{g} "
         f"({len(list(clean_hcg.glob('*.dng')))} total); "
         f"{remaining_lcg} genuine LCG frames remain in cabinet_H_2/ag{g}")

print("\n=== rebuilding cabinet_H_2 HCG pairs from the CLEAN set ===")
for g in GAINS:
    files = sorted((BURSTS / "cabinet_H_2_hcg" / f"ag{g}").glob("*.dng"))
    if len(files) < 10:
        print(f"ag{g}: only {len(files)} frames, skip")
        continue
    dest = PI_RAW / "cabinet_H_2" / f"imx662h_ag{g}_test"
    dest.mkdir(parents=True, exist_ok=True)
    noisy_src = files[min(NOISY_PICK, len(files) - 1)]
    shutil.copyfile(noisy_src, dest / "noisy.dng")
    rgb16 = demosaic_mean(files, min(GT_FRAMES, len(files)))
    cv2.imwrite(str(dest / "gt.tif"), cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
    gj = dest / "gain.json"
    existing = json.loads(gj.read_text()) if gj.exists() else {}
    existing.update({"requested_gain": g, "hcg_enabled": True,
                     "source": "recovered_from_pi_ctt_cache_FIXED"})
    gj.write_text(json.dumps(existing, indent=2))
    print(f"cabinet_H_2/imx662h_ag{g}_test: {len(files)} clean HCG frames -> noisy.dng + gt.tif")

print("\ndone")
