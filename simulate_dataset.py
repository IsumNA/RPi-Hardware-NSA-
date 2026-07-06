#!/usr/bin/env python3
"""Simulate a PI_RAW training dataset from clean source images.

Turn a folder of clean pictures into denoise-hw / NAS layout::

    PI_RAW/Data/<scene>/<sensor>_ag<gain>_test/noisy.png
    PI_RAW/Data/<scene>/<sensor>_ag<gain>_test/gt.png

Examples
--------
  # One scene folder of clean PNGs → default output datasets/PI_RAW_sim
  python simulate_dataset.py --input ~/photos/cabinet_D50_100 --scene cabinet_D50_100

  # Parent folder with scene subdirs (cabinet_*, colour_strips, study, …)
  python simulate_dataset.py --input ~/captures/clean --output datasets/PI_RAW

  # denoise-hw filter-compatible folder name (imx662_ag12_test)
  python simulate_dataset.py --input ~/study --scene study --sensor imx662 \\
      --gain 256 --ag-tag ag12 --write-config

  # Full 5-phase workflow (calibrated IMX662 model):
  python calibrate_noise.py -i calibration/imx662_gain256 -o models/noise/imx662_gain256.json
  python simulate_dataset.py -i clean_scenes/ -o datasets/PI_RAW \\
      --calibration models/noise/imx662_gain256.json

  # Then train on the result
  python run_demo.py --real --dataset datasets/PI_RAW --extended-train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.config import GAINS, SENSOR_KEYS
from nsa.dataset_sim import build_dataset, discover_clean_images, plan_jobs
from nsa.denoise_hw_data import patch_config_dataset
from nsa.theme import banner, console, kv_table, log


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="folder of clean images (flat or scene subfolders)")
    p.add_argument("--output", "-o", default="datasets/PI_RAW",
                   help="PI_RAW root to write (default: datasets/PI_RAW)")
    p.add_argument("--sensor", choices=list(SENSOR_KEYS), default="imx662",
                   help="sensor noise profile to simulate")
    p.add_argument("--gain", type=int, choices=GAINS, default=256,
                   help="analog gain for the photon-transfer model")
    p.add_argument("--ag-tag",
                   help="folder tag for denoise-hw filters (e.g. ag12 → imx662_ag12_test). "
                        "Default: ag<GAIN> e.g. ag256")
    p.add_argument("--layout", choices=("auto", "flat", "scenes"), default="auto",
                   help="auto: subfolders=scenes; flat: one scene; scenes: only children")
    p.add_argument("--scene", default="scene_001",
                   help="scene name when --layout flat (folder under Data/)")
    p.add_argument("--temporal-frames", type=int, default=64,
                   help="independent reads averaged for gt (higher = cleaner GT)")
    p.add_argument("--noise-std", type=float,
                   help="override read-noise std in electrons RMS (denoise-hw style)")
    p.add_argument("--seed", type=int, default=662)
    p.add_argument("--max-side", type=int, default=0,
                   help="downscale so longest side <= N (0 = keep full resolution)")
    p.add_argument("--no-recursive", action="store_true",
                   help="only look at images directly inside each scene folder")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate pairs even if noisy.png/gt.png already exist")
    p.add_argument("--calibration", metavar="JSON",
                   help="Phase 3 noise model from calibrate_noise.py (5-phase synthesis)")
    p.add_argument("--dry-run", action="store_true",
                   help="show planned output paths without writing files")
    p.add_argument("--write-config", action="store_true",
                   help="point config.yaml at the output dataset + real_capture")
    args = p.parse_args()

    banner("Noise simulation  ·  build PI_RAW dataset")
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    try:
        if args.dry_run:
            jobs = plan_jobs(
                inp, out, sensor=args.sensor, gain=args.gain,
                layout=args.layout, scene=args.scene, ag_tag=args.ag_tag,
                recursive=not args.no_recursive,
            )
            log(f"Dry run — {len(jobs)} pair(s) would be written under {out}", "info")
            for j in jobs[:20]:
                rel = j.out_dir.relative_to(out) if j.out_dir.is_relative_to(out) else j.out_dir
                console.print(f"  [dim]·[/] {rel}  ←  {j.image.name}")
            if len(jobs) > 20:
                console.print(f"  [dim]… and {len(jobs) - 20} more[/]")
            return 0

        manifest = build_dataset(
            inp, out,
            sensor=args.sensor,
            gain=args.gain,
            layout=args.layout,
            scene=args.scene,
            ag_tag=args.ag_tag,
            temporal_frames=max(1, args.temporal_frames),
            seed=args.seed,
            noise_std=args.noise_std,
            max_side=max(0, args.max_side),
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            calibration=args.calibration,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        log(str(exc), "err")
        return 1

    console.print()
    console.print(kv_table([
        ("output", manifest["output"]),
        ("sensor", manifest["sensor"]),
        ("gain", str(manifest["gain"])),
        ("pairs written", str(manifest["pairs_written"])),
        ("scenes", ", ".join(manifest["scenes"][:8]) or "—"),
    ], title="Dataset ready"))

    log(f"Manifest → {out / 'simulation_manifest.json'}", "ok")
    console.print()
    console.print("[bold]Train on this dataset:[/]")
    console.print(f"  python run_demo.py --real --dataset {out} --sensor {args.sensor}")
    console.print(f"  python run_demo.py --real --dataset {out} --extended-train")

    if args.write_config:
        patch_config_dataset(ROOT / "config.yaml", out)
        log(f"Updated config.yaml → dataset_path: {out}", "ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
