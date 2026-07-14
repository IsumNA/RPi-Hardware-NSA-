#!/usr/bin/env python3
"""IMX662 noise calibration — Phases 1–4.

Phase 1 (capture): organise calibration frames under one folder::

    calibration/imx662_gain256/
      bias/           lens capped, minimal exposure (≥2 frames)
      dark/           lens capped, normal exposure (≥1 frame)
      flat/
        level_01/     uniform light pair at brightness 1
        level_02/     … 10–15 levels recommended
        …

Repeat for every gain/ISO and temperature you need.

Phases 2–4 run automatically: extract samples → fit {a, read_dist, row_dist,
adc_bits} → validate on held-out frames → write JSON model.

Phase 5 (synthesis) is ``simulate_dataset.py --calibration <model.json>``.

Examples
--------
  python calibrate_noise.py --input calibration/imx662_gain256 \\
      --output models/noise/imx662_gain256.json

  python simulate_dataset.py --input clean_scenes/ --calibration models/noise/imx662_gain256.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.noise_calib import run_calibration_pipeline
from nsa.noise_calib.io import discover_phase1_root
from nsa.theme import banner, console, kv_table, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i",
                   help="Phase-1 calibration folder (bias/, dark/, flat/)")
    p.add_argument("--from-pairs", dest="from_pairs", metavar="DATASET",
                   help="fit the model from real noisy/gt PAIRS in this dataset "
                        "(no bias/dark/flat needed) — one per gain, e.g. "
                        "--from-pairs datasets/PI_RAW --sensor imx662h --gain 512")
    p.add_argument("--output", "-o", default="models/noise/imx662.json",
                   help="JSON noise model output path")
    p.add_argument("--sensor", default="imx662")
    p.add_argument("--gain", type=int, default=256)
    p.add_argument("--temperature", type=float, dest="temperature_c",
                   help="capture temperature °C (metadata only)")
    p.add_argument("--no-holdout", action="store_true",
                   help="use all frames for fitting (skip Phase-4 validation)")
    p.add_argument("--list", action="store_true",
                   help="list discovered calibration frames and exit")
    p.add_argument("--seed", type=int, default=662)
    args = p.parse_args()

    banner("Noise calibration  ·  Phases 1–4")

    # From-pairs mode: fit directly from real noisy/gt captures (per gain).
    if args.from_pairs:
        from nsa.noise_calib.from_pairs import run_pair_calibration
        log(f"Fitting from real pairs: {args.from_pairs}  "
            f"[{args.sensor} @ gain {args.gain}]", "step")
        model, validation = run_pair_calibration(
            args.from_pairs, args.output, sensor=args.sensor, gain=args.gain,
            filter_tokens=[args.sensor], seed=args.seed)
        log(f"Fitted from {validation['n_pairs']} pair(s): "
            f"shot_a={model.shot_a:.4g}  read σ={model.read_dist.sigma:.4g}", "ok")
        log(f"Model → {args.output}", "ok")
        if validation.get("report_png"):
            log(f"Visual report → {validation['report_png']}", "ok")
        return 0

    if not args.input:
        log("Provide --input (bias/dark/flat folder) or --from-pairs DATASET", "err")
        return 1
    root = Path(args.input).expanduser().resolve()

    try:
        discovered = discover_phase1_root(root)
    except FileNotFoundError as exc:
        log(str(exc), "err")
        return 1

    if args.list:
        console.print(kv_table([
            ("bias frames", str(len(discovered["bias"]))),
            ("dark frames", str(len(discovered["dark"]))),
            ("flat levels", str(len(discovered["flat_pairs"]))),
        ], title="Phase 1 captures"))
        return 0

    try:
        model, validation = run_calibration_pipeline(
            root, args.output,
            sensor=args.sensor,
            gain=args.gain,
            temperature_c=args.temperature_c,
            holdout=not args.no_holdout,
            seed=args.seed,
        )
    except ValueError as exc:
        log(str(exc), "err")
        return 1

    console.print()
    console.print(kv_table([
        ("shot a (Poisson)", f"{model.shot_a:.6g}"),
        ("read noise", f"{model.read_dist.kind} σ={model.read_dist.sigma:.4g}"),
        ("row noise", (f"{model.row_dist.kind} ×{model.row_strength:.3f}"
                       if model.row_dist else "none")),
        ("ADC bits", str(model.adc_bits)),
        ("quant scale", f"{model.quant_scale:.6g}"),
        ("validation", "PASS" if validation.get("ok") else "CHECK"),
    ], title="Phase 3 model"))

    val_path = Path(args.output).with_suffix(".validation.json")
    val_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    log(f"Model → {args.output}", "ok")
    log(f"Validation report → {val_path}", "ok")
    if validation.get("report_png"):
        log(f"Visual report → {validation['report_png']}  "
            f"(photon-transfer curve, read-noise fit, real-vs-synthetic noise)", "ok")

    if not validation.get("ok"):
        log("One or more Phase-4 checks did not pass — review before synthesis",
            "warn")
        for chk in validation.get("checks", []):
            status = "ok" if chk.get("pass") else "warn"
            log(f"  {chk.get('name')}: {chk.get('metric')}={chk.get('value')}", status)

    console.print()
    console.print("[bold]Phase 5 — build training dataset:[/]")
    console.print(f"  python simulate_dataset.py --input <clean_images> "
                  f"--calibration {args.output}")

    return 0 if validation.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
