#!/usr/bin/env python3
"""Fit per-gain IMX662 noise models (LCG + HCG) from calibration + bursts.

Writes ``models/noise/<sensor>_ag<gain>.json`` for each requested gain, where
``<sensor>`` is ``imx662`` (LCG) or ``imx662h`` (HCG).

Two sources are combined:

* ``calibration/imx662{,h}_gain256/``   → fit at gain 256 from bias/dark/flat.
* ``bursts/<scene>/ag<g>/*.dng``       → fit at every gain a burst covers.

For gains where only one source exists we take that source.  For gain 256 we
report both fits so you can compare; the burst fit is written last (used at
train time by default) but the calibration JSON is saved alongside for
reference.  Missing gains (e.g. we only have bursts at ag128) are linearly
scaled from the nearest available gain in the same conversion-gain mode.

Examples
--------
    # fit LCG + HCG for the deploy gains
    .venv/bin/python fit_imx662_noise.py --gains 128 256 512

    # only LCG
    .venv/bin/python fit_imx662_noise.py --gains 128 256 512 --sensors imx662

    # inspect one burst without writing
    .venv/bin/python fit_imx662_noise.py --dry-run --sensors imx662 --gains 128
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.synth.fit import (
    fit_from_burst,
    fit_from_calibration,
    discover_bursts,
)
from nsa.synth.noise import (
    GainNoiseModel,
    save_gain_model,
    scale_model_to_gain,
)


SENSOR_MODES = {
    "imx662": "imx662_gain256",     # LCG
    "imx662h": "imx662h_gain256",   # HCG
}


def _fmt(model: GainNoiseModel) -> str:
    K = ", ".join(f"{k:.5f}" for k in model.K)
    R = ", ".join(f"{s:.5f}" for s in model.read_sigma)
    r2 = ", ".join(f"{r:.3f}" for r in model.fit_r2)
    return (f"K=[{K}]  read_sigma=[{R}]  R^2=[{r2}]  "
            f"row=~{max(model.row_sigma or [0.0]):.4f}  "
            f"BLE=~{model.black_level_norm:.4g}  n={model.n_frames}")


def fit_one(sensor: str, gain: int, *,
            calib_root: Path, bursts_root: Path,
            burst_n_gt: int, burst_n_use: int,
            crop: int | None,
            ) -> tuple[GainNoiseModel | None, str]:
    """Return (best_model, log_line).  Best-source priority: burst → calib → scaled."""
    # 1) Burst fit — only from bursts captured in the same conversion-gain
    #    mode.  Repo convention: ``bursts/<scene>/ag<g>/``  is LCG; a suffix
    #    like ``_hcg`` (in the sub-folder or scene name) marks HCG.  For gains
    #    where no matching burst exists we fall through to the calibration
    #    fit (or later, a linear scale from another gain).
    burst_model: GainNoiseModel | None = None
    burst_err = ""
    want_hcg = sensor.endswith("h")
    candidates: list[Path] = []
    for scene in sorted(bursts_root.iterdir()):
        if not scene.is_dir():
            continue
        scene_is_hcg = "hcg" in scene.name.lower()
        for sub in sorted(scene.iterdir()):
            if not sub.is_dir() or f"ag{gain}" not in sub.name.lower():
                continue
            if not any(sub.glob("*.dng")):
                continue
            sub_is_hcg = ("hcg" in sub.name.lower()) or scene_is_hcg
            if sub_is_hcg == want_hcg:
                candidates.append(sub)
    for bd in candidates:
        try:
            burst_model = fit_from_burst(
                bd, sensor=sensor, gain=gain,
                n_gt=burst_n_gt, n_burst=burst_n_use, center_crop=crop,
            )
            burst_model.fit_source = f"burst:{bd.relative_to(bursts_root.parent)}"
            break
        except Exception as exc:
            burst_err = f"{bd}: {exc}"
            continue

    # 2) Calibration fit — only meaningful at whatever gain the calibration
    #    tree was captured at (gain 256 in this repo).
    calib_model: GainNoiseModel | None = None
    calib_err = ""
    tag = SENSOR_MODES.get(sensor)
    if tag and (calib_root / tag).is_dir():
        try:
            fitted = fit_from_calibration(calib_root / tag, sensor=sensor,
                                          gain=256)
            calib_model = fitted if gain == 256 else scale_model_to_gain(fitted, gain)
        except Exception as exc:
            calib_err = str(exc)

    def _mean_r2(m: GainNoiseModel) -> float:
        return float(np.mean(m.fit_r2)) if m.fit_r2 else 0.0

    if burst_model and calib_model:
        # Prefer whichever fit has higher mean per-channel R^2 — real bursts
        # have residual signal (micro-alignment, illumination flicker) that
        # inflates variance and worsens R^2, so calib-scaled usually wins at
        # gains where the burst is noisy.
        br2, cr2 = _mean_r2(burst_model), _mean_r2(calib_model)
        if br2 >= cr2:
            other = f"calib-ref R^2={cr2:.3f} {_fmt(calib_model)}"
            return burst_model, f"burst (R^2={br2:.3f}) preferred  ·  {other}"
        other = f"burst-ref R^2={br2:.3f} {_fmt(burst_model)}"
        return calib_model, f"calib (R^2={cr2:.3f}) preferred  ·  {other}"
    if burst_model:
        return burst_model, f"burst  · calib unavailable ({calib_err or 'no dir'})"
    if calib_model:
        return calib_model, f"calib  · burst unavailable ({burst_err or 'no matching burst'})"
    return None, f"no fit  · burst_err={burst_err!r}  calib_err={calib_err!r}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensors", nargs="+", default=list(SENSOR_MODES.keys()),
                   choices=list(SENSOR_MODES.keys()))
    p.add_argument("--gains", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--calib-root", type=Path,
                   default=ROOT / "datasets/imx662_project/calibration")
    p.add_argument("--bursts-root", type=Path,
                   default=ROOT / "datasets/imx662_project/bursts")
    p.add_argument("--output-dir", type=Path, default=ROOT / "models/noise")
    p.add_argument("--burst-n-gt", type=int, default=128,
                   help="frames averaged into the burst GT (first N)")
    p.add_argument("--burst-n-use", type=int, default=128,
                   help="held-out frames whose residuals build the PTC")
    p.add_argument("--crop", type=int, default=1024,
                   help="post-pack center crop (0 = full frame)")
    p.add_argument("--dry-run", action="store_true",
                   help="fit and print, do not write JSONs")
    args = p.parse_args()

    crop = args.crop or None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    print(f"Calibration:  {args.calib_root}")
    print(f"Bursts:       {args.bursts_root}")
    print(f"Sensors:      {args.sensors}")
    print(f"Gains:        {args.gains}")
    print()

    fitted: dict[tuple[str, int], GainNoiseModel] = {}
    logs: list[tuple[str, int, str]] = []

    for sensor in args.sensors:
        for g in args.gains:
            try:
                model, note = fit_one(
                    sensor, g,
                    calib_root=args.calib_root,
                    bursts_root=args.bursts_root,
                    burst_n_gt=args.burst_n_gt,
                    burst_n_use=args.burst_n_use,
                    crop=crop,
                )
            except Exception:
                traceback.print_exc()
                model, note = None, "exception (see traceback)"
            if model is not None:
                fitted[(sensor, g)] = model
                logs.append((sensor, g, f"OK  {note}  |  {_fmt(model)}"))
            else:
                logs.append((sensor, g, f"MISS  {note}"))
                exit_code = 2

    # Fill in gains with no fit by scaling from the nearest fitted gain in the
    # same conversion-gain mode (linear ELD-style scaling).
    for sensor in args.sensors:
        available = sorted(g for (s, g) in fitted if s == sensor)
        if not available:
            continue
        for g in args.gains:
            if (sensor, g) in fitted:
                continue
            nearest = min(available, key=lambda x: abs(x - g))
            fitted[(sensor, g)] = scale_model_to_gain(fitted[(sensor, nearest)], g)
            logs.append((sensor, g,
                         f"SCALED  from ag{nearest}: {_fmt(fitted[(sensor, g)])}"))

    for sensor, g, msg in logs:
        print(f"  {sensor} ag{g:<3d}  {msg}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return exit_code

    print()
    for (sensor, g), model in sorted(fitted.items()):
        out = args.output_dir / f"{sensor}_ag{g}.json"
        save_gain_model(model, out)
        print(f"  wrote {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
