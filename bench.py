#!/usr/bin/env python3
"""Model-quality benchmark — find the best denoiser for your Pi.

Trains every architecture family on the configured dataset with an identical
calibration budget and reports PSNR gain, latency, params and Pareto fitness so
you can see, apples-to-apples, which model actually denoises best.

Examples
--------
  python bench.py                         # all families, default config data
  python bench.py --steps 300 --hardware rpi5_cpu
  python bench.py --families drunet nafnet restormer --channels 32
  python bench.py --quick                 # small budget, fast signal
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nsa.config import (Config, DataConfig, ModelConfig, OptimizationConfig,
                        OutputConfig, RunConfig, SensorConfig, MODEL_FAMILIES,
                        load_config, project_root, resolve_config_path)
from nsa.compiler import compile_stack
from nsa.denoise_hw_data import finalize_dataset_config
from nsa.inference import (calibrate_multi, estimate_device_latency_ms,
                           fake_quantize_int8, psnr, run)
from nsa.model_opts import search_combo_valid
from nsa.models import build_model, count_params
from nsa.raw_io import build_frame, build_frame_from_source, list_frames
from nsa.report import compute_fitness
from nsa.sensors import get_sensor
from nsa.theme import banner, console, log

ROOT = project_root()


def _load_frames(cfg: Config, n: int):
    sensor = get_sensor(cfg.sensor.sensor)
    frames = []
    if cfg.sensor.real_capture:
        for src in list_frames(cfg.sensor.dataset_path, cfg.sensor.filter or [], limit=n):
            try:
                frames.append(build_frame_from_source(
                    src, cfg.sensor.gain, cfg.data.temporal_frames,
                    cfg.optimization.patch_size, sensor, cfg.output.seed,
                    simulate_noise=cfg.sensor.simulate_noise))
            except Exception as exc:  # noqa: BLE001
                log(f"skip {src.get('name','?')}: {exc}", "warn")
    if not frames:
        frames.append(build_frame(
            input_raw=None, gain=cfg.sensor.gain,
            temporal_frames=cfg.data.temporal_frames,
            patch=cfg.optimization.patch_size, sensor=sensor, seed=cfg.output.seed))
    return frames


def _bench_one(family: str, ch: int, depth: int, conv: str, act: str,
               hardware: str, steps: int, frames, seed: int,
               quantize: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = Config(
        hardware=hardware,
        model=ModelConfig(model_family=family, base_channels=ch,
                          block_depth=depth, conv_type=conv, activation=act),
        optimization=OptimizationConfig(quantize=quantize, calibration_steps=steps),
        output=OutputConfig(dir="outputs", show_window=False, seed=seed),
    )
    model = build_model(cfg.model)
    n_params = count_params(model)

    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        comp = compile_stack(cfg, n_params)

    pairs = [(f.noisy_rgb, f.clean_rgb) for f in frames]
    t0 = time.perf_counter()
    calibrate_multi(model, pairs, steps, seed)
    train_s = time.perf_counter() - t0

    psnr_in = float(np.mean([psnr(f.noisy_rgb, f.clean_rgb) for f in frames]))
    psnr_fp32 = float(np.mean([psnr(run(model, f.noisy_rgb)[0], f.clean_rgb)
                               for f in frames]))
    use_int8 = quantize and hardware in ("hailo8", "deepx")
    if use_int8:
        q = fake_quantize_int8(model)
        psnr_int8 = float(np.mean([psnr(run(q, f.noisy_rgb)[0], f.clean_rgb)
                                   for f in frames]))
        drop = psnr_int8 - psnr_fp32
        final = psnr_int8
        emodel = q
    else:
        drop = 0.0
        final = psnr_fp32
        emodel = model

    latency = estimate_device_latency_ms(emodel, cfg.optimization.patch_size,
                                         hardware, use_int8)
    weight_kb = n_params / 1024.0 if use_int8 else n_params * 4 / 1024.0
    fit = compute_fitness(final, latency, drop, weight_kb=weight_kb,
                          act_kb=comp.est_sram_kb, sram_budget_kb=comp.sram_budget_kb,
                          psnr_in=psnr_in)
    return {
        "family": family, "channels": ch, "depth": depth,
        "conv": conv, "act": act,
        "psnr_in": round(psnr_in, 2), "psnr_out": round(final, 2),
        "gain": round(final - psnr_in, 2), "quant_drop": round(drop, 2),
        "latency_ms": round(latency, 1), "params": n_params,
        "fitness": fit.score, "grade": fit.grade, "train_s": round(train_s, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--families", nargs="*", default=list(MODEL_FAMILIES))
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--conv-type", dest="conv", default="standard")
    p.add_argument("--activation", dest="act", default="relu")
    p.add_argument("--hardware", default=None)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--no-quantize", dest="quantize", action="store_false")
    p.add_argument("--quick", action="store_true", help="steps=120 for fast signal")
    p.add_argument("--out", default="outputs/bench.json")
    args = p.parse_args()

    if args.quick:
        args.steps = 120

    banner("Model quality benchmark")
    cfg = load_config(resolve_config_path(args.config, ROOT))
    finalize_dataset_config(cfg, ROOT)
    hardware = args.hardware or cfg.hardware
    frames = _load_frames(cfg, args.frames)
    psnr_in = float(np.mean([psnr(f.noisy_rgb, f.clean_rgb) for f in frames]))
    log(f"Data: {cfg.sensor.dataset_path}  ·  input PSNR {psnr_in:.2f} dB  ·  "
        f"{frames[0].width}x{frames[0].height}px  ·  {len(frames)} frame(s)", "info")
    log(f"Budget: {args.steps} steps · {args.channels}ch × depth {args.depth} · "
        f"target {hardware}", "info")
    console.print()

    rows = []
    for fam in args.families:
        conv = args.conv if fam not in ("nafnet", "restormer") else "depthwise"
        act = args.act if fam not in ("nafnet", "restormer") else "relu"
        if not search_combo_valid(fam, conv, act):
            continue
        console.print(f"  training [bold]{fam.upper():<10}[/] ...", end="")
        try:
            r = _bench_one(fam, args.channels, args.depth, conv, act, hardware,
                           args.steps, frames, cfg.output.seed, args.quantize)
            rows.append(r)
            console.print(f"  gain [bold cyan]{r['gain']:+5.2f} dB[/]  "
                          f"out {r['psnr_out']:5.2f}  {r['latency_ms']:6.1f} ms  "
                          f"{r['params']/1000:6.1f}k  fit {r['fitness']:5.1f} {r['grade']}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]ERROR: {exc}[/]")

    rows.sort(key=lambda r: r["gain"], reverse=True)
    console.print()
    console.print("  [bold]RANKING BY PSNR GAIN[/]")
    console.print(f"  {'#':<3}{'family':<11}{'gain':>8}{'out':>8}{'lat(ms)':>9}"
                  f"{'params':>10}{'fit':>7}  grade")
    console.print("  " + "-" * 62)
    for i, r in enumerate(rows, 1):
        console.print(f"  {i:<3}{r['family']:<11}{r['gain']:>+7.2f} {r['psnr_out']:>7.2f} "
                      f"{r['latency_ms']:>8.1f} {r['params']/1000:>8.1f}k {r['fitness']:>6.1f}"
                      f"  {r['grade']}")

    if rows:
        best_q = rows[0]
        best_fit = max(rows, key=lambda r: r["fitness"])
        console.print()
        log(f"Best quality : {best_q['family'].upper()} "
            f"(+{best_q['gain']:.2f} dB, {best_q['latency_ms']:.0f} ms)", "ok")
        log(f"Best fitness : {best_fit['family'].upper()} "
            f"({best_fit['fitness']:.1f}, +{best_fit['gain']:.2f} dB, "
            f"{best_fit['latency_ms']:.0f} ms)", "ok")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "input_psnr": round(psnr_in, 2), "hardware": hardware,
        "steps": args.steps, "channels": args.channels, "depth": args.depth,
        "results": rows,
    }, indent=2), encoding="utf-8")
    log(f"Wrote {out}", "info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
