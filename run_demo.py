#!/usr/bin/env python3
"""NSA - 6-Level Optimization Stack :: prototype demo entry point.

Runs a noisy IMX662 Bayer RAW frame and a target-hardware configuration through
all six levels of the optimization stack and delivers the four demo outputs:

    1. a live compilation log,
    2. on-disk model artifacts (.onnx + hardware-ready binary),
    3. a 3-panel visual validation matrix, and
    4. a Pareto fitness performance report.

Usage:
    python run_demo.py                       # uses config.yaml
    python run_demo.py --hardware deepx --activation gelu
    python run_demo.py --hardware hailo8 --model-family nafnet --gain 512
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn)

from nsa import __version__
from nsa.compiler import compile_stack
from nsa.config import apply_overrides, build_parser, load_config, ConfigError
from nsa.export import export_onnx, write_device_artifact
from nsa.inference import (calibrate, estimate_device_latency_ms,
                           fake_quantize_int8, psnr, run)
from nsa.models import build_model, count_params
from nsa.raw_io import build_frame
from nsa.report import compute_fitness, print_report
from nsa.theme import (RPI_GREEN, banner, console, kv_table, level_rule, log,
                       pause)
from nsa.visualize import render_panel


def main() -> int:
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    banner(f"Neural Sensor Architecture  ·  v{__version__}")

    try:
        cfg.validate()
    except ConfigError as exc:
        log(str(exc), "err")
        return 2

    torch.manual_seed(cfg.output.seed)
    np.random.seed(cfg.output.seed)

    # -- Configuration summary --------------------------------------------------
    console.print(kv_table(
        [
            ("hardware", f"{cfg.hardware}  ({cfg.hardware_name})"),
            ("model_family", cfg.model.model_family),
            ("base_channels", str(cfg.model.base_channels)),
            ("block_depth", str(cfg.model.block_depth)),
            ("conv_type", cfg.model.conv_type),
            ("activation", cfg.model.activation),
            ("sensor", f"{cfg.sensor.model}  {cfg.sensor.bayer_pattern}  {cfg.sensor.bit_depth}-bit"),
            ("gain", f"{cfg.sensor.gain}×"),
            ("input_raw", cfg.sensor.input_raw or "synthetic (auto-generated)"),
            ("quantize", "INT8" if cfg.optimization.quantize else "off"),
        ],
        title="COMPILATION PROFILE  ·  selected inputs",
    ))

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================================
    # LEVEL 1 - SENSOR / INPUT
    # ===========================================================================
    level_rule(1, "SENSOR  ·  IMX662 Bayer RAW ingestion")
    log(f"Reading {cfg.sensor.model} RAW @ {cfg.sensor.gain}× analog gain "
        f"({cfg.sensor.bayer_pattern}, {cfg.sensor.bit_depth}-bit)", "step")
    frame = build_frame(
        input_raw=cfg.sensor.input_raw,
        gain=cfg.sensor.gain,
        temporal_frames=cfg.data.temporal_frames,
        patch=cfg.optimization.patch_size,
        bayer_pattern=cfg.sensor.bayer_pattern,
        seed=cfg.output.seed,
    )
    src = "synthetic IMX662 capture" if frame.source == "synthetic" else frame.source
    log(f"Frame source: {src}", "info")
    log(f"Working resolution: {frame.width}×{frame.height}  ·  demosaiced linear RGB", "ok")

    # ===========================================================================
    # LEVEL 2 - GROUND TRUTH / DATA
    # ===========================================================================
    level_rule(2, "DATA  ·  temporal ground-truth synthesis")
    log(f"Averaging {cfg.data.temporal_frames} independent reads to build "
        "clean reference", "step")
    pause(0.2)
    psnr_in = psnr(frame.noisy_rgb, frame.clean_rgb)
    log(f"Input frame PSNR vs reference: {psnr_in:.2f} dB  "
        f"(heavily corrupted, as expected at {cfg.sensor.gain}×)", "warn")

    # ===========================================================================
    # LEVEL 3 - MODEL ARCHITECTURE
    # ===========================================================================
    level_rule(3, "ARCHITECTURE  ·  building denoiser graph")
    model = build_model(cfg.model)
    n_params = count_params(model)
    log(f"Instantiated {cfg.model.model_family.upper()} "
        f"({cfg.model.conv_type} conv, {cfg.model.base_channels}ch × "
        f"{cfg.model.block_depth} blocks, {cfg.model.activation})", "step")
    log(f"Trainable parameters: {n_params:,}", "ok")

    # ===========================================================================
    # LEVEL 4 - COMPILER / OPTIMIZATION  (live compilation log)
    # ===========================================================================
    level_rule(4, "COMPILER  ·  hardware-aware legalization")
    result = compile_stack(cfg, n_params)

    # ===========================================================================
    # LEVEL 5 - CALIBRATION + QUANTIZATION
    # ===========================================================================
    level_rule(5, "CALIBRATION  ·  live fit + INT8 quantization")
    with Progress(
        TextColumn("[muted]{task.description}"),
        BarColumn(complete_style=RPI_GREEN, finished_style=RPI_GREEN),
        TextColumn("[muted]{task.percentage:>3.0f}%"),
        TextColumn("[val]loss {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("calibrating", total=cfg.optimization.calibration_steps, loss=1.0)

        def on_step(i, total, loss):
            progress.update(task, completed=i, loss=loss)

        calibrate(model, frame.noisy_rgb, frame.clean_rgb,
                  cfg.optimization.calibration_steps, cfg.output.seed, on_step)

    fp32_out, fwd_ms = run(model, frame.noisy_rgb)
    psnr_fp32 = psnr(fp32_out, frame.clean_rgb)
    log(f"FP32 inference complete  ·  forward pass {fwd_ms:.1f} ms (host) "
        f"·  PSNR {psnr_fp32:.2f} dB", "ok")

    quantized = bool(cfg.uses_accelerator and cfg.optimization.quantize)
    if quantized:
        qmodel = fake_quantize_int8(model)
        int8_out, _ = run(qmodel, frame.noisy_rgb)
        psnr_int8 = psnr(int8_out, frame.clean_rgb)
        quant_drop = psnr_int8 - psnr_fp32
        log(f"INT8 quantization applied ({result.quant_scheme})  ·  "
            f"PSNR {psnr_int8:.2f} dB  ·  drop {quant_drop:+.2f} dB", "info")
        final_out, final_psnr, export_model = int8_out, psnr_int8, qmodel
    else:
        quant_drop = 0.0
        final_out, final_psnr, export_model = fp32_out, psnr_fp32, model

    latency_ms = estimate_device_latency_ms(
        export_model, cfg.optimization.patch_size, cfg.hardware, quantized)
    log(f"Estimated on-device latency ({cfg.hardware}): {latency_ms:.1f} ms "
        f"({1000.0/latency_ms:.0f} FPS)", "step")

    # ===========================================================================
    # LEVEL 6 - EXPORT PROFILE
    # ===========================================================================
    level_rule(6, "EXPORT  ·  writing hardware-ready artifacts")
    onnx_path = out_dir / "exported_model.onnx"
    export_onnx(model, cfg.optimization.patch_size, onnx_path)
    log(f"Wrote {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)  "
        "[FP32 baseline graph]", "ok")

    artifact_path = out_dir / f"hardware_ready{cfg.artifact_ext}"
    info = write_device_artifact(export_model, cfg, result, artifact_path)
    log(f"Wrote {artifact_path}  ({info['total_bytes']/1024:.1f} KB)  "
        f"[{result.export_format}, {info['layers']} layers]", "ok")

    # ===========================================================================
    # OUTPUT 3 - VISUAL VALIDATION MATRIX
    # ===========================================================================
    level_rule(0, "VALIDATION  ·  before / ground-truth / after")
    panel_path = out_dir / "validation_panel.png"
    meta = {
        "gain": cfg.sensor.gain,
        "frames": cfg.data.temporal_frames,
        "family": cfg.model.model_family,
        "precision": "INT8" if quantized else result.precision.upper(),
        "hardware_name": cfg.hardware_name,
        "psnr_in": psnr_in,
        "psnr_out": final_psnr,
    }
    render_panel(frame.noisy_rgb, frame.clean_rgb, final_out, meta,
                 panel_path, show=cfg.output.show_window)
    log(f"Saved validation matrix -> {panel_path}", "ok")
    if cfg.output.show_window:
        log("Opened 3-panel validation window", "info")

    # ===========================================================================
    # OUTPUT 4 - PARETO FITNESS REPORT
    # ===========================================================================
    fit = compute_fitness(final_psnr, latency_ms, quant_drop)
    profile = (f"{cfg.model.model_family.upper()} · {cfg.model.base_channels}ch × "
               f"{cfg.model.block_depth} · {cfg.model.conv_type} · "
               f"{cfg.model.activation} · {meta['precision']}")
    print_report(fit, cfg.hardware_name, profile)

    if result.warnings:
        console.print()
        log(f"{len(result.warnings)} compiler note(s) issued during this build:", "warn")
        for w in result.warnings:
            console.print(f"      [warn]▲[/warn] {w}")

    console.print()
    log("Artifacts written to: " + str(out_dir.resolve()), "ok")
    console.print(
        f"   [muted]·[/muted] exported_model.onnx   "
        f"[muted]·[/muted] hardware_ready{cfg.artifact_ext}   "
        f"[muted]·[/muted] validation_panel.png"
    )
    console.print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[err]Aborted by user.[/err]")
        sys.exit(130)
