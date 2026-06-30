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

import os
import sys
from pathlib import Path

import numpy as np
import torch
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn)

from nsa import __version__
from nsa.compiler import compile_stack
from nsa.config import apply_overrides, build_parser, load_config, ConfigError
from nsa.export import export_onnx, write_device_artifact
from nsa.inference import (calibrate, calibrate_multi, estimate_device_latency_ms,
                           fake_quantize_int8, psnr, run)
from nsa.models import build_model, count_params
from nsa.raw_io import build_frame, build_frame_from_source, list_frames
from nsa.sensors import get_sensor
from nsa.report import compute_fitness, print_report
from nsa.theme import (RPI_GREEN, banner, console, kv_table, level_rule, log,
                       pause)
from nsa.visualize import render_panel


def _has_display() -> bool:
    """True if a GUI display is available for the validation pop-up window."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> int:
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    # Headless box (e.g. Pi over SSH with no X): never try to open a window.
    headless = cfg.output.show_window and not _has_display()
    if headless:
        cfg.output.show_window = False

    banner(f"Neural Sensor Architecture  ·  v{__version__}")

    try:
        cfg.validate()
    except ConfigError as exc:
        log(str(exc), "err")
        return 2

    torch.manual_seed(cfg.output.seed)
    np.random.seed(cfg.output.seed)

    sensor = get_sensor(cfg.sensor.sensor)

    # -- Configuration summary --------------------------------------------------
    console.print(kv_table(
        [
            ("hardware", f"{cfg.hardware}  ({cfg.hardware_name})"),
            ("model_family", cfg.model.model_family),
            ("base_channels", str(cfg.model.base_channels)),
            ("block_depth", str(cfg.model.block_depth)),
            ("conv_type", cfg.model.conv_type),
            ("activation", cfg.model.activation),
            ("sensor", f"{sensor.label} — {sensor.family}  ·  "
                       f"{sensor.bayer}  {sensor.bit_depth}-bit"),
            ("gain", f"{cfg.sensor.gain}×"),
            ("input", (cfg.sensor.dataset_path or cfg.sensor.input_raw or
                       "synthetic (auto-generated)") if cfg.sensor.real_capture
                      else (cfg.sensor.input_raw or "synthetic (auto-generated)")),
            ("capture_mode", ("REAL captures" if cfg.sensor.real_capture
                              else "simulated physics")
                             + (" + simulated noise" if cfg.sensor.simulate_noise else "")),
            ("dataset_filter", " ".join(cfg.sensor.filter) if cfg.sensor.filter else "—"),
            ("run_mode", f"batch ×{cfg.run.batch_size}" if cfg.run.mode == "batch"
                         else "single frame"),
            ("quantize", "INT8" if cfg.optimization.quantize else "off"),
        ],
        title="COMPILATION PROFILE  ·  selected inputs",
    ))

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================================
    # LEVEL 1 - SENSOR / INPUT
    # ===========================================================================
    level_rule(1, f"SENSOR  ·  {sensor.label} Bayer RAW ingestion")
    log(f"Sensor profile: {sensor.label} — {sensor.family}  ·  "
        f"QE {sensor.qe:.0%}, read {sensor.read_noise:.1f}e-, "
        f"well {sensor.full_well:,.0f}e-", "step")
    real_capture = bool(cfg.sensor.real_capture)
    real_source = cfg.sensor.dataset_path or cfg.sensor.input_raw
    simulate_noise = bool(cfg.sensor.simulate_noise)
    filter_tokens = list(cfg.sensor.filter or [])
    batch = cfg.run.mode == "batch"
    n_want = cfg.run.batch_size if batch else 1
    patch = cfg.optimization.patch_size
    tframes = cfg.data.temporal_frames

    # -- Assemble the working frame set (1 frame for single, N for batch) -----
    frames = []
    if real_capture:
        sources = list_frames(real_source, filter_tokens, limit=n_want)
        if filter_tokens:
            log(f"Dataset filter: {' '.join(filter_tokens)}  "
                f"({len(sources)} matching frame(s))", "step")
        for i, s in enumerate(sources):
            try:
                frames.append(build_frame_from_source(
                    s, cfg.sensor.gain, tframes, patch, sensor,
                    cfg.output.seed + i, simulate_noise=simulate_noise))
            except Exception as exc:
                log(f"Skipped {s.get('name', '?')}: {exc}", "warn")
        if not frames:
            log(f"No usable real frames at {real_source!r} — falling back to "
                f"synthetic {sensor.label}", "warn")
    if not frames:                                  # synthetic (or fallback)
        for i in range(n_want):
            frames.append(build_frame(
                input_raw=(None if real_capture else cfg.sensor.input_raw),
                gain=cfg.sensor.gain, temporal_frames=tframes, patch=patch,
                sensor=sensor, seed=cfg.output.seed + i))

    frame = frames[0]                               # representative for the panel
    gt_kind = frame.gt_kind
    real_loaded = real_capture and frame.source != "synthetic"

    if batch:
        log(f"Batch mode: {len(frames)} frame(s) loaded for calibration", "step")
    if real_loaded:
        log(f"Real-capture mode: {frame.source}", "step")
    else:
        log(f"Reading {sensor.label} RAW @ {cfg.sensor.gain}× analog gain "
            f"({sensor.bayer}, {sensor.bit_depth}-bit)", "step")
    log(f"Frame source: {frame.source if frame.source != 'synthetic' else f'synthetic {sensor.label} capture'}", "info")
    log(f"Working resolution: {frame.width}×{frame.height}  ·  demosaiced linear RGB", "ok")

    # ===========================================================================
    # LEVEL 2 - GROUND TRUTH / DATA
    # ===========================================================================
    level_rule(2, "DATA  ·  ground-truth reference")
    _GT_MSG = {
        "paired": "Paired real ground truth (noisy/gt folder, denoise-hw convention)",
        "paired+sim": "Paired gt frame used as clean source; sensor noise simulated",
        "clean+sim": "Loaded frame used as clean source; sensor noise simulated",
        "reference": "NL-means + edge-preserving reference (single real capture)",
        "temporal": f"Temporal average of {tframes} independent reads",
    }
    log(_GT_MSG.get(gt_kind, "ground truth"), "step")
    pause(0.2)
    psnr_in = float(np.mean([psnr(f.noisy_rgb, f.clean_rgb) for f in frames]))
    label = "Avg input PSNR" if batch else "Input frame PSNR"
    log(f"{label} vs reference: {psnr_in:.2f} dB  "
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

        pairs = [(f.noisy_rgb, f.clean_rgb) for f in frames]
        calibrate_multi(model, pairs, cfg.optimization.calibration_steps,
                        cfg.output.seed, on_step)

    fp32_out, fwd_ms = run(model, frame.noisy_rgb)
    psnr_fp32 = float(np.mean([psnr(run(model, f.noisy_rgb)[0], f.clean_rgb)
                               for f in frames]))
    lbl = "avg PSNR" if batch else "PSNR"
    log(f"FP32 inference complete  ·  forward pass {fwd_ms:.1f} ms (host) "
        f"·  {lbl} {psnr_fp32:.2f} dB", "ok")

    quantized = bool(cfg.uses_accelerator and cfg.optimization.quantize)
    if quantized:
        qmodel = fake_quantize_int8(model)
        int8_out, _ = run(qmodel, frame.noisy_rgb)
        psnr_int8 = float(np.mean([psnr(run(qmodel, f.noisy_rgb)[0], f.clean_rgb)
                                   for f in frames]))
        quant_drop = psnr_int8 - psnr_fp32
        log(f"INT8 quantization applied ({result.quant_scheme})  ·  "
            f"{lbl} {psnr_int8:.2f} dB  ·  drop {quant_drop:+.2f} dB", "info")
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
    if export_onnx(model, cfg.optimization.patch_size, onnx_path) and onnx_path.exists():
        log(f"Wrote {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)  "
            "[FP32 baseline graph]", "ok")
    else:
        log("ONNX export skipped ('onnx' package unavailable) — "
            "install it with: pip install onnx", "warn")

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
        "sensor": sensor.label,
        "gain": cfg.sensor.gain,
        "frames": cfg.data.temporal_frames,
        "real_capture": real_loaded,
        "gt_kind": gt_kind,
        "batch": len(frames) if batch else 0,
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
    elif headless:
        log("No display detected (headless) — window skipped; "
            "open validation_panel.png to view the result", "info")

    # ===========================================================================
    # OUTPUT 4 - PARETO FITNESS REPORT
    # ===========================================================================
    fit = compute_fitness(final_psnr, latency_ms, quant_drop,
                          weight_kb=info["total_bytes"] / 1024.0,
                          act_kb=result.est_sram_kb,
                          sram_budget_kb=result.sram_budget_kb)
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
    onnx_note = "exported_model.onnx   [muted]·[/muted] " if onnx_path.exists() else ""
    console.print(
        f"   [muted]·[/muted] {onnx_note}"
        f"hardware_ready{cfg.artifact_ext}   "
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
