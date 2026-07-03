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

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn)

from nsa import __version__
from nsa.compiler import assess_targets, compile_stack
from nsa.config import (apply_overrides, build_parser, finalize_dataset_config,
                        load_config, project_root, resolve_config_path, ConfigError)

ROOT = Path(__file__).resolve().parent
from nsa.export import export_onnx, write_device_artifact
from nsa.hf_runner import (calibration_steps_for_hf, copy_hf_onnx,
                           is_hf_pretrained, load_hf_model)
from nsa.inference import (build_loss, calibrate, calibrate_multi,
                           estimate_device_latency_ms, fake_quantize_int8, lpips,
                           psnr, run, ssim, temporal_denoise)
from nsa.models import build_model, count_params
from nsa.raw_io import (build_burst, build_frame, build_frame_from_source,
                        list_frames)
from nsa.sensors import get_sensor, with_noise_std
from nsa.report import compute_fitness, print_report, print_target_suitability
from nsa.scaling import render_scaling_chart, scaling_curves
from nsa.model_opts import (instantiate_summary, model_display_line, profile_rows,
                            uses_activation)
from nsa.theme import (RPI_GREEN, banner, console, kv_table, level_rule, log,
                       pause)
from nsa.visualize import render_panel


def _has_display() -> bool:
    """True if a GUI display is available for the validation pop-up window."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _loss_summary(lc) -> str:
    """Human-readable one-liner for the active loss + its relevant parameter(s)."""
    name = lc.name.lower()
    if name == "charbonnier":
        return f"Charbonnier (eps={lc.charbonnier_eps:g})"
    if name == "l1":
        return "L1 (mean absolute error)"
    if name in ("l2", "mse"):
        return "L2 (mean squared error)"
    if name == "huber":
        return f"Huber (delta={lc.huber_delta:g})"
    if name == "ssim":
        return f"SSIM (window={lc.ssim_window})"
    if name == "charbonnier_ssim":
        return (f"Charbonnier+SSIM (eps={lc.charbonnier_eps:g}, "
                f"ssim_weight={lc.ssim_weight:g}, window={lc.ssim_window})")
    return name


def main() -> int:
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(resolve_config_path(args.config, ROOT)), args)
    finalize_dataset_config(cfg, ROOT)
    ds_quality_notice = None
    if cfg.sensor.real_capture and cfg.sensor.dataset_path:
        from nsa.denoise_hw_data import dataset_quality_notice
        ds_quality_notice = dataset_quality_notice(cfg.sensor.dataset_path)

    # Headless box (e.g. Pi over SSH with no X): never try to open a window.
    headless = cfg.output.show_window and not _has_display()
    if headless:
        cfg.output.show_window = False

    banner(f"Neural Architecture Search  ·  v{__version__}")

    try:
        cfg.validate()
    except ConfigError as exc:
        log(str(exc), "err")
        return 2

    torch.manual_seed(cfg.output.seed)
    np.random.seed(cfg.output.seed)

    sensor = with_noise_std(get_sensor(cfg.sensor.sensor), cfg.sensor.noise_std)

    # -- Configuration summary --------------------------------------------------
    console.print(kv_table(
        [("hardware", f"{cfg.hardware}  ({cfg.hardware_name})")]
        + profile_rows(cfg.model)
        + ([("hf_model", cfg.model.hf_model)] if cfg.model.hf_model else [])
        + [
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
                         else (f"temporal burst ×{cfg.run.burst}"
                               if cfg.run.mode == "temporal" else "single frame")),
            ("quantize", ("INT8 (QAT)" if cfg.optimization.qat else "INT8")
                         if cfg.optimization.quantize else "off"),
        ],
        title="COMPILATION PROFILE  ·  selected inputs",
    ))

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================================
    # LEVEL 1 - SENSOR / INPUT
    # ===========================================================================
    level_rule(1, f"SENSOR  ·  {sensor.label} Bayer RAW ingestion")
    if ds_quality_notice:
        log(ds_quality_notice, "warn")
    log(f"Sensor profile: {sensor.label} — {sensor.family}  ·  "
        f"QE {sensor.qe:.0%}, read {sensor.read_noise:.1f}e-, "
        f"well {sensor.full_well:,.0f}e-", "step")
    if cfg.sensor.noise_std is not None and cfg.sensor.noise_std >= 0:
        log(f"Noise override: injected read-noise std forced to "
            f"{sensor.read_noise:.1f}e- (denoise-hw style)", "step")
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
        # If the filter matched nothing, don't jump straight to a synthetic scene:
        # try again unfiltered so real captures are still used. In simulate-noise
        # mode any real image is a valid clean source; in plain real mode any real
        # frame is still a genuine capture (better than fabricated pixels).
        if not sources and filter_tokens:
            sources = list_frames(real_source, None, limit=n_want)
            if sources:
                log(f"No frames matched {' '.join(filter_tokens)!r}; using "
                    f"{len(sources)} unfiltered real frame(s) instead", "warn")
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
    hf_spec = None
    custom_naf = bool(cfg.model.model_family == "nafnet" and cfg.model.nafnet_enc_blocks)
    if cfg.model.hf_model:
        log(f"Hugging Face model: {cfg.model.hf_model}", "step")
        try:
            model, hf_spec = load_hf_model(
                cfg.model.hf_model, cfg.model,
                weight=cfg.model.hf_weight, download=True, root=ROOT)
        except Exception as exc:  # noqa: BLE001
            log(f"Could not load Hugging Face model: {exc}", "err")
            return 2
        n_params = count_params(model)
        log(f"Loaded Hub weights: {hf_spec.weight_path.name} "
            f"({hf_spec.weight_kind}) @ {hf_spec.sha[:12]}", "ok")
        if n_params:
            log(f"Trainable parameters: {n_params:,}", "ok")
        else:
            log("ONNX graph — parameter count not applicable", "info")
    else:
        model = build_model(cfg.model)
        n_params = count_params(model)
        log(instantiate_summary(cfg.model), "step")
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
    hf_onnx = bool(hf_spec and hf_spec.weight_kind in ("onnx", "imgutils"))
    cal_steps = calibration_steps_for_hf(cfg, cfg.optimization.calibration_steps)
    if is_hf_pretrained(cfg.model) and hf_onnx:
        cal_steps = 0
        log("Skipping calibration — using pretrained Hugging Face ONNX weights", "info")
    elif is_hf_pretrained(cfg.model) and cal_steps < cfg.optimization.calibration_steps:
        log(f"Light calibration ({cal_steps} steps) on top of Hub pretrained weights",
            "info")
    use_qat = bool(cfg.uses_accelerator and cfg.optimization.quantize
                   and not hf_onnx
                   and (cfg.optimization.qat or result.quant_scheme == "QAT"))
    if use_qat:
        log("QAT enabled -> training with INT8 fake-quant in the loop "
            "(straight-through estimator)", "step")
    lc = cfg.optimization.loss
    loss_fn = build_loss(
        lc.name, charbonnier_eps=lc.charbonnier_eps, huber_delta=lc.huber_delta,
        ssim_window=lc.ssim_window, ssim_weight=lc.ssim_weight)
    if cal_steps > 0:
        log(f"Loss: {_loss_summary(lc)}", "step")
    if cal_steps > 0:
        with Progress(
            TextColumn("[muted]{task.description}"),
            BarColumn(complete_style=RPI_GREEN, finished_style=RPI_GREEN),
            TextColumn("[muted]{task.percentage:>3.0f}%"),
            TextColumn("[val]loss {task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("calibrating", total=cal_steps, loss=1.0)

            def on_step(i, total, loss):
                progress.update(task, completed=i, loss=loss)

            pairs = [(f.noisy_rgb, f.clean_rgb) for f in frames]
            calibrate_multi(model, pairs, cal_steps,
                            cfg.output.seed, on_step, qat=use_qat,
                            loss_fn=loss_fn)

    fp32_out, fwd_ms = run(model, frame.noisy_rgb)
    psnr_fp32 = float(np.mean([psnr(run(model, f.noisy_rgb)[0], f.clean_rgb)
                               for f in frames]))
    lbl = "avg PSNR" if batch else "PSNR"
    log(f"FP32 inference complete  ·  forward pass {fwd_ms:.1f} ms (host) "
        f"·  {lbl} {psnr_fp32:.2f} dB", "ok")

    quantized = bool(cfg.uses_accelerator and cfg.optimization.quantize and not hf_onnx)
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

    # -- Temporal video-denoise pass (recursive burst denoising) --------------
    temporal = cfg.run.mode == "temporal"
    n_video = 0
    if temporal:
        if len(frames) >= 2 and real_loaded:
            burst = [f.noisy_rgb for f in frames]           # real sequence
        else:
            burst = build_burst(frame.clean_rgb, cfg.sensor.gain, sensor,
                                cfg.run.burst, cfg.output.seed)
        outputs, per_frame_ms = temporal_denoise(export_model, burst,
                                                 cfg.run.temporal_alpha)
        video_dir = out_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        for i, out in enumerate(outputs):
            bgr = cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(video_dir / f"frame_{i:03d}.png"), bgr)
        n_video = len(outputs)
        tpsnr = float(np.mean([psnr(o, frame.clean_rgb) for o in outputs]))
        # Per-frame PSNR vs the single noisy read shows the temporal gain.
        single_psnr = psnr(burst[0], frame.clean_rgb)
        log(f"Temporal denoise: {n_video} frames  ·  alpha {cfg.run.temporal_alpha}  "
            f"·  PSNR {single_psnr:.2f} -> {tpsnr:.2f} dB (recursive IIR)", "ok")
        log(f"Wrote {n_video} denoised frames -> {video_dir}", "ok")
        final_out, final_psnr = outputs[-1], tpsnr
        frame_noisy_for_panel = burst[0]
    else:
        frame_noisy_for_panel = frame.noisy_rgb

    # ===========================================================================
    # LEVEL 6 - EXPORT PROFILE
    # ===========================================================================
    level_rule(6, "EXPORT  ·  writing hardware-ready artifacts")
    onnx_path = out_dir / "exported_model.onnx"
    if hf_spec and hf_spec.weight_kind == "imgutils" and hf_spec.weight_path.is_file():
        import shutil
        shutil.copy2(hf_spec.weight_path, onnx_path)
        log(f"Wrote {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)  "
            f"[Hub ONNX: {hf_spec.weight_path.name}]", "ok")
    elif hf_spec and copy_hf_onnx(hf_spec, onnx_path):
        log(f"Wrote {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)  "
            f"[Hub ONNX: {hf_spec.weight_path.name}]", "ok")
    elif export_onnx(model, cfg.optimization.patch_size, onnx_path) and onnx_path.exists():
        log(f"Wrote {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)  "
            "[FP32 baseline graph]", "ok")
    else:
        log("ONNX export skipped ('onnx' package unavailable) — "
            "install it with: pip install onnx", "warn")

    artifact_path = out_dir / f"hardware_ready{cfg.artifact_ext}"
    info = write_device_artifact(export_model, cfg, result, artifact_path)
    log(f"Wrote {artifact_path}  ({info['total_bytes']/1024:.1f} KB)  "
        f"[{result.export_format}, {info['layers']} layers]", "ok")

    # Save the trained FP32 weights so `live.py` can run THIS exact model on a
    # live camera stream without re-training (Level-7 live testing).
    try:
        ckpt_path = out_dir / "model.pt"
        torch.save({
            "state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
            "model": {
                "family": cfg.model.model_family,
                "base_channels": cfg.model.base_channels,
                "block_depth": cfg.model.block_depth,
                "conv_type": cfg.model.conv_type,
                "activation": cfg.model.activation,
                "nafnet_enc": list(cfg.model.nafnet_enc_blocks),
                "nafnet_middle": cfg.model.nafnet_middle_blocks,
                "nafnet_dec": list(cfg.model.nafnet_dec_blocks),
                "hf_model": cfg.model.hf_model,
                "hf_weight": cfg.model.hf_weight,
                "hf_onnx": str(hf_spec.weight_path) if hf_spec and hf_spec.weight_kind == "onnx" else None,
                "hf_imgutils": bool(hf_spec and hf_spec.weight_kind == "imgutils"),
                "hf_variant": hf_spec.imgutils_variant if hf_spec else None,
            },
            "sensor": cfg.sensor.sensor,
            "gain": cfg.sensor.gain,
            "hardware": cfg.hardware,
            "params": n_params,
            "psnr_out": final_psnr,
        }, ckpt_path)
        log(f"Wrote {ckpt_path}  (FP32 weights for live testing)", "ok")
    except Exception as exc:  # noqa: BLE001
        log(f"Could not save live-testing checkpoint: {exc}", "warn")

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
        "frame_source": str(frame.source) if real_loaded else "synthetic",
        "dataset_path": str(real_source) if real_loaded else None,
        "batch": len(frames) if batch else 0,
        "family": cfg.model.model_family,
        "precision": "INT8" if quantized else result.precision.upper(),
        "hardware_name": cfg.hardware_name,
        "psnr_in": psnr_in,
        "psnr_out": final_psnr,
    }
    render_panel(frame_noisy_for_panel, frame.clean_rgb, final_out, meta,
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
    # Perceptual metrics on the representative frame: SSIM (structure) and LPIPS
    # (learned perceptual distance) catch the over-smoothing PSNR misses.
    final_ssim = ssim(final_out, frame.clean_rgb)
    ssim_in = ssim(frame_noisy_for_panel, frame.clean_rgb)
    try:
        final_lpips = lpips(final_out, frame.clean_rgb)
        lpips_in = lpips(frame_noisy_for_panel, frame.clean_rgb)
    except Exception as exc:  # noqa: BLE001
        final_lpips = lpips_in = None
        log(f"LPIPS unavailable ({exc}) — install 'lpips' for perceptual scoring",
            "warn")
    _lp = f"  ·  LPIPS {lpips_in:.3f} -> {final_lpips:.3f}" if final_lpips is not None else ""
    log(f"Perceptual quality: SSIM {ssim_in:.3f} -> {final_ssim:.3f}{_lp}", "ok")

    fit = compute_fitness(final_psnr, latency_ms, quant_drop,
                          weight_kb=info["total_bytes"] / 1024.0,
                          act_kb=result.est_sram_kb,
                          sram_budget_kb=result.sram_budget_kb,
                          psnr_in=psnr_in,
                          ssim=final_ssim, lpips=final_lpips)
    profile = (f"{cfg.model.model_family.upper()} · {cfg.model.base_channels}ch × "
               f"{cfg.model.block_depth} · {cfg.model.conv_type} · "
               f"{cfg.model.activation} · {meta['precision']}")
    print_report(fit, cfg.hardware_name, profile)

    # Cross-chip suitability: will this exact model run on each Pi-class target?
    assessments = assess_targets(cfg, model, cfg.optimization.quantize,
                                 chosen=cfg.hardware)
    print_target_suitability(assessments, chosen=cfg.hardware)

    # Resolution vs TOPS scaling chart (all Pi-class targets).
    scaling_path = out_dir / "resolution_tops_scaling.png"
    try:
        render_scaling_chart(
            model, scaling_path,
            current_patch=cfg.optimization.patch_size,
            selected_hardware=cfg.hardware,
            show=False)
        log(f"Saved resolution/TOPS scaling chart -> {scaling_path}", "ok")
    except Exception as exc:  # noqa: BLE001
        log(f"Could not render scaling chart: {exc}", "warn")
        scaling_path = None

    # Machine-readable summary so the GUI can render a rich results screen.
    prof = dict(profile_rows(cfg.model))
    summary = {
        "kind": "compile",
        "hardware": cfg.hardware,
        "hardware_name": cfg.hardware_name,
        "model": {
            "family": cfg.model.model_family,
            "base_channels": cfg.model.base_channels,
            "block_depth": cfg.model.block_depth,
            "conv_type": prof.get("conv_type", cfg.model.conv_type),
            "activation": (cfg.model.activation
                             if uses_activation(cfg.model.model_family) else None),
            "nonlinearity": prof.get("nonlinearity"),
            "display": model_display_line(cfg.model),
            "params": n_params,
            "custom_nafnet": custom_naf,
            "nafnet_enc": list(cfg.model.nafnet_enc_blocks),
            "nafnet_middle": cfg.model.nafnet_middle_blocks,
            "nafnet_dec": list(cfg.model.nafnet_dec_blocks),
        },
        "sensor": sensor.label,
        "sensor_key": cfg.sensor.sensor,
        "gain": cfg.sensor.gain,
        "capture_mode": ("real" if real_loaded else "simulated")
                        + (" + simulated noise" if cfg.sensor.simulate_noise else ""),
        "frame_source": str(frame.source) if real_loaded else "synthetic",
        "dataset_path": str(real_source) if real_loaded else None,
        "gt_kind": gt_kind,
        "run_mode": cfg.run.mode,
        "frames": len(frames),
        "quant_scheme": result.quant_scheme,
        "qat": use_qat,
        "temporal_frames_out": n_video,
        "precision": meta["precision"],
        "psnr_in": round(psnr_in, 2),
        "psnr_out": round(final_psnr, 2),
        "psnr_gain": round(final_psnr - psnr_in, 2),
        "ssim_in": round(ssim_in, 4),
        "ssim_out": round(final_ssim, 4),
        "ssim_gain": round(final_ssim - ssim_in, 4),
        "lpips_in": round(lpips_in, 4) if lpips_in is not None else None,
        "lpips_out": round(final_lpips, 4) if final_lpips is not None else None,
        "lpips_gain": (round(lpips_in - final_lpips, 4)
                       if lpips_in is not None and final_lpips is not None else None),
        "quant_drop_db": round(quant_drop, 3),
        "latency_ms": round(latency_ms, 1),
        "fps": round(1000.0 / latency_ms, 1),
        "weight_kb": round(info["total_bytes"] / 1024.0, 1),
        "act_kb": round(result.est_sram_kb, 0),
        "sram_budget_kb": result.sram_budget_kb,
        "fitness": fit.score,
        "grade": fit.grade,
        "warnings": result.warnings,
        "targets": [
            {
                "key": a.key, "label": a.label, "precision": a.precision,
                "format": a.format, "verdict": a.verdict,
                "act_kb": round(a.act_kb, 0), "budget_kb": a.budget_kb,
                "mem_frac": round(a.mem_frac, 4), "tiled": a.tiled,
                "fits": a.fits, "fps": round(a.fps, 1),
                "latency_ms": round(a.latency_ms, 1),
                "act_native": a.act_native, "notes": a.notes,
                "selected": a.key == cfg.hardware,
            }
            for a in assessments
        ],
        "panel": str((out_dir / "validation_panel.png").resolve()),
        "scaling_chart": str(scaling_path.resolve()) if scaling_path else None,
        "scaling": scaling_curves(model) if scaling_path else None,
        "artifacts": [str((out_dir / f).resolve()) for f in (
            "exported_model.onnx", f"hardware_ready{cfg.artifact_ext}",
            "validation_panel.png", "resolution_tops_scaling.png")
            if (out_dir / f).exists()],
    }
    try:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    except Exception:
        pass

    # -- Optional: build a transferable hardware deployment package -----------
    if cfg.output.export:
        try:
            from deploy import build_package
            res = build_package(summary, out_dir, make_zip=True)
            summary["package_dir"] = str(Path(res["pkg"]).resolve())
            summary["package_zip"] = str(Path(res["zip"]).resolve()) if res["zip"] else None
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
            log(f"Exported transferable package -> {res['zip'] or res['pkg']}", "ok")
            log(f"Flash steps in {Path(res['pkg']).name}/FLASH_INSTRUCTIONS.md "
                f"({res['label']})", "info")
        except Exception as exc:
            log(f"Deployment export failed: {exc}", "warn")

    # -- Save this run into the persistent history (so it can be browsed/reused) -
    try:
        from nsa.history import record_run
        rec = record_run(summary, out_dir)
        log(f"Saved to run history -> {Path(rec['dir']).name}  "
            f"(browse past runs in the GUI · HISTORY)", "ok")
    except Exception as exc:  # noqa: BLE001
        log(f"Could not write run history: {exc}", "warn")

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
