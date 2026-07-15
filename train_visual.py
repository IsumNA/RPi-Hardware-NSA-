#!/usr/bin/env python3
"""GPU-friendly training on the AI server with frequent validation panels.

Runs calibration + extended training from config.yaml, saving a 3-panel
noisy / GT / denoised image every ``--panel-every`` steps under
``outputs/panels/``. Intended to be launched on ``ssh ai`` while the laptop
polls panels via ``scripts/watch_ai_panels.sh``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.config import (Config, apply_overrides, build_parser, finalize_dataset_config,
                        load_config, project_root, resolve_config_path)
from nsa.inference import build_loss, calibrate_multi, psnr, run, ssim
from nsa.models import build_model, count_params
from nsa.raw_io import (build_frame_from_source, list_frames, load_training_pairs,
                        training_sample_weights)
from nsa.sensors import get_sensor, with_noise_std
from nsa.theme import banner, log
from nsa.visualize import render_panel

try:
    from nsa.inference import lpips as lpips_metric
except Exception:
    lpips_metric = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _panel_meta(cfg: Config, frame, psnr_in: float, psnr_out: float) -> dict:
    return {
        "sensor": get_sensor(cfg.sensor.sensor).label,
        "gain": cfg.sensor.gain,
        "frames": cfg.data.temporal_frames,
        "real_capture": True,
        "gt_kind": "paired",
        "frame_source": str(frame.source),
        "family": cfg.model.model_family,
        "precision": "FP32",
        "hardware_name": cfg.hardware_name,
        "psnr_in": psnr_in,
        "psnr_out": psnr_out,
    }


def main() -> int:
    ap = build_parser()
    ap.description = "Train with frequent validation panels"
    ap.add_argument("--panel-every", type=int, default=100,
                    help="save validation panel every N steps (default 100)")
    ap.add_argument("--panel-dir", default="outputs/panels")
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument("--cal-steps", type=int, default=None)
    ap.add_argument("--ext-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(resolve_config_path(args.config, ROOT)), args)
    finalize_dataset_config(cfg, ROOT)
    dev = _device()
    banner(f"Visual training on {dev}")

    sensor = with_noise_std(get_sensor(cfg.sensor.sensor), cfg.sensor.noise_std)
    real_source = Path(cfg.sensor.dataset_path or "datasets/PI_RAW")
    filter_tokens = list(cfg.sensor.filter or [])

    frames = []
    sources = list_frames(str(real_source), filter_tokens)
    if not sources:
        log("No frames matched filter — aborting", "err")
        return 1
    log(f"Dataset filter: {filter_tokens}  ({len(sources)} frames)", "step")
    sources = sources[:1]
    for i, s in enumerate(sources):
        frames.append(build_frame_from_source(
            s, cfg.sensor.gain, cfg.data.temporal_frames,
            cfg.optimization.patch_size, sensor,
            cfg.output.seed + i, simulate_noise=False))
    frame = frames[0]
    log(f"Validation frame: {frame.source}  ({frame.width}×{frame.height})", "ok")
    log(f"Dataset: {real_source}  filter={filter_tokens}", "info")

    model = build_model(cfg.model)
    n_params = count_params(model)
    log(f"{cfg.model.model_family}  {n_params:,} params", "ok")

    lc = cfg.optimization.loss
    loss_fn = build_loss(
        lc.name, charbonnier_eps=lc.charbonnier_eps, huber_delta=lc.huber_delta,
        ssim_window=lc.ssim_window, ssim_weight=lc.ssim_weight, weights=lc.weights)

    panel_dir = Path(args.panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    psnr_in = psnr(frame.noisy_rgb, frame.clean_rgb)
    ref = (frame.noisy_rgb, frame.clean_rgb)
    pmeta = _panel_meta(cfg, frame, psnr_in, psnr_in)

    cal_steps = args.cal_steps or cfg.optimization.calibration_steps
    pairs_cal = [(f.noisy_rgb, f.clean_rgb) for f in frames]
    cal_w = training_sample_weights(
        [f.source for f in frames], [f.clean_rgb for f in frames],
        gain_exp=cfg.optimization.gain_emphasis,
        dark_emphasis=cfg.optimization.dark_emphasis) if len(frames) > 1 else None

    log(f"Calibration: {cal_steps} steps  panel every {args.panel_every}", "step")

    def on_step(i, total, loss):
        if i % 20 == 0 or i == total:
            log(f"  cal {i}/{total}  loss={loss:.4f}", "info")

    t0 = time.time()
    calibrate_multi(
        model, pairs_cal, cal_steps, cfg.output.seed, on_step,
        crop=cfg.optimization.patch_size, batch=6, loss_fn=loss_fn,
        weights=cal_w, device=dev, panel_every=args.panel_every,
        panel_ref=ref, panel_dir=panel_dir, panel_meta=pmeta)
    log(f"Calibration done in {(time.time()-t0)/60:.1f} min", "ok")

    ext_steps = 0 if args.no_extended else (args.ext_steps or cfg.optimization.extended_steps)
    if ext_steps > 0 and cfg.optimization.extended_train:
        ext_named = load_training_pairs(
            real_source, filter_tokens or None, sensor=sensor,
            gain=cfg.sensor.gain, simulate_noise=False,
            seed=cfg.output.seed, temporal_frames=cfg.data.temporal_frames,
            max_side=int(cfg.optimization.extended_max_side),
            with_names=True,
            tile=int(cfg.optimization.extended_tile),
            tiles_per_image=int(cfg.optimization.extended_tiles_per_image))
        ext_pairs = [(n, c) for _, n, c in ext_named]
        ext_w = training_sample_weights(
            [name for name, _, _ in ext_named],
            [c for _, _, c in ext_named],
            gain_exp=cfg.optimization.gain_emphasis,
            dark_emphasis=cfg.optimization.dark_emphasis)
        log(f"Extended: {len(ext_pairs)} tiles  {ext_steps} steps", "step")

        def on_ext(i, total, loss):
            if i % 50 == 0 or i == total:
                log(f"  ext {i}/{total}  loss={loss:.4f}", "info")

        t1 = time.time()
        calibrate_multi(
            model, ext_pairs, ext_steps, cfg.output.seed + 1, on_ext,
            crop=cfg.optimization.patch_size, batch=6, loss_fn=loss_fn,
            weights=ext_w, device=dev, panel_every=args.panel_every,
            panel_ref=ref, panel_dir=panel_dir, panel_meta=pmeta)
        log(f"Extended done in {(time.time()-t1)/60:.1f} min", "ok")

    out, _ = run(model.cpu(), frame.noisy_rgb)
    model.to(dev)
    psnr_out = psnr(out, frame.clean_rgb)
    ssim_out = ssim(out, frame.clean_rgb)
    lp = None
    if lpips_metric:
        try:
            lp = lpips_metric(out, frame.clean_rgb)
        except Exception:
            pass

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_panel = out_dir / "validation_panel.png"
    fmeta = _panel_meta(cfg, frame, psnr_in, psnr_out)
    render_panel(frame.noisy_rgb, frame.clean_rgb, out, fmeta, final_panel, show=False)
    shutil_copy = panel_dir / "final.png"
    import shutil
    shutil.copy2(final_panel, shutil_copy)

    ckpt = out_dir / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": {
            "family": cfg.model.model_family,
            "base_channels": cfg.model.base_channels,
            "block_depth": cfg.model.block_depth,
            "conv_type": cfg.model.conv_type,
            "activation": cfg.model.activation,
            "nafnet_enc": list(cfg.model.nafnet_enc_blocks),
            "nafnet_middle": cfg.model.nafnet_middle_blocks,
            "nafnet_dec": list(cfg.model.nafnet_dec_blocks),
        },
        "hardware": cfg.hardware,
        "psnr_out": psnr_out,
    }, ckpt)

    summary = {
        "psnr_in": psnr_in, "psnr_out": psnr_out, "ssim_out": ssim_out,
        "lpips_out": lp, "panels": str(panel_dir.resolve()),
        "device": str(dev), "params": n_params,
    }
    (out_dir / "train_visual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Final PSNR {psnr_out:.2f} dB  SSIM {ssim_out:.3f}"
        + (f"  LPIPS {lp:.3f}" if lp is not None else ""), "ok")
    log(f"Panel gallery: {panel_dir}", "ok")
    log(f"Final panel: {final_panel}", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
