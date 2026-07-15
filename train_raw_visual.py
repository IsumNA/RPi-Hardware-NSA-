#!/usr/bin/env python3
"""GPU training on AI for 5-channel RawDenoiser (Phase 2B).

Pipeline per sample:
  burst DNGs → fuse_burst_packed → stack_fusion_input (5ch) → RawDenoiser → 4ch packed

Uses LCG ``cabinet_D50_100`` bursts as a proxy until Pi HCG cache sync completes.
Saves fused-input / denoised / GT panels every ``--panel-every`` steps under
``outputs/raw_panels/`` (same polling workflow as ``train_visual.py``).

Run on the AI server only::

  ssh ai
  cd ~/RPi-Hardware-NSA-
  .venv/bin/python -u train_raw_visual.py --panel-every 50

Refuses to train when Pi HCG sync coverage is below ``--min-sync-pct`` unless
``--force`` or ``--lcg-only`` (default proxy data).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.config import (Config, apply_overrides, build_parser, finalize_dataset_config,
                        load_config, resolve_config_path)
from nsa.inference import build_loss, psnr, ssim, to_tensor, to_image
from nsa.models import build_model, count_params
from nsa.raw_domain import (burst_clean, load_packed, packed_to_rgb,
                            stack_fusion_input)
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed
from nsa.theme import banner, console, kv_table, log

try:
    from nsa.dataset_align import (build_hcg_sort_manifest, cache_readiness,
                                   default_pi_unique_cache, project_json_in_cache)
except ImportError:
    build_hcg_sort_manifest = None  # type: ignore[misc, assignment]
    cache_readiness = None  # type: ignore[misc, assignment]
    default_pi_unique_cache = lambda: Path("/opt/datasets/PI_RAW/Pi_Unique_Cache")  # noqa: E731
    project_json_in_cache = lambda c: Path(c) / "project.json"  # noqa: E731

DISPLAY_GAIN = 8.0
# High-gain / low-light is the product target — skip clean ag1–ag64 folders.
DEFAULT_BURST_SCENE = "cabinet_H_2"
DEFAULT_GAINS = (256, 512)
# GT = large-N burst mean (as clean as the capture allows).
GT_FRAMES = 512
# Network INPUT uses a short window so it must learn real denoising toward the
# 512-frame GT — not "average 12 frames then barely touch residual".
FUSION_FRAMES = 1
TRAIN_STRIDE = 4


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _burst_root(cfg: Config, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    for cand in (
        ROOT / "datasets/imx662_project/bursts",
        Path("/opt/datasets/PI_RAW").parent / "imx662_project/bursts",
        ROOT / "datasets/PI_RAW/imx662_project/bursts",
    ):
        if cand.is_dir():
            return cand
    return ROOT / "datasets/imx662_project/bursts"


def _hcg_sync_fraction() -> tuple[float, dict]:
    """Return HCG manifest coverage fraction (0 if project.json absent)."""
    if cache_readiness is None or build_hcg_sort_manifest is None:
        return 0.0, {"note": "dataset_align unavailable (cv2?)"}
    cache = default_pi_unique_cache()
    pj = project_json_in_cache(cache)
    if not pj.is_file():
        return 0.0, {
            "cache_root": str(cache),
            "project_json": str(pj),
            "status": "project.json missing — sync in progress",
        }
    manifest = build_hcg_sort_manifest(pj)
    report = cache_readiness(cache, manifest)
    return float(report["fraction"]), report


def _list_burst_dirs(bursts_root: Path, scene: str, gains: tuple[int, ...]) -> list[tuple[str, int, Path]]:
    scene_dir = bursts_root / scene
    out: list[tuple[str, int, Path]] = []
    for g in gains:
        for tag in (f"ag{g}",):
            d = scene_dir / tag
            if d.is_dir() and list(d.glob("*.dng")):
                out.append((scene, g, d))
    return out


def _fuse_input_from_burst(
    files: list[Path],
    *,
    end_idx: int,
    fusion_frames: int,
    fusion_cfg: FusionConfig,
) -> np.ndarray:
    """Fuse frames ending at ``end_idx`` → 5-channel model input."""
    start = max(0, end_idx - fusion_frames + 1)
    subset = files[start : end_idx + 1]
    packed = [load_packed(p) for p in subset]
    fused, weight = fuse_burst_packed(packed, fusion_cfg)
    return stack_fusion_input(fused, weight, k_cap=fusion_cfg.k_cap)


def build_raw_pairs(
    burst_dirs: list[tuple[str, int, Path]],
    *,
    gt_frames: int = GT_FRAMES,
    fusion_frames: int = FUSION_FRAMES,
    stride: int = TRAIN_STRIDE,
    fusion_cfg: FusionConfig | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float], dict]:
    """Build (noisy_5ch, clean_4ch) pairs and per-pair sample weights."""
    fusion_cfg = fusion_cfg or FusionConfig(
        n_frames=fusion_frames, k_cap=float(max(fusion_frames, 1)), mode="mean")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    weights: list[float] = []

    ref_scene, ref_gain, ref_dir = burst_dirs[0]
    ref_files = sorted(ref_dir.glob("*.dng"))
    ref_gt = burst_clean(ref_files, limit=min(gt_frames, len(ref_files)))
    ref_end = min(gt_frames, len(ref_files)) - 1
    ref_noisy = _fuse_input_from_burst(
        ref_files, end_idx=ref_end, fusion_frames=fusion_frames, fusion_cfg=fusion_cfg)

    stats: dict = {
        "scenes": [],
        "ref_scene": ref_scene,
        "ref_gain": ref_gain,
        "ref_burst": str(ref_dir),
        "ref_frames": len(ref_files),
    }

    for scene, gain, bdir in burst_dirs:
        files = sorted(bdir.glob("*.dng"))
        n = len(files)
        if n < fusion_frames + 4:
            log(f"skip {scene}/ag{gain}: only {n} frames", "warn")
            continue
        gt = burst_clean(files, limit=min(gt_frames, n))
        limit = min(gt_frames, n)
        n_pairs = 0
        for end_idx in range(fusion_frames - 1, limit, stride):
            noisy = _fuse_input_from_burst(
                files, end_idx=end_idx, fusion_frames=fusion_frames, fusion_cfg=fusion_cfg)
            h = min(noisy.shape[0], gt.shape[0])
            w = min(noisy.shape[1], gt.shape[1])
            pairs.append((noisy[:h, :w], gt[:h, :w]))
            # Heavily oversample ag512 vs ag256 (log2 gain).
            weights.append(max(1.0, math.log2(max(gain, 1)) ** 2))
            n_pairs += 1
        stats["scenes"].append({"scene": scene, "gain": gain, "frames": n, "pairs": n_pairs})
        log(f"  {scene}/ag{gain}: {n} DNGs → {n_pairs} fusion pairs", "ok")

    return pairs, weights, {
        **stats,
        "ref_noisy": ref_noisy,
        "ref_gt": ref_gt,
        "total_pairs": len(pairs),
        "gt_frames": gt_frames,
        "fusion_frames": fusion_frames,
    }


def _save_raw_panel(
    model: nn.Module,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    dev: torch.device,
    panel_dir: Path,
    step: int,
    meta: dict,
    loss: float,
    display_gain: float = DISPLAY_GAIN,
) -> None:
    from nsa.visualize import render_panel

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(ref_x.to(dev))
    fused = to_image(ref_x)[..., :4]
    clean = to_image(ref_y)
    denoised = to_image(out.cpu())
    noisy_rgb = packed_to_rgb(fused, display_gain)
    clean_rgb = packed_to_rgb(clean, display_gain)
    denoised_rgb = packed_to_rgb(denoised, display_gain)
    pmeta = dict(meta)
    pmeta["psnr_in"] = psnr(noisy_rgb, clean_rgb)
    pmeta["psnr_out"] = psnr(denoised_rgb, clean_rgb)
    pmeta["step"] = step
    pmeta["loss"] = loss
    pmeta["domain"] = "packed_raw_5ch"
    path = panel_dir / f"step_{step:05d}.png"
    render_panel(noisy_rgb, clean_rgb, denoised_rgb, pmeta, path, show=False)
    latest = panel_dir / "latest.png"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        shutil.copy2(path, panel_dir / "latest.png")
    if was_training:
        model.train()


def _train_raw(
    model: nn.Module,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    seed: int,
    progress,
    *,
    crop: int,
    batch: int,
    lr: float,
    loss_fn,
    weights: list[float] | None,
    device: torch.device,
    panel_every: int,
    panel_ref: tuple[np.ndarray, np.ndarray] | None,
    panel_dir: Path | None,
    panel_meta: dict | None,
) -> nn.Module:
    """Training loop mirroring inference._train with raw-domain panels."""
    from nsa.inference import _augment_pair, _sample_batch

    tensors = [(to_tensor(n), to_tensor(c)) for n, c in pairs]
    wt = None
    if weights is not None and len(weights) == len(tensors):
        wt = torch.as_tensor(weights, dtype=torch.float32).clamp(min=1e-6)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    steps = max(1, steps)
    warmup = max(1, steps // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model = model.to(device)
    model.train()

    ref_x = ref_y = None
    if panel_ref is not None:
        ref_x, ref_y = to_tensor(panel_ref[0]), to_tensor(panel_ref[1])
    if panel_every > 0 and ref_x is not None and panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)

    for i in range(steps):
        xb, yb = _sample_batch(tensors, crop, batch, g, weights=wt)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if progress is not None and (i % 4 == 0 or i == steps - 1):
            progress(i + 1, steps, float(loss.item()))
        if (panel_every > 0 and ref_x is not None and panel_dir is not None
                and panel_meta is not None
                and ((i + 1) % panel_every == 0 or i == steps - 1)):
            _save_raw_panel(model, ref_x, ref_y, device, panel_dir, i + 1,
                            panel_meta, float(loss.item()))
    model.eval()
    return model


def _panel_meta(cfg: Config, dataset_info: dict, psnr_in: float, psnr_out: float) -> dict:
    return {
        "sensor": cfg.sensor.sensor,
        "gain": dataset_info.get("ref_gain", cfg.sensor.gain),
        "frames": dataset_info.get("fusion_frames", FUSION_FRAMES),
        "real_capture": True,
        "gt_kind": "packed_burst_mean",
        "frame_source": dataset_info.get("ref_burst", ""),
        "family": "raw_denoiser",
        "precision": "FP32",
        "hardware_name": cfg.hardware_name,
        "burst_scene": dataset_info.get("ref_scene", DEFAULT_BURST_SCENE),
        "psnr_in": psnr_in,
        "psnr_out": psnr_out,
    }


def main() -> int:
    ap = build_parser()
    ap.description = "Train 5ch RawDenoiser with frequent validation panels (AI GPU)"
    ap.add_argument("--panel-every", type=int, default=50,
                    help="save validation panel every N steps (default 50)")
    ap.add_argument("--panel-dir", default="outputs/raw_panels")
    ap.add_argument("--burst-scene", default=DEFAULT_BURST_SCENE,
                    help=f"LCG/HCG burst scene folder (default {DEFAULT_BURST_SCENE})")
    ap.add_argument("--burst-root", default="",
                    help="override bursts/ root (default datasets/imx662_project/bursts)")
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS),
                    help="comma-separated ag gains to train on (default 256,512)")
    ap.add_argument("--gt-frames", type=int, default=GT_FRAMES,
                    help="frames averaged for clean GT (default 512)")
    ap.add_argument("--fusion-frames", type=int, default=FUSION_FRAMES,
                    help="frames fused for model INPUT (default 1 = single noisy)")
    ap.add_argument("--cal-steps", type=int, default=None)
    ap.add_argument("--ext-steps", type=int, default=None)
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument("--min-sync-pct", type=float, default=50.0,
                    help="abort training if HCG Pi sync below this %% (default 50)")
    ap.add_argument("--force", action="store_true",
                    help="train even when HCG sync is below --min-sync-pct")
    ap.add_argument("--readiness-only", action="store_true",
                    help="report sync %% and dataset stats, then exit")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(resolve_config_path(args.config, ROOT)), args)
    cfg.model.model_family = "raw_denoiser"
    finalize_dataset_config(cfg, ROOT)

    sync_frac, sync_report = _hcg_sync_fraction()
    sync_pct = 100.0 * sync_frac
    banner(f"Raw visual training  ·  HCG sync {sync_pct:.1f}%")

    console.print(kv_table([
        ("HCG sync", f"{sync_pct:.1f}%"),
        ("project.json", sync_report.get("project_json", sync_report.get("cache_root", "?"))),
        ("present/wanted", f"{sync_report.get('present_files', '?')}/{sync_report.get('wanted_files', '?')}"),
        ("proxy scene", args.burst_scene),
    ], title="Dataset readiness"))

    if args.readiness_only:
        burst_dirs = _list_burst_dirs(
            _burst_root(cfg, args.burst_root or None), args.burst_scene,
            tuple(int(x.strip()) for x in args.gains.split(",") if x.strip()))
        console.print(kv_table([
            ("LCG bursts", str(len(burst_dirs))),
            ("burst root", str(_burst_root(cfg, args.burst_root or None))),
        ], title="LCG proxy (cabinet_D50_100 until HCG aligned)"))
        return 0

    if sync_pct < args.min_sync_pct and not args.force:
        log(f"HCG sync {sync_pct:.1f}% < {args.min_sync_pct:.0f}% — script deployed, "
            f"training deferred (re-run after sync or pass --force)", "warn")
        burst_dirs = _list_burst_dirs(
            _burst_root(cfg, args.burst_root or None), args.burst_scene,
            tuple(int(x.strip()) for x in args.gains.split(",") if x.strip()))
        if burst_dirs:
            log(f"LCG proxy ready: {len(burst_dirs)} burst folder(s) under "
                f"{args.burst_scene}", "info")
        return 2

    if sync_pct < 100.0:
        log(f"HCG sync {sync_pct:.1f}% — training on LCG proxy ({args.burst_scene}) "
            f"until HCG bursts are aligned", "warn")

    dev = _device()
    log(f"Device: {dev}", "step")

    gains = tuple(int(x.strip()) for x in args.gains.split(",") if x.strip())
    bursts_root = _burst_root(cfg, args.burst_root or None)
    burst_dirs = _list_burst_dirs(bursts_root, args.burst_scene, gains)
    if not burst_dirs:
        log(f"No burst dirs under {bursts_root}/{args.burst_scene} for gains {gains}", "err")
        return 1

    gt_frames = int(args.gt_frames)
    fusion_frames = int(args.fusion_frames)
    log(f"Recipe: INPUT={fusion_frames}-frame mean → TARGET={gt_frames}-frame GT  "
        f"gains={gains}  scene={args.burst_scene}", "step")
    fusion_cfg = FusionConfig(
        n_frames=fusion_frames, k_cap=float(max(fusion_frames, 1)), mode="mean")
    pairs, sample_w, ds_info = build_raw_pairs(
        burst_dirs, fusion_cfg=fusion_cfg,
        gt_frames=gt_frames, fusion_frames=fusion_frames, stride=TRAIN_STRIDE)
    if not pairs:
        log("No training pairs built — check burst directories", "err")
        return 1
    log(f"Training pairs: {len(pairs)}", "ok")

    model = build_model(cfg.model)
    n_params = count_params(model)
    log(f"RawDenoiser 5→4  {n_params:,} params  "
        f"({cfg.model.base_channels}ch × {cfg.model.block_depth} blocks)", "ok")

    lc = cfg.optimization.loss
    loss_fn = build_loss(
        lc.name, charbonnier_eps=lc.charbonnier_eps, huber_delta=lc.huber_delta,
        ssim_window=lc.ssim_window, ssim_weight=lc.ssim_weight, weights=lc.weights)

    ref_noisy = ds_info["ref_noisy"]
    ref_gt = ds_info["ref_gt"]
    ref_rgb_in = packed_to_rgb(ref_noisy[..., :4], DISPLAY_GAIN)
    ref_rgb_gt = packed_to_rgb(ref_gt, DISPLAY_GAIN)
    psnr_in = psnr(ref_rgb_in, ref_rgb_gt)
    pmeta = _panel_meta(cfg, ds_info, psnr_in, psnr_in)

    panel_dir = Path(args.panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_ref = (ref_noisy, ref_gt)

    cal_steps = args.cal_steps or cfg.optimization.calibration_steps
    log(f"Calibration: {cal_steps} steps  panel every {args.panel_every}", "step")

    def on_step(i, total, loss):
        if i % 20 == 0 or i == total:
            log(f"  cal {i}/{total}  loss={loss:.4f}", "info")

    t0 = time.time()
    _train_raw(
        model, pairs, cal_steps, cfg.output.seed, on_step,
        crop=cfg.optimization.patch_size, batch=6, lr=3e-3, loss_fn=loss_fn,
        weights=sample_w, device=dev, panel_every=args.panel_every,
        panel_ref=panel_ref, panel_dir=panel_dir, panel_meta=pmeta)
    log(f"Calibration done in {(time.time() - t0) / 60:.1f} min", "ok")

    ext_steps = 0 if args.no_extended else (args.ext_steps or cfg.optimization.extended_steps)
    if ext_steps > 0 and cfg.optimization.extended_train:
        log(f"Extended: {ext_steps} steps on same fusion pairs", "step")

        def on_ext(i, total, loss):
            if i % 50 == 0 or i == total:
                log(f"  ext {i}/{total}  loss={loss:.4f}", "info")

        t1 = time.time()
        _train_raw(
            model, pairs, ext_steps, cfg.output.seed + 1, on_ext,
            crop=cfg.optimization.patch_size, batch=6, lr=3e-3, loss_fn=loss_fn,
            weights=sample_w, device=dev, panel_every=args.panel_every,
            panel_ref=panel_ref, panel_dir=panel_dir, panel_meta=pmeta)
        log(f"Extended done in {(time.time() - t1) / 60:.1f} min", "ok")

    with torch.no_grad():
        out_t = model(to_tensor(ref_noisy).to(dev))
    denoised = to_image(out_t.cpu())
    psnr_out = psnr(packed_to_rgb(denoised, DISPLAY_GAIN), ref_rgb_gt)
    ssim_out = ssim(packed_to_rgb(denoised, DISPLAY_GAIN), ref_rgb_gt)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from nsa.visualize import render_panel

    final_panel = out_dir / "raw_validation_panel.png"
    fmeta = _panel_meta(cfg, ds_info, psnr_in, psnr_out)
    render_panel(ref_rgb_in, ref_rgb_gt, packed_to_rgb(denoised, DISPLAY_GAIN),
                 fmeta, final_panel, show=False)
    shutil.copy2(final_panel, panel_dir / "final.png")

    ckpt = out_dir / "raw_denoiser_5ch.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": {
            "family": "raw_denoiser",
            "base_channels": cfg.model.base_channels,
            "block_depth": cfg.model.block_depth,
            "in_ch": 5,
            "out_ch": 4,
        },
        "burst_scene": args.burst_scene,
        "gains": list(gains),
        "hcg_sync_pct": sync_pct,
        "psnr_out": psnr_out,
    }, ckpt)

    summary = {
        "psnr_in": psnr_in,
        "psnr_out": psnr_out,
        "ssim_out": ssim_out,
        "panels": str(panel_dir.resolve()),
        "device": str(dev),
        "params": n_params,
        "pairs": len(pairs),
        "burst_scene": args.burst_scene,
        "gains": list(gains),
        "hcg_sync_pct": sync_pct,
        "lcg_proxy": sync_pct < 100.0,
        "scenes": ds_info.get("scenes", []),
    }
    (out_dir / "train_raw_visual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Final PSNR {psnr_out:.2f} dB  SSIM {ssim_out:.3f}", "ok")
    log(f"Panel gallery: {panel_dir}", "ok")
    log(f"Checkpoint: {ckpt}", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
