#!/usr/bin/env python3
"""Train: live noisy frame(s) → burst-averaged GT. Deploy that model on the Pi.

This is NOT "average 500 frames at inference". GT was already built from your
bursts. Training teaches a network to map a *single stream frame* (or a short
live window of T frames stacked on channels) toward that GT so the Pi can
denoise in real time.

Run on AI GPU::

  .venv/bin/python -u train_stream_to_gt.py \\
      --gains 128,256,512 --steps 16000 --channels 128 --depth 8 \\
      --stride 2 --temporal 4
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
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.inference import build_loss, psnr, ssim, to_image, to_tensor
from nsa.raw_domain import (
    RawDenoiser,
    burst_clean,
    load_or_build_alpha_gt,
    load_packed,
    packed_to_rgb,
)

DISPLAY_GAIN = 8.0
# Prefer long bursts; colour_stripes / hcg included when present on disk.
DEFAULT_SCENES = (
    "cabinet_H_2", "cabinet_H_10", "cabinet_F11_25",
    "cabinet_D50_100", "cabinet_D_10", "cabinet_F_5",
    "colour_stripes", "cabinet_H_2_hcg",
)
DEFAULT_GAINS = (128, 256, 512)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_frames(frames: list[np.ndarray], idx: int, temporal: int) -> np.ndarray:
    """Channel-stack ``temporal`` packed frames ending at ``idx`` (current last→first).

    Layout: [current, t-1, …, t-(T-1)] along C → shape H×W×(4T).
    RawDenoiser residual is applied to channels 0:4 (the live current frame).
    """
    parts = []
    for k in range(temporal):
        j = max(0, idx - k)
        parts.append(frames[j])
    return np.concatenate(parts, axis=-1)


def build_pairs(
    bursts_root: Path,
    scenes: tuple[str, ...],
    gains: tuple[int, ...],
    *,
    gt_frames: int,
    stride: int,
    holdout_start: int,
    temporal: int,
    gt_mode: str = "mean",
    gt_cache_root: Path | None = None,
    alpha_trim: float = 0.25,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict], dict]:
    """Each live window → packed GT (temporal mean or alpha-trim of burst).

    ``gt_mode``: ``mean`` (legacy ``burst_clean``) or ``alpha_trim`` (cached
    under ``gt_cache_root``, default ``datasets/imx662_project/gt_alpha16``).

    ``meta["pair_gains"]`` parallels ``pairs`` with the folder analogue gain
    (ag128/256/512) for FiLM / gain-channel conditioning.
    """
    mode = (gt_mode or "mean").strip().lower()
    if mode not in ("mean", "alpha_trim"):
        raise ValueError(f"unknown gt_mode={gt_mode!r}")
    if gt_cache_root is None:
        gt_cache_root = bursts_root.parent / "gt_alpha16"
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pair_gains: list[int] = []
    evals: list[dict] = []
    stats: list[dict] = []
    T = max(1, int(temporal))

    for scene in scenes:
        for g in gains:
            bdir = bursts_root / scene / f"ag{g}"
            files = sorted(bdir.glob("*.dng"))
            if len(files) < 32:
                print(f"  skip {scene}/ag{g}: {len(files)} DNGs", flush=True)
                continue
            n = len(files)
            n_gt = min(gt_frames, n)
            packed = [load_packed(f) for f in files]
            if mode == "alpha_trim":
                gt = load_or_build_alpha_gt(
                    bursts_root, gt_cache_root, scene, g,
                    limit=n_gt, trim=alpha_trim)
            else:
                gt = burst_clean(files, limit=n_gt)
            train_idx = list(range(0, n_gt, stride))
            for i in train_idx:
                pairs.append((_stack_frames(packed, i, T), gt))
                pair_gains.append(int(g))
            # Held-out frames outside the GT window when possible
            span = max(1, (n - holdout_start) // 3)
            hold = [i for i in range(holdout_start, n, span) if i < n][:3]
            if not hold:
                hold = [n_gt - 1]
            evals.append({
                "scene": scene, "gain": g,
                "gt": gt,
                "noisy": [(i, _stack_frames(packed, i, T)) for i in hold],
            })
            stats.append({
                "scene": scene, "gain": g, "frames": n,
                "train": len(train_idx), "holdout": hold, "temporal": T,
                "gt_mode": mode,
            })
            print(f"  {scene}/ag{g}: {n} DNGs → {len(train_idx)} train "
                  f"(T={T}), holdout {hold}", flush=True)
            del packed  # free before next burst
    return pairs, evals, {
        "scenes": stats, "total_pairs": len(pairs),
        "temporal": T, "pair_gains": pair_gains,
        "gt_mode": mode, "gt_cache_root": str(gt_cache_root),
    }


def _train(
    model: nn.Module,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    loss_fn,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    best_path: Path | None = None,
    start_step: int = 0,
    best_psnr: float = -1.0,
    last_path: Path | None = None,
) -> nn.Module:
    from nsa.inference import _sample_batch

    tensors = [(to_tensor(n), to_tensor(c)) for n, c in pairs]
    # Emphasise darker / higher-noise samples via mean intensity of current frame
    wts = torch.tensor(
        [1.0 / max(float(n[..., :4].mean()), 1e-3) for n, _ in pairs],
        dtype=torch.float32)
    wts = (wts / wts.mean()).clamp(0.5, 4.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)  # shorter warmup; peak LR less aggressive

    def lr_at(epoch: int) -> float:
        # LambdaLR calls this with the number of sched.step() calls so far.
        # Align with absolute training step so resume keeps the cosine phase.
        i = start_step + epoch
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(662 + max(0, start_step))
    model = model.to(device)
    model.train()
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_state: dict | None = (
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_psnr > 0 else None
    )
    skipped = 0
    done = 0

    if start_step > 0:
        print(f"  resume from step {start_step} → {steps} "
              f"(best_score={best_psnr:.2f})", flush=True)
    if start_step >= steps:
        print(f"  start_step {start_step} >= steps {steps}; nothing to train",
              flush=True)
        model.eval()
        return model

    for i in range(start_step, steps):
        xb, yb = _sample_batch(tensors, crop, batch, g, weights=wts)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if not torch.isfinite(loss):
            skipped += 1
            if skipped <= 5 or skipped % 50 == 0:
                print(f"  skip non-finite loss at step {i+1} "
                      f"(total skipped {skipped})", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()
        done += 1
        if i % 50 == 0 or i == steps - 1:
            elapsed = max(done, 1)
            print(f"  step {i+1}/{steps}  loss={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/elapsed:.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            panel_ev = next((e for e in evals if e.get('gain', 0) >= 256), evals[0])
            pout = _save_eval_panel(model, panel_ev, device, panel_dir, i + 1)
            # Quick multi-scene probe (first holdout frame of up to 6 evals)
            probe = _quick_psnr(model, evals[:6], device)
            print(f"  probe mean PSNR (n={min(6, len(evals))}): "
                  f"{probe:.2f} dB", flush=True)
            score = 0.5 * pout + 0.5 * probe
            if last_path is not None:
                torch.save({"state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()},
                    "step": i + 1, "score": score,
                    "panel_psnr": pout, "probe_psnr": probe}, last_path)
            if score > best_psnr:
                best_psnr = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                if best_path is not None:
                    torch.save({"state_dict": best_state,
                                "step": i + 1, "score": best_psnr,
                                "panel_psnr": pout, "probe_psnr": probe},
                               best_path)
                print(f"  ★ best@{i+1}: panel={pout:.2f} probe={probe:.2f} "
                      f"score={best_psnr:.2f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (score={best_psnr:.2f} dB)", flush=True)
    if skipped:
        print(f"Skipped {skipped} non-finite steps", flush=True)
    model.eval()
    return model


def _load_resume(path: Path) -> tuple[dict, int, float]:
    """Load checkpoint weights + infer completed step / best score."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob
    step = int(blob.get("step") or blob.get("best_step") or 0)
    score = float(blob.get("score") or blob.get("best_score") or -1.0)
    return state, step, score


@torch.no_grad()
def _quick_psnr(model, evals: list[dict], device) -> float:
    model.eval()
    vals = []
    for ev in evals:
        idx, noisy = ev["noisy"][0]
        gt = ev["gt"]
        out = to_image(model(to_tensor(noisy).to(device)).cpu())
        vals.append(psnr(_rgb(out), _rgb(gt)))
    model.train()
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def _save_eval_panel(model, ev: dict, device, panel_dir: Path, step: int) -> float:
    was_training = model.training
    model.eval()
    idx, noisy = ev["noisy"][0]
    gt = ev["gt"]
    out = to_image(model(to_tensor(noisy).to(device)).cpu())
    # Visualise current frame only (first 4 packed channels)
    nr, gr, or_ = (_rgb(noisy[..., :4]), _rgb(gt), _rgb(out))
    pin, pout = psnr(nr, gr), psnr(or_, gr)
    strip = np.concatenate([nr, or_, gr], axis=1)
    img = (np.clip(strip, 0, 1) * 255 + 0.5).astype(np.uint8)
    path = panel_dir / f"step_{step:05d}.png"
    Image.fromarray(img).save(path)
    latest = panel_dir / "latest.png"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        shutil.copy2(path, latest)
    print(f"  panel step {step}: {ev['scene']}/ag{ev['gain']} "
          f"frame {idx}  {pin:.1f}→{pout:.1f} dB", flush=True)
    if was_training:
        model.train()
    return float(pout)


def _rgb(pk: np.ndarray) -> np.ndarray:
    return packed_to_rgb(pk, DISPLAY_GAIN)


@torch.no_grad()
def evaluate(model, evals, device) -> list[dict]:
    rows = []
    model.eval()
    for ev in evals:
        gt = ev["gt"]
        gr = _rgb(gt)
        for idx, noisy in ev["noisy"]:
            out = to_image(model(to_tensor(noisy).to(device)).cpu())
            nr, or_ = _rgb(noisy[..., :4]), _rgb(out)
            rows.append({
                "scene": ev["scene"], "gain": ev["gain"], "frame": idx,
                "psnr_in": round(psnr(nr, gr), 2),
                "psnr_out": round(psnr(or_, gr), 2),
                "ssim_in": round(ssim(nr, gr), 4),
                "ssim_out": round(ssim(or_, gr), 4),
            })
    return rows


def _export_onnx(model: nn.Module, in_ch: int, out_path: Path, patch: int = 256) -> Path:
    model = model.cpu().eval()
    dummy = torch.randn(1, in_ch, patch, patch)
    kwargs = dict(
        input_names=["packed"],
        output_names=["packed_denoised"],
        opset_version=18,
        dynamic_axes={
            "packed": {2: "h", 3: "w"},
            "packed_denoised": {2: "h", 3: "w"},
        },
    )
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    return out_path


def _make_loss():
    """Prefer charbonnier+edge+swt on 4ch; fall back if a term breaks."""
    candidates = (
        ("charbonnier+edge+swt",
         {"charbonnier": 1.0, "edge": 0.12, "swt": 0.15}),
        ("charbonnier+swt",
         {"charbonnier": 1.0, "swt": 0.15}),
        ("charbonnier+edge",
         {"charbonnier": 1.0, "edge": 0.15}),
    )
    probe = torch.zeros(1, 4, 32, 32)
    for name, weights in candidates:
        try:
            fn = build_loss(name, charbonnier_eps=5e-4, weights=weights)
            fn(probe, probe)
            print(f"Loss: {name}", flush=True)
            return fn, name
        except Exception as e:
            print(f"  loss probe {name} failed ({e})", flush=True)
    raise RuntimeError("no usable loss")


def _synth_pairs(
    manifest: Path,
    models_dir: Path,
    *,
    n: int,
    crop: int,
    temporal: int,
    seed: int,
    sensors: tuple[str, ...],
    gains: tuple[int, ...],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict], list[int]]:
    """Materialize ``n`` synthetic pairs from a clean-image manifest + per-gain models.

    Returns (pairs, evals, pair_gains) matching ``build_pairs``'s outputs so the
    rest of the training pipeline is unchanged.
    """
    from nsa.synth.dataset import SynthPairDataset
    from nsa.synth.noise import load_gain_model

    models = {}
    for s in sensors:
        for g in gains:
            path = models_dir / f"{s}_ag{g}.json"
            if path.exists():
                models[(s, int(g))] = load_gain_model(path)
    if not models:
        raise FileNotFoundError(
            f"No noise models found under {models_dir} for {sensors} × {gains}")

    ds = SynthPairDataset(
        manifest=manifest, models=models,
        crop=crop, temporal=temporal, length=n, seed=seed,
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pair_gains: list[int] = []
    evals: list[dict] = []
    # take a handful of pairs aside as held-out eval visualization
    n_eval = min(6, max(1, n // 100))
    for i in range(n):
        item = ds[i]
        pairs.append((item["noisy"], item["gt"]))
        pair_gains.append(int(item["gain"]))
        if i < n_eval:
            evals.append({
                "scene": f"synth_{item['sensor']}_ag{item['gain']}",
                "gain": int(item["gain"]),
                "gt": item["gt"],
                "noisy": [(0, item["noisy"])],
            })
    return pairs, evals, pair_gains


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    # Synthetic-pair options (opt-in; real bursts remain the default source)
    ap.add_argument("--synth-manifest", type=Path, default=None,
                    help="clean_manifest.json from build_synth_dataset.py; "
                         "enables synthetic pair generation")
    ap.add_argument("--synth-models-dir", type=Path,
                    default=ROOT / "models/noise",
                    help="dir with imx662{,h}_ag<g>.json noise models")
    ap.add_argument("--synth-n", type=int, default=1024,
                    help="how many synthetic pairs to materialise")
    ap.add_argument("--synth-crop", type=int, default=384,
                    help="patch size (packed pixels) for synthetic pairs")
    ap.add_argument("--synth-sensors", default="imx662",
                    help="comma-separated sensor tags (imx662, imx662h)")
    ap.add_argument("--synth-only", action="store_true",
                    help="skip real bursts entirely and train only on synth")
    ap.add_argument("--synth-seed", type=int, default=662)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--temporal", type=int, default=4,
                    help="live frames stacked on channels (1 = single-frame)")
    ap.add_argument("--crop", type=int, default=256)
    # 128ch×8 + T=4 + SWT needs ~10GB at batch=2/crop=256 on 16GB GPUs
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--panel-every", type=int, default=200)
    ap.add_argument("--panel-dir", type=Path, default=ROOT / "outputs/stream_gt_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--resume", type=Path, default=None,
                    help="checkpoint to continue from (weights + step/score)")
    ap.add_argument("--start-step", type=int, default=None,
                    help="override resume step (default: ckpt step/best_step)")
    args = ap.parse_args()

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    in_ch = 4 * temporal
    out_ch = 4
    dev = _device()
    print(f"Device {dev}  recipe: STREAM×{temporal} → {args.gt_frames}-frame GT",
          flush=True)
    print(f"Scenes {scenes}  gains {gains}", flush=True)

    pairs: list = []
    evals: list = []
    meta: dict = {"scenes": [], "total_pairs": 0, "temporal": temporal,
                  "pair_gains": [], "gt_mode": "mean", "gt_cache_root": ""}

    if not args.synth_only:
        pairs, evals, meta = build_pairs(
            args.bursts, scenes, gains,
            gt_frames=args.gt_frames, stride=args.stride,
            holdout_start=args.holdout_start, temporal=temporal)
        print(f"Real burst pairs: {len(pairs)}", flush=True)

    if args.synth_manifest is not None:
        sensors_tag = tuple(s.strip() for s in args.synth_sensors.split(",") if s.strip())
        s_pairs, s_evals, s_gains = _synth_pairs(
            args.synth_manifest, args.synth_models_dir,
            n=int(args.synth_n), crop=int(args.synth_crop),
            temporal=temporal, seed=int(args.synth_seed),
            sensors=sensors_tag, gains=gains,
        )
        pairs.extend(s_pairs)
        evals.extend(s_evals)
        meta.setdefault("pair_gains", []).extend(s_gains)
        meta["synth"] = {
            "manifest": str(args.synth_manifest),
            "n": len(s_pairs),
            "crop": int(args.synth_crop),
            "sensors": list(sensors_tag),
        }
        print(f"Synth pairs: {len(s_pairs)}  (sensors={sensors_tag}, "
              f"crop={args.synth_crop})", flush=True)

    if not pairs:
        print("No training pairs — check bursts/ or pass --synth-manifest",
              file=sys.stderr)
        return 1
    print(f"Total train pairs: {len(pairs)}  in_ch={in_ch}", flush=True)

    model = RawDenoiser(
        base_channels=args.channels, block_depth=args.depth,
        in_ch=in_ch, out_ch=out_ch)
    start_step = 0
    best_psnr = -1.0
    if args.resume is not None:
        state, ckpt_step, ckpt_score = _load_resume(args.resume)
        model.load_state_dict(state)
        start_step = int(args.start_step) if args.start_step is not None else ckpt_step
        best_psnr = ckpt_score
        print(f"Resumed {args.resume}  step={start_step}  "
              f"score={best_psnr:.2f}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RawDenoiser {in_ch}→{out_ch}  {n_params:,} params  "
          f"({args.channels}ch × {args.depth}, T={temporal})", flush=True)

    loss_fn, loss_name = _make_loss()

    args.out.mkdir(parents=True, exist_ok=True)
    model = _train(
        model, pairs, args.steps, crop=args.crop, batch=args.batch, lr=args.lr,
        loss_fn=loss_fn, device=dev, panel_every=args.panel_every,
        panel_dir=args.panel_dir, evals=evals,
        best_path=args.out / "stream_to_gt_best.pt",
        start_step=start_step, best_psnr=best_psnr,
        last_path=args.out / "stream_to_gt_last.pt")

    rows = evaluate(model, evals, dev)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    print(f"Held-out mean PSNR {mean_in:.2f} → {mean_out:.2f} dB "
          f"(Δ{mean_out-mean_in:+.2f})", flush=True)
    print(f"Baseline (prior run): 12.98 → 25.29 dB", flush=True)

    best_meta = {}
    best_file = args.out / "stream_to_gt_best.pt"
    if best_file.exists():
        try:
            bb = torch.load(best_file, map_location="cpu", weights_only=False)
            best_meta = {"steps_best": bb.get("step"), "best_score": bb.get("score")}
        except Exception:
            pass
    ckpt = args.out / "stream_to_gt.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": {
            "family": "raw_denoiser_stream",
            "base_channels": args.channels,
            "block_depth": args.depth,
            "in_ch": in_ch, "out_ch": out_ch,
            "temporal": temporal,
        },
        "recipe": "stream_frame_to_burst_gt",
        "gt_frames": args.gt_frames,
        "gains": list(gains),
        "scenes": list(scenes),
        "stride": args.stride,
        "temporal": temporal,
        "loss": loss_name,
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "baseline_psnr_out": 25.29,
        "best_step": best_meta.get("steps_best"),
        "best_score": best_meta.get("best_score"),
        "eval": rows,
        "resumed_from": str(args.resume) if args.resume else None,
        "start_step": start_step,
        "steps": args.steps,
    }, ckpt)
    summary = {
        "psnr_in": mean_in, "psnr_out": mean_out,
        "baseline_psnr_out": 25.29,
        "delta_vs_baseline": mean_out - 25.29,
        "loss": loss_name, "params": n_params, "pairs": len(pairs),
        "channels": args.channels, "depth": args.depth,
        "steps": args.steps, "start_step": start_step,
        "stride": args.stride,
        "temporal": temporal, "in_ch": in_ch,
        "gains": list(gains), "eval": rows, **meta, **best_meta,
    }
    (args.out / "stream_to_gt_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)

    if evals:
        _save_eval_panel(model, evals[0], dev, args.panel_dir, args.steps)
        shutil.copy2(args.panel_dir / f"step_{args.steps:05d}.png",
                     args.out / "stream_to_gt_panel.png")

    if not args.no_onnx:
        onnx_path = args.out / "stream_to_gt.onnx"
        _export_onnx(model, in_ch, onnx_path)
        print(f"ONNX: {onnx_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
