#!/usr/bin/env python3
"""Train rectified-flow teacher: noisy frame → sharp packed GT | noisy stream.

This is the generative "Artist" model. It does NOT minimise pixel L1 to the
conditional mean (that is the blurry Old Way). It learns a velocity field
along the rectified-flow path from the live noisy frame (cond[:, :4]) to the
clean sample, conditioned on the full noisy temporal stack — so the ODE starts
at a good draft and needs only a few Euler steps.

Run on AI GPU (after / instead of regression)::

  .venv/bin/python -u train_cfm_teacher.py \\
      --gains 128,256,512 --steps 12000 --channels 128 --depth 8 \\
      --stride 2 --temporal 4 --sample-steps 20

Distill later with ``train_cfm_distill.py`` into a 1-step Pi student.
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
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.flow_matching import FlowVelocityNet, cfm_loss, euler_sample, grad_ratio
from nsa.inference import psnr, ssim, to_image, to_tensor
from nsa.raw_domain import packed_to_rgb
from train_stream_to_gt import (
    DEFAULT_GAINS,
    DEFAULT_SCENES,
    DISPLAY_GAIN,
    build_pairs,
)

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rgb(pk: np.ndarray) -> np.ndarray:
    return packed_to_rgb(pk, DISPLAY_GAIN)


def _rgb_t(pk: torch.Tensor) -> torch.Tensor:
    """Packed B×4×H×W → approximate RGB B×3×H×W for grad_ratio."""
    r = pk[:, 0:1]
    g = 0.5 * (pk[:, 1:2] + pk[:, 2:3])
    b = pk[:, 3:4]
    return torch.clamp(torch.cat([r, g, b], dim=1) * DISPLAY_GAIN, 0.0, 1.0)


def _sample_cond_clean(tensors, crop, batch, g, weights, pair_gains=None):
    """Minibatch of (cond, clean[, gain]) crops — clean is always 4ch GT.

    ``tensors`` entries are ``(cond, clean)``. ``pair_gains`` (parallel ints)
    yields a (B,) analogue-gain tensor when provided.
    """
    from nsa.inference import _augment_pair

    c = min([crop] + [min(t[0].shape[-2:]) for t in tensors])
    if weights is not None:
        idxs = torch.multinomial(weights, batch, replacement=True, generator=g)
    else:
        idxs = torch.randint(0, len(tensors), (batch,), generator=g)
    xs, ys, gs = [], [], []
    for k in range(batch):
        i = int(idxs[k])
        xi, yi = tensors[i]
        # yi may be 4ch; xi is 4T
        h, w = xi.shape[-2:]
        iy = int(torch.randint(0, h - c + 1, (1,), generator=g))
        ix = int(torch.randint(0, w - c + 1, (1,), generator=g))
        xc = xi[..., iy:iy + c, ix:ix + c]
        yc = yi[..., iy:iy + c, ix:ix + c]
        if yc.shape[1] > 4:
            yc = yc[:, :4]
        xc, yc = _augment_pair(xc, yc, g)
        xs.append(xc)
        ys.append(yc)
        if pair_gains is not None:
            gs.append(float(pair_gains[i]))
    cond = torch.cat(xs, 0)
    clean = torch.cat(ys, 0)
    if pair_gains is None:
        return cond, clean
    return cond, clean, torch.tensor(gs, dtype=torch.float32)


def measure_sigmas(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    x0_jitter: float = 0.02,
    max_pairs: int = 64,
) -> tuple[float, float]:
    """(sigma_data, sigma_flow) measured from the dataset.

    sigma_data = std of clean GT pixels (EDM signal scale; dark RAW ⇒ ≪ 0.5).
    sigma_flow = std of x₀ − x₁ = frame noise ⊕ x0_jitter (in quadrature) —
    the σ at flow time t=0.
    """
    step = max(1, len(pairs) // max_pairs)
    sig2_data, sig2_noise, n = 0.0, 0.0, 0
    for noisy, clean in pairs[::step]:
        c = clean[..., :4].astype(np.float64)
        d = noisy[..., :4].astype(np.float64) - c
        sig2_data += float(c.var())
        sig2_noise += float(d.var())
        n += 1
    sig2_data /= max(n, 1)
    sig2_noise /= max(n, 1)
    return math.sqrt(sig2_data), math.sqrt(sig2_noise + x0_jitter ** 2)


def _train(
    model: FlowVelocityNet,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    sample_steps: int,
    best_path: Path | None = None,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    ckpt_meta: dict | None = None,
    pair_gains: list[int] | None = None,
) -> FlowVelocityNet:
    tensors = [(to_tensor(n), to_tensor(c[..., :4] if c.shape[-1] > 4 else c))
               for n, c in pairs]
    wts = torch.tensor(
        [1.0 / max(float(n[..., :4].mean()), 1e-3) for n, _ in pairs],
        dtype=torch.float32)
    wts = (wts / wts.mean()).clamp(0.5, 4.0)
    use_gain = bool(getattr(model, "gain_film", False)) and pair_gains is not None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(662)
    model = model.to(device)
    model.train()
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_score = -1.0
    best_state: dict | None = None
    skipped = 0

    for i in range(steps):
        if use_gain:
            cond, clean, gain = _sample_cond_clean(
                tensors, crop, batch, g, wts, pair_gains)
            gain = gain.to(device)
        else:
            cond, clean = _sample_cond_clean(tensors, crop, batch, g, wts)
            gain = None
        cond, clean = cond.to(device), clean.to(device)
        opt.zero_grad(set_to_none=True)
        loss = cfm_loss(
            model, clean, cond, gain=gain, p_mean=p_mean, p_std=p_std)
        if not torch.isfinite(loss):
            skipped += 1
            if skipped <= 5 or skipped % 50 == 0:
                print(f"  skip non-finite loss at step {i+1} "
                      f"(total skipped {skipped})", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % 50 == 0 or i == steps - 1:
            print(f"  step {i+1}/{steps}  cfm={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/max(i,1):.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            panel_ev = next((e for e in evals if e.get("gain", 0) >= 256), evals[0])
            pout, gr = _save_eval_panel(
                model, panel_ev, device, panel_dir, i + 1, sample_steps)
            probe, probe_gr = _quick_probe(model, evals[:6], device, sample_steps)
            print(f"  probe mean PSNR={probe:.2f} dB  grad_ratio={probe_gr:.3f}",
                  flush=True)
            # Favour PSNR + texture energy matching GT (grad_ratio → 1.0);
            # both blur (ratio<1) and residual noise (ratio>1) are penalised.
            score = (0.4 * pout + 0.3 * probe
                     + 0.3 * (20.0 * max(0.0, 1.0 - abs(probe_gr - 1.0))))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                if best_path is not None:
                    torch.save({
                        "state_dict": best_state,
                        "model": ckpt_meta or {},
                        "step": i + 1,
                        "score": best_score,
                        "panel_psnr": pout,
                        "probe_psnr": probe,
                        "probe_grad_ratio": probe_gr,
                    }, best_path)
                print(f"  ★ best@{i+1}: panel={pout:.2f} probe={probe:.2f} "
                      f"grad_r={probe_gr:.3f} score={best_score:.2f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (score={best_score:.2f})", flush=True)
    if skipped:
        print(f"Skipped {skipped} non-finite steps", flush=True)
    model.eval()
    return model


def _eval_gain(model, ev, device) -> torch.Tensor | None:
    if not getattr(model, "gain_film", False):
        return None
    return torch.tensor([float(ev["gain"])], device=device, dtype=torch.float32)


@torch.no_grad()
def _quick_probe(model, evals, device, sample_steps) -> tuple[float, float]:
    model.eval()
    vals, grs = [], []
    for ev in evals:
        _, noisy = ev["noisy"][0]
        gt = ev["gt"]
        cond = to_tensor(noisy).to(device)
        gain = _eval_gain(model, ev, device)
        out = to_image(euler_sample(
            model, cond, gain=gain, steps=sample_steps).cpu())
        vals.append(psnr(_rgb(out), _rgb(gt)))
        grs.append(grad_ratio(
            _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0)),
            _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)),
        ))
    model.train()
    return (float(np.mean(vals)) if vals else 0.0,
            float(np.mean(grs)) if grs else 0.0)


@torch.no_grad()
def _save_eval_panel(model, ev, device, panel_dir, step, sample_steps) -> tuple[float, float]:
    was = model.training
    model.eval()
    idx, noisy = ev["noisy"][0]
    gt = ev["gt"]
    cond = to_tensor(noisy).to(device)
    gain = _eval_gain(model, ev, device)
    out = to_image(euler_sample(
        model, cond, gain=gain, steps=sample_steps).cpu())
    nr, gr, or_ = _rgb(noisy[..., :4]), _rgb(gt), _rgb(out)
    pin, pout = psnr(nr, gr), psnr(or_, gr)
    g_ratio = grad_ratio(
        _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0)),
        _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)),
    )
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
    sampler = ("Heun-ρ7" if getattr(model, "edm_precond", False) else "Euler")
    print(f"  panel step {step}: {ev['scene']}/ag{ev['gain']} "
          f"frame {idx}  {pin:.1f}→{pout:.1f} dB  grad_r={g_ratio:.3f}  "
          f"({sampler}×{sample_steps})", flush=True)
    if was:
        model.train()
    return float(pout), float(g_ratio)


@torch.no_grad()
def evaluate(model, evals, device, sample_steps) -> list[dict]:
    rows = []
    model.eval()
    for ev in evals:
        gt = ev["gt"]
        gr = _rgb(gt)
        gt_t = _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0))
        gain = _eval_gain(model, ev, device)
        for idx, noisy in ev["noisy"]:
            g_b = gain
            if gain is not None:
                g_b = gain  # (1,) — euler_sample batches size 1
            out = to_image(euler_sample(
                model, to_tensor(noisy).to(device), gain=g_b,
                steps=sample_steps).cpu())
            nr, or_ = _rgb(noisy[..., :4]), _rgb(out)
            out_t = _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0))
            rows.append({
                "scene": ev["scene"], "gain": ev["gain"], "frame": idx,
                "psnr_in": round(psnr(nr, gr), 2),
                "psnr_out": round(psnr(or_, gr), 2),
                "ssim_in": round(ssim(nr, gr), 4),
                "ssim_out": round(ssim(or_, gr), 4),
                "grad_ratio": round(grad_ratio(out_t, gt_t), 4),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument(
        "--gt-mode", choices=("mean", "alpha_trim"), default="mean",
        help="GT target: burst mean (default) or alpha-trim cache",
    )
    ap.add_argument(
        "--gt-cache", type=Path, default=ROOT / "datasets/imx662_project/gt_alpha16",
        help="Alpha-trim GT cache root (gt_mode=alpha_trim)",
    )
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--temporal", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--sample-steps", type=int, default=20,
                    help="ODE steps for panels / eval (EDM mode: Heun + ρ=7)")
    ap.add_argument("--edm", action="store_true",
                    help="EDM preconditioning (c_in/c_skip/c_out/c_noise), "
                         "logit-normal t + loss weighting, Heun ρ=7 sampler")
    ap.add_argument("--p-mean", type=float, default=0.0,
                    help="logit-normal t location (EDM P_mean analog)")
    ap.add_argument("--p-std", type=float, default=1.0,
                    help="logit-normal t scale (EDM P_std analog)")
    ap.add_argument("--panel-every", type=int, default=400)
    ap.add_argument("--panel-dir", type=Path, default=ROOT / "outputs/cfm_teacher_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--gain-film", action="store_true",
                    help="FiLM-modulate features by log2(gain/128) analogue gain")
    ap.add_argument("--init", type=Path, default=None,
                    help="Warm-start from an existing teacher checkpoint "
                         "(strict=False so new gain_mlp layers init to identity)")
    args = ap.parse_args()

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    gt_frames = int(args.gt_frames)
    if args.gt_mode == "alpha_trim" and gt_frames == 512:
        gt_frames = 16
    cond_ch = 4 * temporal
    out_ch = 4
    dev = _device()
    print(f"Device {dev}  recipe: CFM Teacher  noisy→GT | STREAM×{temporal}"
          + ("  [gain FiLM]" if args.gain_film else ""),
          flush=True)
    print(f"Scenes {scenes}  gains {gains}  Euler steps={args.sample_steps}",
          flush=True)

    pairs, evals, meta = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=gt_frames, stride=args.stride,
        holdout_start=args.holdout_start, temporal=temporal,
        gt_mode=args.gt_mode, gt_cache_root=args.gt_cache)
    if not pairs:
        print("No training pairs — check bursts/", file=sys.stderr)
        return 1
    pair_gains = meta.get("pair_gains")
    print(f"Total train pairs: {len(pairs)}  cond_ch={cond_ch}", flush=True)

    sigma_data = sigma_flow = None
    if args.edm:
        sigma_data, sigma_flow = measure_sigmas(pairs)
        print(f"EDM precond: measured sigma_data={sigma_data:.5f}  "
              f"sigma_flow={sigma_flow:.5f}  "
              f"t~logitN({args.p_mean},{args.p_std})", flush=True)
    model = FlowVelocityNet(
        cond_ch=cond_ch, out_ch=out_ch,
        base_channels=args.channels, block_depth=args.depth,
        edm_precond=args.edm,
        sigma_data=sigma_data or 0.05, sigma_flow=sigma_flow or 0.06,
        gain_film=args.gain_film)
    if args.init is not None:
        if not args.init.is_file():
            print(f"Init checkpoint missing: {args.init}", file=sys.stderr)
            return 1
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        src = blob["state_dict"]
        dst = model.state_dict()
        compatible = {
            k: v for k, v in src.items()
            if k in dst and v.shape == dst[k].shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        skipped = len(src) - len(compatible)
        print(f"Warm-start {args.init}: loaded {len(compatible)} tensors  "
              f"skipped_shape={skipped}  missing={len(missing)}  "
              f"unexpected={len(unexpected)}", flush=True)
        # Re-assert identity FiLM on newly added gain layers after partial load.
        if model.gain_mlp is not None and any("gain_mlp" in m for m in missing):
            torch.nn.init.zeros_(model.gain_mlp[-1].weight)
            torch.nn.init.zeros_(model.gain_mlp[-1].bias)
            print("gain_mlp last Linear zero-init (identity FiLM)", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowVelocityNet cond={cond_ch} → vel={out_ch}  {n_params:,} params  "
          f"({args.channels}ch × {args.depth})"
          + ("  [EDM precond]" if args.edm else "")
          + ("  [gain FiLM]" if args.gain_film else ""), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)  # best_path saves mid-training
    ckpt_meta = {
        "family": "cfm_teacher",
        "base_channels": args.channels,
        "block_depth": args.depth,
        "cond_ch": cond_ch,
        "out_ch": out_ch,
        "temporal": temporal,
        "edm_precond": bool(args.edm),
        "sigma_data": sigma_data,
        "sigma_flow": sigma_flow,
        "gain_film": bool(args.gain_film),
    }
    model = _train(
        model, pairs, args.steps, crop=args.crop, batch=args.batch, lr=args.lr,
        device=dev, panel_every=args.panel_every, panel_dir=args.panel_dir,
        evals=evals, sample_steps=args.sample_steps,
        best_path=args.out / "cfm_teacher_best.pt",
        p_mean=args.p_mean, p_std=args.p_std, ckpt_meta=ckpt_meta,
        pair_gains=pair_gains if args.gain_film else None)

    rows = evaluate(model, evals, dev, args.sample_steps)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    mean_gr = float(np.mean([r["grad_ratio"] for r in rows]))
    print(f"Held-out mean PSNR {mean_in:.2f} → {mean_out:.2f} dB  "
          f"grad_ratio={mean_gr:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "cfm_teacher.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model": ckpt_meta,
        "recipe": ("conditional_flow_matching_edm" if args.edm
                   else "conditional_flow_matching"),
        "sample_steps": args.sample_steps,
        "gt_frames": gt_frames,
        "gt_mode": args.gt_mode,
        "gains": list(gains),
        "scenes": list(scenes),
        "stride": args.stride,
        "temporal": temporal,
        "gain_film": bool(args.gain_film),
        "init": str(args.init) if args.init else None,
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "grad_ratio": mean_gr,
        "eval": rows,
    }, ckpt)
    # Drop pair_gains from JSON summary (large, redundant with scenes).
    meta_json = {k: v for k, v in meta.items() if k != "pair_gains"}
    summary = {
        "recipe": ("conditional_flow_matching_edm" if args.edm
                   else "conditional_flow_matching"),
        "edm_precond": bool(args.edm),
        "gain_film": bool(args.gain_film),
        "sigma_data": sigma_data, "sigma_flow": sigma_flow,
        "p_mean": args.p_mean, "p_std": args.p_std,
        "psnr_in": mean_in, "psnr_out": mean_out,
        "grad_ratio": mean_gr,
        "params": n_params, "pairs": len(pairs),
        "channels": args.channels, "depth": args.depth,
        "steps": args.steps, "sample_steps": args.sample_steps,
        "temporal": temporal, "cond_ch": cond_ch,
        "gains": list(gains), "eval": rows, **meta_json,
    }
    (args.out / "cfm_teacher_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)

    if evals:
        _save_eval_panel(model, evals[0], dev, args.panel_dir, args.steps,
                         args.sample_steps)
        shutil.copy2(args.panel_dir / f"step_{args.steps:05d}.png",
                     args.out / "cfm_teacher_panel.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
