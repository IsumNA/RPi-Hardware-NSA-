#!/usr/bin/env python3
"""Train Stage B detail head on frozen Stage A (Charbonnier + LPIPS + FFL).

Requires a trained Stage A boundary student (--stage-a). Optional QAT fine-tune
on the detail head only (``enable_qat`` / ``fake_quantize_int8``).

Example::

  .venv/bin/python -u train_cfm_stage_b.py \\
      --stage-a outputs/cfm_l1/cfm_student.pt \\
      --steps 4000 --out outputs/cfm_stage_b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.detail_head import (
    DetailHead,
    StageBDeployWrapper,
    StageBRefiner,
    load_stage_a,
)
from nsa.flow_matching import grad_ratio
from nsa.inference import (
    build_loss,
    disable_qat,
    enable_qat,
    fake_quantize_int8,
    psnr,
    to_image,
    to_tensor,
)
from train_cfm_distill import _cond_for_eval, evaluate
from train_cfm_teacher import _rgb, _rgb_t, _sample_cond_clean
from train_stream_to_gt import (
    DEFAULT_GAINS,
    DEFAULT_SCENES,
    _export_onnx,
    build_pairs,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pairs_to_tensors(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    temporal: int,
    gain_channel: bool,
    pair_gains: list[int] | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from nsa.flow_matching import append_gain_channel

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (noisy, clean) in enumerate(pairs):
        cond = to_tensor(noisy)
        gt = to_tensor(clean[..., :4])
        if gain_channel and pair_gains is not None:
            g = torch.tensor([float(pair_gains[i])], dtype=torch.float32)
            cond = append_gain_channel(cond, g)
        out.append((cond, gt))
    return out


def _train_detail(
    refiner: StageBRefiner,
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    device: torch.device,
    loss_fn,
    pair_gains: list[int] | None,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    gain_channel: bool,
    best_path: Path | None,
    qat: bool = False,
) -> StageBRefiner:
    refiner.stage_a.eval()
    for p in refiner.stage_a.parameters():
        p.requires_grad_(False)
    detail = refiner.detail
    detail.train()
    if qat:
        enable_qat(detail)

    weights = None
    if pair_gains is not None and len(pair_gains) == len(tensors):
        weights = torch.ones(len(tensors), dtype=torch.float32)

    torch.manual_seed(42)
    g = torch.Generator().manual_seed(42)
    steps = max(1, steps)
    warmup = max(1, steps // 10)
    opt = torch.optim.AdamW(detail.parameters(), lr=lr, weight_decay=1e-4)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * (1.0 - 0.02) + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    deploy = StageBDeployWrapper(refiner)
    best_score = -1e9
    panel_dir.mkdir(parents=True, exist_ok=True)

    for i in range(steps):
        batch_out = _sample_cond_clean(
            tensors, crop, batch, g, weights, pair_gains=None,
        )
        cond, clean = batch_out[0], batch_out[1]
        cond = cond.to(device)
        clean = clean.to(device)
        opt.zero_grad()
        pred = refiner(cond)
        loss = loss_fn(pred, clean)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(detail.parameters(), 1.0)
        opt.step()
        sched.step()

        if panel_every > 0 and (i % panel_every == 0 or i == steps - 1):
            with torch.no_grad():
                ev = evals[0] if evals else None
                if ev is not None:
                    idx, noisy = ev["noisy"][0]
                    gt = ev["gt"]
                    c = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
                    out = to_image(deploy(c).cpu())
                    nr, gr, or_ = _rgb(noisy[..., :4]), _rgb(gt), _rgb(out)
                    pout = psnr(or_, gr)
                    g_ratio = grad_ratio(
                        _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0)),
                        _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)),
                    )
                    strip = np.concatenate([nr, or_, gr], axis=1)
                    img = (np.clip(strip, 0, 1) * 255 + 0.5).astype(np.uint8)
                    path = panel_dir / f"step_{i:05d}.png"
                    Image.fromarray(img).save(path)
                    print(
                        f"  step {i}/{steps} loss={float(loss.item()):.5f}  "
                        f"panel PSNR={pout:.2f} dB grad_r={g_ratio:.3f}",
                        flush=True,
                    )
                    score = pout - 2.0 * abs(g_ratio - 1.0)
                    if best_path and score > best_score:
                        best_score = score
                        torch.save({
                            "state_dict": detail.state_dict(),
                            "model": {
                                "family": "cfm_detail_head",
                                "out_ch": detail.out_ch,
                                "base_channels": detail.head.out_channels
                                if hasattr(detail.head, "out_channels")
                                else 32,
                                "block_depth": len(detail.body),
                                "use_noisy_hf": detail.use_noisy_hf,
                                "residual_scale": detail.residual_scale,
                                "hf_sigma": detail.hf_sigma,
                            },
                        }, best_path)
        elif i % max(1, steps // 20) == 0:
            print(f"  step {i}/{steps} loss={float(loss.item()):.5f}", flush=True)

    if qat:
        disable_qat(detail)
    detail.eval()
    return refiner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", type=Path, required=True,
                    help="Frozen Stage A cfm_student.pt (required)")
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--temporal", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--detail-channels", type=int, default=32)
    ap.add_argument("--detail-depth", type=int, default=3)
    ap.add_argument("--residual-scale", type=float, default=0.12)
    ap.add_argument("--no-noisy-hf", action="store_true")
    ap.add_argument(
        "--loss", default="charbonnier+perceptual+ffl",
        help="Composite loss (see nsa.inference.build_loss)",
    )
    ap.add_argument("--panel-every", type=int, default=400)
    ap.add_argument("--panel-dir", type=Path,
                    default=ROOT / "outputs/cfm_stage_b_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cfm_stage_b")
    ap.add_argument("--qat", action="store_true",
                    help="Fake-quant train on DetailHead only")
    ap.add_argument("--qat-steps", type=int, default=800,
                    help="Extra steps when --qat (0 = use --steps only)")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    if not args.stage_a.is_file():
        print(f"Missing --stage-a: {args.stage_a}", file=sys.stderr)
        return 1

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    dev = _device()
    stage_a, ameta = load_stage_a(args.stage_a, dev)
    gain_channel = bool(ameta.get("gain_channel", False))
    in_ch = int(ameta.get("cond_ch", ameta.get("in_ch", 4 * temporal)))

    print(f"Device {dev}  frozen Stage A: {args.stage_a}", flush=True)

    pairs, evals, meta = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=args.gt_frames, stride=args.stride,
        holdout_start=args.holdout_start, temporal=temporal,
    )
    if not pairs:
        print("No training pairs", file=sys.stderr)
        return 1
    pair_gains = meta.get("pair_gains")
    tensors = _pairs_to_tensors(pairs, temporal, gain_channel, pair_gains)
    print(f"Train pairs: {len(tensors)}  in_ch={in_ch}", flush=True)

    detail = DetailHead(
        out_ch=4,
        base_channels=args.detail_channels,
        block_depth=args.detail_depth,
        use_noisy_hf=not args.no_noisy_hf,
        residual_scale=args.residual_scale,
    ).to(dev)
    refiner = StageBRefiner(stage_a, detail)
    n_params = sum(p.numel() for p in detail.parameters())
    print(f"DetailHead {n_params:,} params  loss={args.loss}", flush=True)

    loss_fn = build_loss(
        args.loss,
        weights={"charbonnier": 1.0, "perceptual": 0.08, "ffl": 0.12},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / "cfm_detail_best.pt"

    train_steps = args.steps
    if args.qat and args.qat_steps > 0:
        train_steps = max(1, args.steps - args.qat_steps)

    refiner = _train_detail(
        refiner, tensors, train_steps,
        crop=args.crop, batch=args.batch, lr=args.lr, device=dev,
        loss_fn=loss_fn, pair_gains=pair_gains,
        panel_every=args.panel_every, panel_dir=args.panel_dir, evals=evals,
        gain_channel=gain_channel, best_path=best_path, qat=False,
    )

    if args.qat:
        qsteps = args.qat_steps if args.qat_steps > 0 else args.steps
        print(f"QAT fine-tune on DetailHead ({qsteps} steps)", flush=True)
        refiner = _train_detail(
            refiner, tensors, qsteps,
            crop=args.crop, batch=args.batch, lr=args.lr * 0.5, device=dev,
            loss_fn=loss_fn, pair_gains=pair_gains,
            panel_every=0, panel_dir=args.panel_dir, evals=evals,
            gain_channel=gain_channel, best_path=None, qat=True,
        )

    deploy = StageBDeployWrapper(refiner).to(dev).eval()
    rows = evaluate(deploy, evals, dev, gain_channel=gain_channel)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    mean_gr = float(np.mean([r["grad_ratio"] for r in rows]))

    stage_a_only = stage_a
    rows_a = evaluate(stage_a_only, evals, dev, gain_channel=gain_channel)
    mean_out_a = float(np.mean([r["psnr_out"] for r in rows_a]))

    print(
        f"Held-out PSNR {mean_in:.2f} → {mean_out:.2f} dB "
        f"(Stage A alone {mean_out_a:.2f})  grad_ratio={mean_gr:.3f}",
        flush=True,
    )

    ckpt = args.out / "cfm_stage_b.pt"
    torch.save({
        "state_dict": {
            **{f"stage_a.{k}": v for k, v in stage_a.state_dict().items()},
            **{f"detail.{k}": v for k, v in detail.state_dict().items()},
        },
        "model": {
            "family": "cfm_stage_a_b",
            "stage_a": str(args.stage_a),
            "base_channels": args.detail_channels,
            "block_depth": args.detail_depth,
            "use_noisy_hf": not args.no_noisy_hf,
            "residual_scale": args.residual_scale,
            "cond_ch": in_ch,
            "in_ch": in_ch,
            "out_ch": 4,
            "temporal": int(ameta.get("temporal", temporal)),
            "gain_channel": gain_channel,
            "loss": args.loss,
            "qat": bool(args.qat),
        },
        "recipe": "cfm_stage_b_detail",
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "psnr_stage_a": mean_out_a,
        "grad_ratio": mean_gr,
        "eval": rows,
    }, ckpt)

    detail_ckpt = args.out / "cfm_detail_head.pt"
    torch.save({
        "state_dict": detail.state_dict(),
        "model": {
            "family": "cfm_detail_head",
            "out_ch": 4,
            "base_channels": args.detail_channels,
            "block_depth": args.detail_depth,
            "use_noisy_hf": not args.no_noisy_hf,
            "residual_scale": args.residual_scale,
        },
        "stage_a": str(args.stage_a),
        "psnr_out": mean_out,
        "grad_ratio": mean_gr,
    }, detail_ckpt)

    (args.out / "cfm_stage_b_summary.json").write_text(json.dumps({
        "recipe": "cfm_stage_b_detail",
        "stage_a": str(args.stage_a),
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "psnr_stage_a": mean_out_a,
        "grad_ratio": mean_gr,
        "loss": args.loss,
        "eval": rows,
    }, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)

    if not args.no_onnx:
        onnx_path = args.out / "cfm_stage_b.onnx"
        _export_onnx(deploy.cpu(), in_ch, onnx_path)
        print(f"ONNX: {onnx_path}", flush=True)

        q_detail = fake_quantize_int8(detail.cpu(), quant_activations=True)
        q_refiner = StageBRefiner(stage_a.cpu(), q_detail)
        q_deploy = StageBDeployWrapper(q_refiner).cpu().eval()
        int8_path = args.out / "cfm_stage_b_int8.onnx"
        _export_onnx(q_deploy, in_ch, int8_path)
        print(f"INT8-emulated ONNX: {int8_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
