#!/usr/bin/env python3
"""Holdout eval for CFM Stage A / A+B: PSNR, LPIPS, SSIM; grad_ratio reported only.

Optional temporal flicker strip over consecutive burst frames (t..t+7).

Example::

  .venv/bin/python -u eval_cfm_pipeline.py \\
      --stage-a outputs/cfm_l1/cfm_student.pt \\
      --stage-b outputs/cfm_stage_b/cfm_detail_head.pt \\
      --flicker --flicker-scene office --flicker-gain 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.detail_head import build_deploy_from_checkpoints, load_stage_a
from nsa.flow_matching import grad_ratio
from nsa.inference import lpips, psnr, quality_metrics, ssim, to_image, to_tensor
from train_cfm_distill import _cond_for_eval
from train_cfm_teacher import _rgb, _rgb_t
from train_stream_to_gt import DEFAULT_GAINS, DEFAULT_SCENES, build_pairs


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_rows(
    model,
    evals: list[dict],
    device: torch.device,
    *,
    gain_channel: bool,
    label: str,
) -> list[dict]:
    rows = []
    model.eval()
    for ev in evals:
        gt = ev["gt"]
        gr = _rgb(gt)
        gt_t = _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0))
        for idx, noisy in ev["noisy"]:
            cond = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
            out = to_image(model(cond).cpu())
            nr, or_ = _rgb(noisy[..., :4]), _rgb(out)
            qm = quality_metrics(or_, gr)
            out_t = _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0))
            rows.append({
                "model": label,
                "scene": ev["scene"],
                "gain": ev["gain"],
                "frame": idx,
                "psnr_in": round(psnr(nr, gr), 2),
                "psnr_out": round(qm["psnr"], 2),
                "ssim_out": round(qm["ssim"], 4),
                "lpips_out": round(qm["lpips"], 4),
                "grad_ratio": round(grad_ratio(out_t, gt_t), 4),
            })
    return rows


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "psnr_in_mean": float(np.mean([r["psnr_in"] for r in rows])),
        "psnr_out_mean": float(np.mean([r["psnr_out"] for r in rows])),
        "ssim_out_mean": float(np.mean([r["ssim_out"] for r in rows])),
        "lpips_out_mean": float(np.mean([r["lpips_out"] for r in rows])),
        "grad_ratio_mean": float(np.mean([r["grad_ratio"] for r in rows])),
    }


def _flicker_strip(
    model,
    evals: list[dict],
    scene: str,
    gain: int,
    *,
    device: torch.device,
    gain_channel: bool,
    start_frame: int,
    n_frames: int,
    out_path: Path,
) -> dict | None:
    """Vertical strip: for each frame, row = [noisy | denoised | GT]."""
    ev = next(
        (e for e in evals if e["scene"] == scene and int(e["gain"]) == int(gain)),
        None,
    )
    if ev is None:
        print(f"flicker: no holdout eval for {scene}/ag{gain}", flush=True)
        return None

    gt4 = ev["gt"]
    gr = _rgb(gt4[..., :4] if gt4.shape[-1] > 4 else gt4)
    noisy_list = sorted(ev["noisy"], key=lambda x: x[0])
    rows_img = []
    metrics_before, metrics_after = [], []

    for idx, noisy in noisy_list:
        if idx < start_frame:
            continue
        if len(rows_img) >= n_frames:
            break
        cond = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
        with torch.no_grad():
            out = to_image(model(cond).cpu())
        nr, or_ = _rgb(noisy[..., :4]), _rgb(out)
        metrics_before.append(lpips(nr, gr))
        metrics_after.append(lpips(or_, gr))
        row = np.concatenate([nr, or_, gr], axis=1)
        rows_img.append(row)

    if not rows_img:
        return None
    strip = np.concatenate(rows_img, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(strip, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        out_path,
    )
    return {
        "scene": scene,
        "gain": gain,
        "start_frame": start_frame,
        "n_frames": len(rows_img),
        "lpips_noisy_mean": float(np.mean(metrics_before)),
        "lpips_denoised_mean": float(np.mean(metrics_after)),
        "artifact": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--stage-b", type=Path, default=None,
                    help="Optional detail head checkpoint")
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--temporal", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cfm_eval")
    ap.add_argument("--flicker", action="store_true")
    ap.add_argument("--flicker-scene", default="office")
    ap.add_argument("--flicker-gain", type=int, default=512)
    ap.add_argument("--flicker-start", type=int, default=400)
    ap.add_argument("--flicker-frames", type=int, default=8)
    args = ap.parse_args()

    if not args.stage_a.is_file():
        print(f"Missing --stage-a: {args.stage_a}", file=sys.stderr)
        return 1

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    dev = _device()

    _, evals, _ = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=args.gt_frames, stride=args.stride,
        holdout_start=args.holdout_start, temporal=temporal,
    )
    if not evals:
        print("No holdout eval pairs", file=sys.stderr)
        return 1

    stage_a, ameta = load_stage_a(args.stage_a, dev)
    gain_channel = bool(ameta.get("gain_channel", False))

    deploy, _, _ = build_deploy_from_checkpoints(
        args.stage_a, args.stage_b, dev,
    )
    rows_a = _eval_rows(stage_a, evals, dev, gain_channel=gain_channel, label="stage_a")
    rows_ab = _eval_rows(deploy, evals, dev, gain_channel=gain_channel, label="stage_a_b")

    sum_a = _summarize(rows_a)
    sum_ab = _summarize(rows_ab)
    print(
        f"Stage A:  PSNR {sum_a['psnr_out_mean']:.2f} dB  "
        f"LPIPS {sum_a['lpips_out_mean']:.4f}  "
        f"grad_ratio {sum_a['grad_ratio_mean']:.3f} (report only)",
        flush=True,
    )
    if args.stage_b and args.stage_b.is_file():
        print(
            f"Stage A+B: PSNR {sum_ab['psnr_out_mean']:.2f} dB  "
            f"LPIPS {sum_ab['lpips_out_mean']:.4f}  "
            f"grad_ratio {sum_ab['grad_ratio_mean']:.3f} (report only)",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "stage_a": str(args.stage_a),
        "stage_b": str(args.stage_b) if args.stage_b else None,
        "summary_stage_a": sum_a,
        "summary_stage_a_b": sum_ab if args.stage_b else None,
        "rows_stage_a": rows_a,
        "rows_stage_a_b": rows_ab if args.stage_b else None,
        "note": "grad_ratio is reported only, not optimized",
    }

    if args.flicker:
        flick = _flicker_strip(
            deploy, evals, args.flicker_scene, args.flicker_gain,
            device=dev, gain_channel=gain_channel,
            start_frame=args.flicker_start, n_frames=args.flicker_frames,
            out_path=args.out / f"flicker_{args.flicker_scene}_ag{args.flicker_gain}.png",
        )
        report["flicker"] = flick
        if flick:
            print(
                f"Flicker strip: LPIPS noisy {flick['lpips_noisy_mean']:.4f} → "
                f"denoised {flick['lpips_denoised_mean']:.4f}  "
                f"{flick['artifact']}",
                flush=True,
            )

    out_json = args.out / "eval_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Report: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
