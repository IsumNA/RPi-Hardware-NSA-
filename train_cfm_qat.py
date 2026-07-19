#!/usr/bin/env python3
"""QAT fine-tune a CFM 1-step student for INT8 deploy (Pi / Hailo / DeepX).

Loads an existing student checkpoint (``cfm_student.pt``), inserts fake-quant
nodes (STE), and continues a short consistency distill so weights survive INT8.

Run on AI GPU::

  .venv/bin/python -u train_cfm_qat.py \\
      --student outputs/cfm_prod/cfm_student.pt \\
      --teacher outputs/cfm_teacher.pt \\
      --steps 1500 --out outputs/cfm_prod_int8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.inference import enable_qat, disable_qat, fake_quantize_int8
from nsa.flow_matching import ConsistencyStudent, BoundaryConsistencyWrapper
from train_cfm_distill import (
    _device,
    _load_teacher,
    _make_sample_loss,
    _train_consistency,
    evaluate,
)
from train_stream_to_gt import DEFAULT_GAINS, DEFAULT_SCENES, build_pairs, _export_onnx


def _load_student(path: Path, device: torch.device) -> tuple[ConsistencyStudent, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    state = blob["state_dict"]
    if any(k.startswith("student.") for k in state):
        state = {k[len("student."):]: v for k, v in state.items()
                 if k.startswith("student.")}
    gain_channel = bool(meta.get("gain_channel", False))
    temporal = int(meta.get("temporal", 4))
    cond_ch = int(meta.get("cond_ch", meta.get("in_ch", 4 * temporal + int(gain_channel))))
    student = ConsistencyStudent(
        cond_ch=cond_ch,
        out_ch=int(meta.get("out_ch", 4)),
        base_channels=int(meta.get("base_channels", 64)),
        block_depth=int(meta.get("block_depth", 6)),
        gain_channel=gain_channel,
    )
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        print(f"WARN student missing keys: {missing[:8]}…", flush=True)
    if unexpected:
        print(f"WARN student unexpected keys: {unexpected[:8]}…", flush=True)
    student.to(device)
    return student, blob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, default=ROOT / "outputs/cfm_teacher.pt")
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--integrate-steps", type=int, default=6)
    ap.add_argument("--cd-weight", type=float, default=0.25)
    ap.add_argument("--gt-weight", type=float, default=0.0)
    ap.add_argument("--sample-loss", default="l1_hf")
    ap.add_argument("--panel-every", type=int, default=500)
    ap.add_argument("--panel-dir", type=Path,
                    default=ROOT / "outputs/cfm_qat_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cfm_prod_int8")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    if not args.student.is_file():
        print(f"Missing student: {args.student}", file=sys.stderr)
        return 1
    if not args.teacher.is_file():
        print(f"Missing teacher: {args.teacher}", file=sys.stderr)
        return 1

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    dev = _device()
    print(f"Device {dev}  QAT fine-tune from {args.student}", flush=True)

    student, sblob = _load_student(args.student, dev)
    teacher, _ = _load_teacher(args.teacher, dev)
    meta = sblob.get("model", {})
    temporal = int(meta.get("temporal", 4))
    gain_channel = bool(getattr(student, "gain_channel", False))
    in_ch = int(getattr(student, "cond_ch", 4 * temporal + int(gain_channel)))

    pairs, evals, pair_meta = build_pairs(
        args.bursts, scenes, gains,
        gt_frames=args.gt_frames, stride=args.stride,
        holdout_start=args.holdout_start, temporal=temporal)
    if not pairs:
        print("No pairs", file=sys.stderr)
        return 1
    pair_gains = pair_meta.get("pair_gains")

    enable_qat(student)
    print("QAT fake-quant enabled on all Conv2d layers", flush=True)
    sample_loss, sample_loss_name = _make_sample_loss(name=args.sample_loss)

    args.out.mkdir(parents=True, exist_ok=True)
    student = _train_consistency(
        student, teacher, pairs, args.steps,
        crop=args.crop, batch=args.batch, lr=args.lr, device=dev,
        panel_every=args.panel_every, panel_dir=args.panel_dir, evals=evals,
        integrate_steps=args.integrate_steps, cd_weight=args.cd_weight,
        cd_intervals=8, ema_decay=0.999, gt_weight=args.gt_weight,
        heun=True, sample_loss=sample_loss, sample_loss_name=sample_loss_name,
        best_path=args.out / "cfm_student_qat_best.pt",
        pair_gains=pair_gains,
    )

    disable_qat(student)
    q_student = fake_quantize_int8(student)
    deploy = BoundaryConsistencyWrapper(q_student).to(dev).eval()
    rows = evaluate(deploy, evals, dev, gain_channel=gain_channel)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    mean_gr = float(np.mean([r["grad_ratio"] for r in rows]))
    print(f"INT8-eval held-out PSNR {mean_in:.2f} → {mean_out:.2f} dB  "
          f"grad_ratio={mean_gr:.3f}", flush=True)

    ckpt = args.out / "cfm_student_int8.pt"
    torch.save({
        "state_dict": deploy.state_dict(),
        "model": {
            "family": "cfm_consistency_1step_int8",
            "base_channels": int(meta.get("base_channels", 64)),
            "block_depth": int(meta.get("block_depth", 6)),
            "cond_ch": in_ch,
            "in_ch": in_ch,
            "out_ch": 4,
            "temporal": temporal,
            "gain_channel": gain_channel,
            "qat": True,
            "precision": "int8_fakequant",
        },
        "recipe": "cfm_qat_int8",
        "teacher": str(args.teacher),
        "student_init": str(args.student),
        "sample_loss": sample_loss_name,
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "grad_ratio": mean_gr,
        "eval": rows,
    }, ckpt)
    (args.out / "cfm_student_int8_summary.json").write_text(
        json.dumps({
            "recipe": "cfm_qat_int8",
            "psnr_in": mean_in, "psnr_out": mean_out,
            "grad_ratio": mean_gr,
            "sample_loss": sample_loss_name,
            "student_init": str(args.student),
            "eval": rows,
        }, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)

    # FP32 graph with QAT-trained weights (vendor SDKs / ORT quantize further).
    fp_deploy = BoundaryConsistencyWrapper(student).cpu().eval()
    fp_ckpt = args.out / "cfm_student_qat_fp32.pt"
    torch.save({
        "state_dict": fp_deploy.state_dict(),
        "model": {
            "family": "cfm_consistency_1step",
            "base_channels": int(meta.get("base_channels", 64)),
            "block_depth": int(meta.get("block_depth", 6)),
            "cond_ch": in_ch, "in_ch": in_ch, "out_ch": 4,
            "temporal": temporal, "gain_channel": gain_channel,
            "qat_trained": True,
        },
        "recipe": "cfm_qat_fp32_weights",
        "psnr_out": mean_out, "grad_ratio": mean_gr,
    }, fp_ckpt)

    if not args.no_onnx:
        onnx_path = args.out / "cfm_student_int8.onnx"
        _export_onnx(fp_deploy, in_ch, onnx_path)
        print(f"ONNX: {onnx_path}", flush=True)
        shutil.copy2(onnx_path, args.out / "stream_to_gt_cfm_int8.onnx")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
