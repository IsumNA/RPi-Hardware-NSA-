#!/usr/bin/env python3
"""Export Stage A+B deploy graph to ONNX (+ optional INT8 detail head).

Follows ``export_cfm_targets.py`` patterns: dynamic CPU ONNX, static tiles for
vendor targets, fake-quant on DetailHead when ``--int8``.

Usage::

  .venv/bin/python -u export_cfm_stage_b.py \\
      --stage-a outputs/cfm_l1/cfm_student.pt \\
      --stage-b outputs/cfm_stage_b/cfm_detail_head.pt \\
      --out outputs/cfm_stage_b_deploy --int8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.detail_head import (
    build_deploy_from_checkpoints,
    fake_quantize_detail_only,
)
from export_cfm_targets import _export_static_onnx
from train_stream_to_gt import _export_onnx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--stage-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cfm_stage_b_deploy")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--targets", default="cpu,hailo10h,deepx")
    ap.add_argument("--int8", action="store_true",
                    help="Fake-quant DetailHead for Hailo/DeepX blobs")
    args = ap.parse_args()

    if not args.stage_a.is_file():
        print(f"Missing --stage-a: {args.stage_a}", file=sys.stderr)
        return 1
    if not args.stage_b.is_file():
        print(f"Missing --stage-b: {args.stage_b}", file=sys.stderr)
        return 1

    model, meta, in_ch = build_deploy_from_checkpoints(
        args.stage_a, args.stage_b, torch.device("cpu"),
    )
    if not isinstance(model, nn.Module):
        print("Expected Stage A+B deploy module", file=sys.stderr)
        return 1

    pack_model: nn.Module = model
    if args.int8 and hasattr(model, "detail"):
        pack_model = fake_quantize_detail_only(model)  # type: ignore[arg-type]

    n_params = sum(p.numel() for p in model.parameters())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "stage_a": str(args.stage_a),
        "stage_b": str(args.stage_b),
        "model": meta,
        "in_ch": in_ch,
        "params": n_params,
        "patch": args.patch,
        "int8_detail": bool(args.int8),
        "targets": {},
        "pipeline_snippet": (
            "Boundary(StageA) → bilinear align → StageA + tanh(StageB); "
            "export via export_cfm_stage_b.py --int8 for DetailHead fake-quant"
        ),
    }

    dyn = args.out / "cfm_stage_b_cpu.onnx"
    _export_onnx(pack_model, in_ch, dyn, patch=args.patch)
    shutil.copy2(dyn, args.out / "cfm_stage_b.onnx")
    manifest["targets"]["cpu"] = {
        "artifact": str(dyn),
        "runtime": "onnxruntime-cpu",
        "precision": "fp32" if not args.int8 else "int8_detail_fakequant",
    }
    print(f"cpu: {dyn}", flush=True)

    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    for tgt in targets:
        if tgt == "cpu":
            continue
        tdir = args.out / tgt
        tdir.mkdir(parents=True, exist_ok=True)
        static = tdir / "cfm_stage_b_static.onnx"
        _export_static_onnx(pack_model, in_ch, static, args.patch)
        manifest["targets"][tgt] = {
            "static_onnx": str(static),
            "precision": "int8_detail" if args.int8 else "fp32",
            "notes": (
                f"Run {tgt} SDK on static ONNX; calibrate packed RAW "
                f"NCHW in_ch={in_ch} {args.patch}x{args.patch}"
            ),
        }
        print(f"{tgt}: {static}", flush=True)

    (args.out / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {args.out / 'deploy_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
