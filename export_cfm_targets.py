#!/usr/bin/env python3
"""Export a CFM 1-step student for every hardware target in the project brief.

Targets:
  * cpu        — Pi 5 ONNX Runtime (dynamic H×W)
  * brainstorm — static ONNX + TFLite conversion notes (BCM2712 NPU)
  * hailo10h   — NSA INT8 ``.hef`` container + static ONNX for vendor DFC
  * deepx      — NSA INT8 ``.bin`` container + static ONNX for vendor SDK
  * intel_npu  — OpenVINO IR when available, else static ONNX

Usage::

  .venv/bin/python -u export_cfm_targets.py \\
      --checkpoint outputs/cfm_prod/cfm_student.pt \\
      --out outputs/cfm_deploy --int8

Stage A + B (detail head): use ``export_cfm_stage_b.py`` with ``--stage-a`` and
``--stage-b`` checkpoints (see deploy_manifest pipeline_snippet).
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

from nsa.config import Config, ModelConfig, OptimizationConfig
from nsa.compiler import compile_stack
from nsa.export import write_device_artifact
from nsa.flow_matching import ConsistencyStudent, BoundaryConsistencyWrapper
from nsa.inference import fake_quantize_int8
from train_stream_to_gt import _export_onnx


TARGETS = ("cpu", "brainstorm", "hailo10h", "deepx", "intel_npu")


def _load_deploy(ckpt: Path, device: str = "cpu") -> tuple[nn.Module, dict, int]:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    state = blob["state_dict"]
    gain_channel = bool(meta.get("gain_channel", False))
    temporal = int(meta.get("temporal", 4))
    in_ch = int(meta.get("in_ch", meta.get("cond_ch", 4 * temporal + int(gain_channel))))
    student = ConsistencyStudent(
        cond_ch=in_ch, out_ch=int(meta.get("out_ch", 4)),
        base_channels=int(meta.get("base_channels", 64)),
        block_depth=int(meta.get("block_depth", 6)),
        gain_channel=gain_channel,
    )
    if any(k.startswith("student.") for k in state):
        inner = {k[len("student."):]: v for k, v in state.items()
                 if k.startswith("student.")}
        student.load_state_dict(inner, strict=False)
    else:
        student.load_state_dict(state, strict=False)
    model = BoundaryConsistencyWrapper(student)
    model.eval()
    return model, meta, in_ch


def _export_static_onnx(model: nn.Module, in_ch: int, path: Path, patch: int) -> Path:
    model = model.cpu().eval()
    dummy = torch.randn(1, in_ch, patch, patch)
    kwargs = dict(
        input_names=["packed"],
        output_names=["packed_denoised"],
        opset_version=18,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cfm_deploy")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--int8", action="store_true",
                    help="fake-quantize before packing Hailo/DeepX blobs")
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        print(f"Missing checkpoint: {args.checkpoint}", file=sys.stderr)
        return 1

    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    model, meta, in_ch = _load_deploy(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source_checkpoint": str(args.checkpoint),
        "model": meta,
        "in_ch": in_ch,
        "params": n_params,
        "patch": args.patch,
        "targets": {},
    }

    dyn = args.out / "cfm_student_cpu.onnx"
    _export_onnx(model, in_ch, dyn, patch=args.patch)
    shutil.copy2(dyn, args.out / "cfm_student.onnx")
    manifest["targets"]["cpu"] = {
        "artifact": str(dyn),
        "runtime": "onnxruntime-cpu (also try NCNN/TFLite converters)",
        "precision": "fp32",
        "notes": "pi_live_cfm.py / pi_stream_denoise.py",
    }
    print(f"cpu: {dyn}", flush=True)

    for tgt in targets:
        if tgt == "cpu":
            continue
        tdir = args.out / tgt
        tdir.mkdir(parents=True, exist_ok=True)
        static = tdir / "cfm_student_static.onnx"
        _export_static_onnx(model, in_ch, static, args.patch)
        print(f"{tgt}: static ONNX {static}", flush=True)

        if tgt == "brainstorm":
            readme = tdir / "README_brainstorm.txt"
            readme.write_text(
                "Brainstorm (BCM2712 NPU) expects TFLite + the RPi NPU delegate.\n"
                "Convert static ONNX → TFLite (example):\n"
                "  onnx2tf -i cfm_student_static.onnx -o tflite_out\n"
                "On a Brainstorm-enabled Pi 5, load the .tflite with the delegate.\n"
                f"Input: NCHW float packed Bayer, in_ch={in_ch}, "
                f"{args.patch}x{args.patch}.\n"
            )
            manifest["targets"]["brainstorm"] = {
                "artifact": str(static),
                "runtime": "tflite + brainstorm delegate",
                "precision": "fp32/fp16",
                "notes": str(readme),
            }
            continue

        if tgt == "intel_npu":
            cfg = Config(
                hardware="intel_npu",
                model=ModelConfig(
                    model_family="raw_denoiser_stream",
                    base_channels=int(meta.get("base_channels", 64)),
                    block_depth=int(meta.get("block_depth", 6)),
                ),
                optimization=OptimizationConfig(
                    quantize=False, patch_size=args.patch),
            )
            try:
                cres = compile_stack(cfg, n_params)
                ir_base = tdir / "cfm_student"
                info = write_device_artifact(
                    model, cfg, cres, ir_base, onnx_path=static)
                manifest["targets"]["intel_npu"] = {
                    "artifact": str(info.get("path", ir_base)),
                    "runtime": "openvino-npu",
                    "precision": "fp16",
                }
                print(f"intel_npu: {info.get('path')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                fb = tdir / "cfm_student_npu_fallback.onnx"
                shutil.copy2(static, fb)
                manifest["targets"]["intel_npu"] = {
                    "artifact": str(fb),
                    "runtime": "openvino (compile on device)",
                    "precision": "fp16",
                    "notes": f"IR compile skipped: {exc}",
                }
                print(f"intel_npu fallback ({exc})", flush=True)
            continue

        if tgt in ("hailo10h", "deepx"):
            hw = tgt
            ext = ".hef" if tgt == "hailo10h" else ".bin"
            cfg = Config(
                hardware=hw,
                model=ModelConfig(
                    model_family="raw_denoiser_stream",
                    base_channels=int(meta.get("base_channels", 64)),
                    block_depth=int(meta.get("block_depth", 6)),
                ),
                optimization=OptimizationConfig(
                    quantize=True, qat=True, patch_size=args.patch),
            )
            cres = compile_stack(cfg, n_params)
            out_bin = tdir / f"cfm_student{ext}"
            pack_model = model
            if args.int8:
                inner = getattr(model, "student", model)
                fake_quantize_int8(inner)
            info = write_device_artifact(
                pack_model, cfg, cres, out_bin, onnx_path=static)
            (tdir / "COMPILE_NOTES.txt").write_text(
                f"NSA INT8 container: {out_bin.name}\n"
                f"For a vendor-loadable binary, run the {tgt} SDK / Dataflow "
                f"Compiler on cfm_student_static.onnx with a calibration set of "
                f"packed RAW tensors (NCHW, in_ch={in_ch}, "
                f"{args.patch}x{args.patch}).\n"
                f"Selected scheme: {cres.quant_scheme}\n"
            )
            manifest["targets"][tgt] = {
                "artifact": str(out_bin),
                "static_onnx": str(static),
                "runtime": f"{tgt} vendor SDK",
                "precision": "int8",
                "quant_scheme": cres.quant_scheme,
                "total_bytes": info.get("total_bytes"),
            }
            print(f"{tgt}: {out_bin} ({info.get('total_bytes')} bytes)", flush=True)

    (args.out / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {args.out / 'deploy_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
