#!/usr/bin/env python3
"""Export trained 5-channel RawDenoiser checkpoint to ONNX (and optional TorchScript).

Usage:
  python export_raw_denoiser.py
  python export_raw_denoiser.py --checkpoint outputs/raw_denoiser_5ch.pt --torchscript
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.config import ModelConfig
from nsa.export import export_raw_onnx, export_torchscript
from nsa.models import build_model, count_params

DEFAULT_CKPT = ROOT / "outputs/raw_denoiser_5ch.pt"
DEFAULT_OUT = ROOT / "outputs/raw_denoiser_5ch.onnx"


def load_raw_denoiser(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("model", {})
    cfg = ModelConfig(
        model_family="raw_denoiser",
        base_channels=int(meta.get("base_channels", 16)),
        block_depth=int(meta.get("block_depth", 4)),
    )
    wrapper = build_model(cfg)
    state = ckpt["state_dict"]
    if not any(k.startswith("net.") for k in state) and hasattr(wrapper, "net"):
        state = {f"net.{k}": v for k, v in state.items()}
    wrapper.load_state_dict(state)
    wrapper.eval()
    return wrapper, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patch", type=int, default=192,
                   help="reference spatial size for export trace (dynamic H/W)")
    p.add_argument("--fixed-shape", action="store_true",
                   help="disable dynamic spatial axes (for NPU compile)")
    p.add_argument("--torchscript", action="store_true",
                   help="also write a TorchScript .pt trace")
    args = p.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    wrapper, meta = load_raw_denoiser(ckpt_path, torch.device("cpu"))
    core = wrapper.net if hasattr(wrapper, "net") else wrapper
    in_ch = int(meta.get("in_ch", 5))
    out_ch = int(meta.get("out_ch", 4))
    n_params = count_params(wrapper)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = export_raw_onnx(
        core, args.patch, out_path,
        in_ch=in_ch, out_ch=out_ch,
        dynamic=not args.fixed_shape,
    )
    if onnx_path is None:
        raise SystemExit("ONNX export failed (is the onnx package installed?)")

    ts_path = None
    if args.torchscript:
        ts_path = export_torchscript(
            core, args.patch,
            out_path.with_suffix(".torchscript.pt"),
            in_ch=in_ch,
        )
        if ts_path is None:
            print("Warning: TorchScript export failed", file=sys.stderr)

    manifest = {
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path),
        "torchscript": str(ts_path) if ts_path else None,
        "in_ch": in_ch,
        "out_ch": out_ch,
        "params": n_params,
        "patch": args.patch,
        "dynamic_spatial": not args.fixed_shape,
        "opset": 18,
        "input_name": "packed_fusion",
        "output_name": "packed_denoised",
    }
    manifest_path = out_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported ONNX  : {onnx_path} ({onnx_path.stat().st_size:,} bytes)")
    if ts_path:
        print(f"Exported TS    : {ts_path} ({ts_path.stat().st_size:,} bytes)")
    print(f"Manifest       : {manifest_path}")
    print(f"Params         : {n_params:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
