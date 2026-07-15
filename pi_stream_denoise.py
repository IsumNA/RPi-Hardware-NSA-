#!/usr/bin/env python3
"""Pi / laptop: denoise a live stream (or folder of DNGs) toward GT quality.

Loads ``stream_to_gt.pt`` or ``stream_to_gt.onnx`` and runs one packed frame
at a time — this is the deployment path, not burst averaging.

Usage::

  # Folder of stream frames
  python pi_stream_denoise.py --input /path/to/frames --checkpoint outputs/stream_to_gt.pt

  # ONNX on Pi
  python pi_stream_denoise.py --input ./burst --onnx outputs/stream_to_gt.onnx --out outputs/stream_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.raw_domain import load_packed, packed_to_rgb

DISPLAY_GAIN = 8.0


def _load_torch(ckpt: Path, device: str):
    import torch
    from nsa.raw_domain import RawDenoiser

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    model = RawDenoiser(
        base_channels=int(meta.get("base_channels", 64)),
        block_depth=int(meta.get("block_depth", 6)),
        in_ch=int(meta.get("in_ch", 4)),
        out_ch=int(meta.get("out_ch", 4)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, torch


def _run_torch(model, torch, packed: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(packed.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        y = model(x)
    return y[0].cpu().numpy().transpose(1, 2, 0)


def _run_onnx(sess, packed: np.ndarray) -> np.ndarray:
    x = packed.transpose(2, 0, 1)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: x})[0]
    return y[0].transpose(1, 2, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="folder of .dng stream frames (or a single .dng)")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/stream_to_gt.pt")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/stream_out")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(args.input.glob("*.dng"))[: args.max_frames]
    if not files:
        print(f"No DNGs in {args.input}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    use_onnx = args.onnx is not None and args.onnx.is_file()
    if use_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"])
        run = lambda pk: _run_onnx(sess, pk)
        print(f"ONNX {args.onnx}")
    else:
        if not args.checkpoint.is_file():
            print(f"Missing checkpoint {args.checkpoint}", file=sys.stderr)
            return 1
        model, torch = _load_torch(args.checkpoint, args.device)
        run = lambda pk: _run_torch(model, torch, pk, args.device)
        print(f"Torch {args.checkpoint} on {args.device}")

    for i, f in enumerate(files):
        packed = load_packed(f)
        out = np.clip(run(packed), 0, 1)
        rgb_in = packed_to_rgb(packed, DISPLAY_GAIN)
        rgb_out = packed_to_rgb(out, DISPLAY_GAIN)
        panel = np.concatenate([rgb_in, rgb_out], axis=1)
        path = args.out / f"stream_{i:04d}.png"
        Image.fromarray((panel * 255 + 0.5).astype(np.uint8)).save(path)
        print(f"  {f.name} → {path}")
    print(f"Done — {len(files)} frames → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
