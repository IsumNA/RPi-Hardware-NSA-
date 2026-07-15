#!/usr/bin/env python3
"""Pi / laptop: denoise a live stream (or folder of DNGs) toward GT quality.

Loads ``stream_to_gt.pt`` or ``stream_to_gt.onnx`` and runs a short live window
(or a single packed frame) — this is the deployment path, not burst averaging.

When the checkpoint was trained with ``temporal>1``, frames are channel-stacked
as [current, t-1, …] (4T channels). A ring buffer holds the last T live frames;
there is still no 500-frame average at inference.

Usage::

  # Folder of stream frames
  python pi_stream_denoise.py --input /path/to/frames --checkpoint outputs/stream_to_gt.pt

  # ONNX on Pi
  python pi_stream_denoise.py --input ./burst --onnx outputs/stream_to_gt.onnx --out outputs/stream_out
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.raw_domain import load_packed, packed_to_rgb

DISPLAY_GAIN = 8.0


def _stack_ring(ring: deque) -> np.ndarray:
    """[current, t-1, …] channel stack from ring (newest last in deque)."""
    frames = list(ring)
    # ring is oldest→newest; we want current first
    ordered = list(reversed(frames))
    return np.concatenate(ordered, axis=-1)


def _load_torch(ckpt: Path, device: str):
    import torch
    from nsa.raw_domain import RawDenoiser

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    temporal = int(meta.get("temporal", 1))
    in_ch = int(meta.get("in_ch", 4 * temporal))
    out_ch = int(meta.get("out_ch", 4))
    model = RawDenoiser(
        base_channels=int(meta.get("base_channels", 64)),
        block_depth=int(meta.get("block_depth", 6)),
        in_ch=in_ch,
        out_ch=out_ch,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, torch, temporal, in_ch


def _run_torch(model, torch, stacked: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(stacked.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        y = model(x)
    return y[0].cpu().numpy().transpose(1, 2, 0)


def _run_onnx(sess, stacked: np.ndarray) -> np.ndarray:
    x = stacked.transpose(2, 0, 1)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: x})[0]
    return y[0].transpose(1, 2, 0)


def _infer_temporal_from_onnx(sess) -> int:
    shape = sess.get_inputs()[0].shape
    # [N, C, H, W] — C may be dynamic str
    try:
        c = int(shape[1])
        return max(1, c // 4)
    except (TypeError, ValueError, IndexError):
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="folder of .dng stream frames (or a single .dng)")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/stream_to_gt.pt")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/stream_out")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temporal", type=int, default=0,
                    help="override stack length (0 = read from checkpoint/ONNX)")
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
    temporal = max(0, int(args.temporal))

    if use_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"])
        if temporal <= 0:
            temporal = _infer_temporal_from_onnx(sess)
        run = lambda st: _run_onnx(sess, st)
        print(f"ONNX {args.onnx}  temporal={temporal}")
    else:
        if not args.checkpoint.is_file():
            print(f"Missing checkpoint {args.checkpoint}", file=sys.stderr)
            return 1
        model, torch, ckpt_t, in_ch = _load_torch(args.checkpoint, args.device)
        if temporal <= 0:
            temporal = ckpt_t
        run = lambda st: _run_torch(model, torch, st, args.device)
        print(f"Torch {args.checkpoint} on {args.device}  "
              f"temporal={temporal} in_ch={in_ch}")

    ring: deque = deque(maxlen=temporal)
    for i, f in enumerate(files):
        packed = load_packed(f)
        if not ring:
            # Cold-start: pad with the first frame
            for _ in range(temporal):
                ring.append(packed)
        else:
            ring.append(packed)
        stacked = _stack_ring(ring)
        out = np.clip(run(stacked), 0, 1)
        rgb_in = packed_to_rgb(packed, DISPLAY_GAIN)
        rgb_out = packed_to_rgb(out, DISPLAY_GAIN)
        panel = np.concatenate([rgb_in, rgb_out], axis=1)
        path = args.out / f"stream_{i:04d}.png"
        Image.fromarray((panel * 255 + 0.5).astype(np.uint8)).save(path)
        print(f"  {f.name} → {path}")
    print(f"Done — {len(files)} frames → {args.out} (T={temporal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
