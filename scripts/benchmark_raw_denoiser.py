#!/usr/bin/env python3
"""CPU latency benchmark for the 5-channel RawDenoiser (PyTorch or ONNX).

Laptop CPU timing is a rough proxy for Raspberry Pi 5 CPU inference before
on-device validation. Reports mean / p50 / p95 forward-pass latency.

Usage:
  python scripts/benchmark_raw_denoiser.py
  python scripts/benchmark_raw_denoiser.py --backend onnx --iters 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_raw_denoiser import load_raw_denoiser

DEFAULT_CKPT = ROOT / "outputs/raw_denoiser_5ch.pt"
DEFAULT_ONNX = ROOT / "outputs/raw_denoiser_5ch.onnx"


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def bench_torch(model: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(times)


def bench_onnx(onnx_path: Path, x: np.ndarray, warmup: int, iters: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    feed = {inp_name: x}

    for _ in range(warmup):
        sess.run(None, feed)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(times)


def _summarize(times: list[float]) -> dict:
    return {
        "mean_ms": round(float(np.mean(times)), 3),
        "p50_ms": round(_percentile(times, 50), 3),
        "p95_ms": round(_percentile(times, 95), 3),
        "min_ms": round(float(np.min(times)), 3),
        "max_ms": round(float(np.max(times)), 3),
        "iters": len(times),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--backend", choices=("torch", "onnx", "both"), default="both")
    p.add_argument("--height", type=int, default=192, help="packed spatial H")
    p.add_argument("--width", type=int, default=192, help="packed spatial W")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--threads", type=int, default=4,
                   help="PyTorch / ORT CPU thread count (Pi 5 has 4 cores)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/raw_denoiser_benchmark.json")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    h, w = args.height, args.width
    rng = np.random.default_rng(662)
    np_x = rng.random((1, 5, h, w), dtype=np.float32)
    torch_x = torch.from_numpy(np_x)

    results: dict = {
        "packed_shape": [1, 5, h, w],
        "rgb_equiv": f"{h * 2}x{w * 2} Bayer",
        "warmup": args.warmup,
        "iters": args.iters,
        "threads": args.threads,
        "backends": {},
    }

    if args.backend in ("torch", "both"):
        wrapper, meta = load_raw_denoiser(args.checkpoint.resolve(), torch.device("cpu"))
        core = wrapper.net if hasattr(wrapper, "net") else wrapper
        results["params"] = sum(p.numel() for p in wrapper.parameters())
        results["model"] = meta
        results["backends"]["torch_cpu"] = bench_torch(core, torch_x, args.warmup, args.iters)

    if args.backend in ("onnx", "both"):
        onnx_path = args.onnx.resolve()
        if not onnx_path.is_file():
            raise SystemExit(f"ONNX model not found: {onnx_path} (run export_raw_denoiser.py first)")
        results["onnx"] = str(onnx_path)
        results["backends"]["onnx_cpu"] = bench_onnx(onnx_path, np_x, args.warmup, args.iters)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Packed input   : {h}x{w} (RGB equiv ~{h*2}x{w*2})")
    print(f"Threads        : {args.threads}")
    for name, stats in results["backends"].items():
        print(f"{name:12s}  mean {stats['mean_ms']:7.3f} ms  "
              f"p50 {stats['p50_ms']:7.3f} ms  p95 {stats['p95_ms']:7.3f} ms")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
