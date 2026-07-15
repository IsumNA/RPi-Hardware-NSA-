#!/usr/bin/env python3
"""Phase 5 — end-to-end hybrid pipeline benchmark (CPU).

Compares four reconstruction paths on a real burst:
  1. single_frame      — first DNG only
  2. naive_mean        — temporal average (no motion gating)
  3. fusion_only       — motion-aware packed-RAW fusion
  4. fusion_denoise    — fusion + 5ch RawDenoiser (torch or onnx)

Reports PSNR/SSIM vs burst_mean ground truth and per-stage latency on CPU.
Optionally checks Pi HCG sync readiness on the AI host (one SSH).

Usage:
  python benchmark_hybrid_pipeline.py
  python benchmark_hybrid_pipeline.py --backend onnx --json outputs/hybrid_benchmark.json
  python benchmark_hybrid_pipeline.py --check-pi-sync --start-hcg-training  # rare
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from export_raw_denoiser import load_raw_denoiser
from nsa.inference import psnr, ssim, to_image, to_tensor
from nsa.raw_domain import (
    burst_clean,
    load_packed,
    packed_to_rgb,
    stack_fusion_input,
)
from nsa.temporal_fusion import FusionConfig, fuse_burst_packed

DISPLAY_GAIN = 8.0
DEFAULT_BURST = ROOT / "datasets/imx662_project/bursts/cabinet_D50_100/ag128"
DEFAULT_CKPT = ROOT / "outputs/raw_denoiser_5ch.pt"
DEFAULT_ONNX = ROOT / "outputs/raw_denoiser_5ch.onnx"
DEFAULT_JSON = ROOT / "outputs/hybrid_benchmark.json"
FUSION_FRAMES = 12
GT_BURST_FRAMES = 256
AI_HOST = "ai"
PI_CACHE = "/opt/datasets/PI_RAW/Pi_Unique_Cache"
HCG_SYNC_THRESHOLD = 50.0


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _summarize_latency(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(float(np.mean(times)), 3),
        "p50_ms": round(_percentile(times, 50), 3),
        "p95_ms": round(_percentile(times, 95), 3),
    }


def _bench(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _summarize_latency(times)


def _to_rgb(packed: np.ndarray) -> np.ndarray:
    return packed_to_rgb(packed, DISPLAY_GAIN)


def _crop_match(*images: np.ndarray) -> list[np.ndarray]:
    h = min(*(a.shape[0] for a in images))
    w = min(*(a.shape[1] for a in images))
    return [a[:h, :w] for a in images]


def _metrics(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> dict[str, float]:
    p, g = _crop_match(pred_rgb, gt_rgb)
    return {
        "psnr": round(psnr(p, g), 2),
        "ssim": round(ssim(p, g), 4),
    }


def _check_pi_sync(host: str = AI_HOST, cache: str = PI_CACHE) -> dict[str, Any]:
    """Single SSH readiness probe (mirrors scripts/watch_pi_sync.sh)."""
    remote_script = f"""cd ~/RPi-Hardware-NSA-
.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
cache = Path({json.dumps(cache)})
repo = Path.home() / "RPi-Hardware-NSA-"
sys.path.insert(0, str(repo))
from nsa.dataset_align import build_hcg_sort_manifest, cache_readiness, project_json_in_cache
pj = project_json_in_cache(cache)
out = {{
    "project_json_present": pj.is_file(),
    "coverage_pct": 0.0,
    "present_files": 0,
    "wanted_files": 0,
    "status": "waiting",
}}
if pj.is_file():
    manifest = build_hcg_sort_manifest(pj)
    report = cache_readiness(cache, manifest)
    cov = round(100.0 * report["fraction"], 2)
    out.update({{
        "coverage_pct": cov,
        "present_files": report["present_files"],
        "wanted_files": report["wanted_files"],
        "status": "ready" if cov >= {HCG_SYNC_THRESHOLD} else "syncing",
    }})
print(json.dumps(out))
PY"""
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        host, remote_script,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode != 0:
            return {
                "reachable": False,
                "error": (proc.stderr or proc.stdout or "ssh failed").strip()[:200],
                "coverage_pct": None,
                "status": "unreachable",
            }
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        data["reachable"] = True
        return data
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        return {
            "reachable": False,
            "error": str(exc),
            "coverage_pct": None,
            "status": "unreachable",
        }


def _maybe_start_hcg_training(
    sync: dict[str, Any],
    *,
    user_flag: bool,
    host: str = AI_HOST,
) -> dict[str, Any] | None:
    """Launch HCG training on AI only when sync > threshold AND user opted in."""
    if not user_flag:
        return None
    cov = sync.get("coverage_pct")
    if cov is None or float(cov) < HCG_SYNC_THRESHOLD:
        return {
            "started": False,
            "reason": f"coverage {cov}% < {HCG_SYNC_THRESHOLD}% threshold",
        }
    remote = (
        f"cd ~/RPi-Hardware-NSA- && mkdir -p outputs/train_logs && "
        "STAMP=$(date +%Y%m%d-%H%M%S) && "
        "LOG=outputs/train_logs/train_raw_visual_hcg_${STAMP}.log && "
        "nohup .venv/bin/python -u train_raw_visual.py "
        "--burst-scene cabinet_H_2 --gains 128,256,512 "
        "--panel-every 50 --panel-dir outputs/raw_panels --force "
        ">\"$LOG\" 2>&1 & echo TRAIN_LOG=$LOG"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, remote],
        capture_output=True, text=True, timeout=30, check=False,
    )
    return {
        "started": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip() if proc.returncode != 0 else "",
    }


def run_benchmark(
    burst_dir: Path,
    ckpt_path: Path,
    onnx_path: Path,
    *,
    backend: str = "torch",
    n_frames: int = FUSION_FRAMES,
    warmup: int = 5,
    iters: int = 20,
    threads: int = 4,
) -> dict[str, Any]:
    burst_dir = burst_dir.resolve()
    dngs = sorted(burst_dir.glob("*.dng"))
    if len(dngs) < 8:
        raise FileNotFoundError(f"Only {len(dngs)} DNGs in {burst_dir}")

    torch.set_num_threads(threads)
    device = torch.device("cpu")

    cfg = FusionConfig(n_frames=min(n_frames, len(dngs)), k_cap=16.0)

    # --- ground truth (burst temporal mean, many frames) ---
    t0 = time.perf_counter()
    gt_packed = burst_clean(dngs, limit=GT_BURST_FRAMES)
    gt_build_ms = (time.perf_counter() - t0) * 1000.0
    gt_rgb = _to_rgb(gt_packed)

    # Pre-load frames once for fair stage timing (I/O excluded from fusion bench).
    frames = [load_packed(p) for p in dngs[: cfg.n_frames]]
    h, w = frames[0].shape[0], frames[0].shape[1]

    # --- single frame ---
    single_packed = frames[0]
    single_rgb = _to_rgb(single_packed)

    # --- naive mean ---
    def _naive():
        acc = None
        for fr in frames:
            acc = fr if acc is None else acc + fr
        return acc / len(frames)

    naive_stats = _bench(_naive, warmup, iters)
    naive_packed = _naive()
    naive_rgb = _to_rgb(naive_packed)

    # --- motion fusion ---
    def _fuse():
        return fuse_burst_packed(frames, cfg)

    fusion_stats = _bench(_fuse, warmup, iters)
    fused_packed, weight = _fuse()
    fused_rgb = _to_rgb(fused_packed)
    x5 = stack_fusion_input(fused_packed, weight, k_cap=cfg.k_cap)

    # --- denoiser ---
    denoised_packed: np.ndarray | None = None
    denoise_stats: dict[str, float] | None = None
    denoise_backend = backend

    if backend in ("torch", "both"):
        wrapper, meta = load_raw_denoiser(ckpt_path.resolve(), device)
        core = wrapper.net if hasattr(wrapper, "net") else wrapper
        core.eval()
        tx = to_tensor(x5)

        def _denoise_torch():
            with torch.no_grad():
                return to_image(core(tx))

        ds = _bench(_denoise_torch, warmup, iters)
        denoised_packed = _denoise_torch()
        if backend == "torch":
            denoise_stats = ds
        else:
            denoise_stats = {"torch_cpu": ds, "model": meta}

    if backend in ("onnx", "both"):
        import onnxruntime as ort

        onnx_file = onnx_path.resolve()
        if not onnx_file.is_file():
            raise FileNotFoundError(f"ONNX missing: {onnx_file}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"],
        )
        inp_name = sess.get_inputs()[0].name
        np_x = to_tensor(x5).numpy()

        def _denoise_onnx():
            out = sess.run(None, {inp_name: np_x})[0]
            return to_image(torch.from_numpy(out))

        ds = _bench(_denoise_onnx, warmup, iters)
        if backend == "onnx":
            denoised_packed = _denoise_onnx()
            denoise_stats = ds
        elif isinstance(denoise_stats, dict):
            denoise_stats["onnx_cpu"] = ds
            if denoised_packed is None:
                denoised_packed = _denoise_onnx()

    if denoised_packed is None:
        raise RuntimeError("denoiser backend produced no output")

    denoised_rgb = _to_rgb(denoised_packed)

    # Full pipeline: fusion + denoise (single pass, not load/GT).
    def _pipeline():
        fp, wt = fuse_burst_packed(frames, cfg)
        inp = stack_fusion_input(fp, wt, k_cap=cfg.k_cap)
        if backend == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                str(onnx_path.resolve()), opts, providers=["CPUExecutionProvider"],
            )
            inp_name = sess.get_inputs()[0].name
            out = sess.run(None, {inp_name: to_tensor(inp).numpy()})[0]
            return to_image(torch.from_numpy(out))
        with torch.no_grad():
            return to_image(core(to_tensor(inp)))

    pipeline_stats = _bench(_pipeline, warmup, iters)

    modes = {
        "single_frame": _metrics(single_rgb, gt_rgb),
        "naive_mean": _metrics(naive_rgb, gt_rgb),
        "fusion_only": _metrics(fused_rgb, gt_rgb),
        "fusion_denoise": _metrics(denoised_rgb, gt_rgb),
    }

    return {
        "burst_dir": str(burst_dir),
        "n_dngs": len(dngs),
        "n_frames": cfg.n_frames,
        "packed_shape": [h, w, 4],
        "rgb_equiv": f"{h * 2}x{w * 2}",
        "gt": "burst_mean",
        "gt_frames": min(GT_BURST_FRAMES, len(dngs)),
        "gt_build_ms": round(gt_build_ms, 1),
        "device": "cpu",
        "threads": threads,
        "denoise_backend": denoise_backend,
        "checkpoint": str(ckpt_path.resolve()),
        "onnx": str(onnx_path.resolve()) if backend in ("onnx", "both") else None,
        "warmup": warmup,
        "iters": iters,
        "modes": modes,
        "latency_ms": {
            "naive_mean": naive_stats,
            "fusion": fusion_stats,
            "denoise": denoise_stats,
            "fusion_plus_denoise": pipeline_stats,
        },
    }


def _print_table(result: dict[str, Any]) -> None:
    modes = result["modes"]
    lat = result["latency_ms"]

    print(f"\nHybrid pipeline benchmark  (GT: {result['gt']}, {result['n_frames']} frames)")
    print(f"Burst: {result['burst_dir']}")
    print(f"Packed {result['packed_shape'][:2]}  RGB ~{result['rgb_equiv']}  CPU threads={result['threads']}")
    print()
    print(f"{'Mode':<18} {'PSNR':>8} {'SSIM':>8}")
    print("-" * 36)
    for name in ("single_frame", "naive_mean", "fusion_only", "fusion_denoise"):
        m = modes[name]
        print(f"{name:<18} {m['psnr']:>7.2f} {m['ssim']:>8.4f}")

    print(f"\n{'Stage':<22} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10}")
    print("-" * 54)
    for stage, key in (
        ("naive mean", "naive_mean"),
        ("motion fusion", "fusion"),
        ("denoise", "denoise"),
        ("fusion + denoise", "fusion_plus_denoise"),
    ):
        stats = lat[key]
        if isinstance(stats, dict) and "mean_ms" not in stats:
            for sub, sub_stats in stats.items():
                if sub == "model":
                    continue
                if isinstance(sub_stats, dict) and "mean_ms" in sub_stats:
                    print(f"{stage + ' (' + sub + ')':<22} "
                          f"{sub_stats['mean_ms']:>10.3f} "
                          f"{sub_stats['p50_ms']:>10.3f} "
                          f"{sub_stats['p95_ms']:>10.3f}")
        else:
            print(f"{stage:<22} {stats['mean_ms']:>10.3f} "
                  f"{stats['p50_ms']:>10.3f} {stats['p95_ms']:>10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 5 hybrid pipeline benchmark (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--burst-dir", type=Path, default=DEFAULT_BURST)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--backend", choices=("torch", "onnx", "both"), default="torch")
    ap.add_argument("--frames", type=int, default=FUSION_FRAMES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", nargs="?", const=str(DEFAULT_JSON), default=None,
                    metavar="PATH", help=f"write JSON report (default: {DEFAULT_JSON})")
    ap.add_argument("--no-json", action="store_true", help="skip JSON output")
    ap.add_argument("--check-pi-sync", action="store_true", default=True,
                    help="probe AI Pi cache sync (default: on)")
    ap.add_argument("--no-pi-sync", action="store_true",
                    help="skip Pi sync SSH check")
    ap.add_argument("--ai-host", default=AI_HOST)
    ap.add_argument("--start-hcg-training", action="store_true",
                    help=f"launch HCG train on AI when sync > {HCG_SYNC_THRESHOLD:.0f}%%")
    args = ap.parse_args()

    if args.no_pi_sync:
        args.check_pi_sync = False

    out: dict[str, Any] = {}

    if args.check_pi_sync:
        sync = _check_pi_sync(args.ai_host)
        out["pi_sync"] = sync
        cov = sync.get("coverage_pct")
        if sync.get("reachable"):
            print(f"Pi HCG sync ({args.ai_host}): {cov}% "
                  f"({sync.get('present_files', '?')}/{sync.get('wanted_files', '?')} files)  "
                  f"status={sync.get('status')}")
        else:
            print(f"Pi HCG sync ({args.ai_host}): unreachable — {sync.get('error', '?')}")

        hcg = _maybe_start_hcg_training(sync, user_flag=args.start_hcg_training, host=args.ai_host)
        if hcg is not None:
            out["hcg_training"] = hcg
            if hcg.get("started"):
                print(f"HCG training started: {hcg.get('stdout', '')}")
            else:
                print(f"HCG training not started: {hcg.get('reason', hcg.get('stderr', '?'))}")
        elif args.start_hcg_training:
            pass
        elif cov is not None and float(cov) >= HCG_SYNC_THRESHOLD:
            print(f"  (sync >= {HCG_SYNC_THRESHOLD:.0f}% — pass --start-hcg-training to launch on AI)")

    try:
        bench = run_benchmark(
            args.burst_dir, args.checkpoint, args.onnx,
            backend=args.backend,
            n_frames=args.frames,
            warmup=args.warmup,
            iters=args.iters,
            threads=args.threads,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out["benchmark"] = bench
    _print_table(bench)

    if not args.no_json and args.json is not None:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
