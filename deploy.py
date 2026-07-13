#!/usr/bin/env python3
"""NSA Deployment Package Builder
================================
Bundles the compiled artifacts in ``outputs/`` into a self-contained, versioned
deployment package (``outputs/deployment/`` + a ``.zip``) ready to hand to a
device-integration step.

What it produces:
  * the compiled device binary (``hardware_ready.hef`` / ``.bin`` / ``.ort``),
  * the FP32 ONNX baseline (if present),
  * a ``manifest.json`` describing target, model, precision and metrics, and
  * ``FLASH_INSTRUCTIONS.md`` with the exact vendor commands to load it.

Note: actually flashing/booting the network on silicon requires the vendor SDK
(Hailo Dataflow Compiler / DeepX toolchain) and the physical accelerator, which
are not present in this environment. This builder prepares everything up to that
hand-off, so the only remaining step on a real device is running the printed
vendor command.

Examples
--------
  python run_demo.py --hardware hailo8 --no-window   # produce artifacts first
  python deploy.py                                   # then package them
  python deploy.py --name imx662_nafnet_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from nsa.theme import banner, console, log

OUT = Path("outputs")

# Vendor-specific load commands (run on the target with the vendor SDK present).
FLASH = {
    "hailo8": (
        "Hailo-8 (Raspberry Pi 5 AI Kit)",
        [
            "# On the Pi 5 with HailoRT installed:",
            "hailortcli run hardware_ready.hef --input-files raw_rgb=frame.npy",
            "# or load it from your app via the HailoRT Python API:",
            "#   from hailo_platform import HEF; hef = HEF('hardware_ready.hef')",
        ],
    ),
    "deepx": (
        "DeepX DX-M1 NPU",
        [
            "# On the host/device with the DeepX runtime installed:",
            "dxrt-cli --model hardware_ready.bin --input frame.npy",
            "# or load it via the DeepX runtime SDK in your application.",
        ],
    ),
    "rpi5_cpu": (
        "Raspberry Pi 5 (CPU, ONNX Runtime)",
        [
            "# On the Pi 5 with onnxruntime installed:",
            "python -c \"import onnxruntime as ort; ort.InferenceSession('exported_model.onnx')\"",
        ],
    ),
    "intel_npu": (
        "Intel AI Boost (NPU via OpenVINO)",
        [
            "# On this host with OpenVINO + Intel NPU driver:",
            "python -c \"from openvino import Core; print(Core().available_devices)\"",
            "# Then: Core().compile_model('hardware_ready.xml', 'NPU')",
        ],
    ),
}


def build_package(summary: dict, out: Path, name: str | None = None,
                  make_zip: bool = True) -> dict:
    """Bundle the compiled artifacts described by ``summary`` into a transferable
    package under ``out/deployment/<name>/`` (+ a ``.zip``).

    Returns a dict with ``pkg`` (dir), ``zip`` (archive path or None),
    ``files`` and ``label``. Raises ``FileNotFoundError`` if the device binary
    for the target is missing.
    """
    target = summary.get("hardware", "hailo8")
    ext = {"hailo8": ".hef", "deepx": ".bin", "rpi5_cpu": ".ort",
           "intel_npu": ".xml"}.get(target, ".bin")
    binary = out / f"hardware_ready{ext}"
    onnx = out / "exported_model.onnx"
    panel = out / "validation_panel.png"
    if not binary.exists():
        raise FileNotFoundError(f"missing device binary {binary}")

    name = name or (f"{summary.get('sensor_key','sensor')}_"
                    f"{summary.get('model',{}).get('family','model')}_{target}")
    pkg = out / "deployment" / name
    if pkg.exists():
        shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True, exist_ok=True)

    shutil.copy(binary, pkg / binary.name)
    # OpenVINO IR companion weights
    companion = binary.with_suffix(".bin")
    if target == "intel_npu" and companion.exists():
        shutil.copy(companion, pkg / companion.name)
    if onnx.exists():
        shutil.copy(onnx, pkg / onnx.name)
    if panel.exists():
        shutil.copy(panel, pkg / panel.name)

    manifest = {
        "package": name,
        "target": target,
        "target_label": summary.get("hardware_name", ""),
        "precision": summary.get("precision", ""),
        "quant_scheme": summary.get("quant_scheme", ""),
        "model": summary.get("model", {}),
        "sensor": summary.get("sensor", ""),
        "metrics": {
            "psnr_out": summary.get("psnr_out"),
            "latency_ms": summary.get("latency_ms"),
            "fps": summary.get("fps"),
            "fitness": summary.get("fitness"),
            "grade": summary.get("grade"),
            "weight_kb": summary.get("weight_kb"),
            "act_kb": summary.get("act_kb"),
        },
        "artifacts": [p.name for p in pkg.iterdir()],
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    label, cmds = FLASH.get(target, FLASH["hailo8"])
    instructions = "\n".join([
        f"# Deploy `{name}` to {label}",
        "",
        "## Package contents",
        *[f"- `{p}`" for p in manifest["artifacts"]],
        "",
        "## Metrics",
        f"- PSNR: {manifest['metrics']['psnr_out']} dB",
        f"- Latency: {manifest['metrics']['latency_ms']} ms "
        f"({manifest['metrics']['fps']} FPS, estimated)",
        f"- Fitness: {manifest['metrics']['fitness']} / 100 "
        f"({manifest['metrics']['grade']})",
        "",
        "## Load on the target device",
        "```bash",
        *cmds,
        "```",
        "",
        "> Requires the vendor SDK + the physical accelerator. This package "
        "prepares everything up to that hand-off.",
    ])
    (pkg / "FLASH_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")

    archive = None
    if make_zip:
        archive = shutil.make_archive(str(out / "deployment" / name), "zip",
                                      root_dir=pkg)

    return {"pkg": pkg, "zip": archive, "label": label,
            "files": manifest["artifacts"]}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy.py",
        description="NSA deployment package builder.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--name", default=None,
                   help="package name (default derived from target + model)")
    p.add_argument("--outputs", default="outputs", help="artifacts directory")
    p.add_argument("--no-zip", dest="no_zip", action="store_true",
                   help="do not create the .zip archive")
    return p


def main() -> int:
    args = build_parser().parse_args()
    banner("NSA Deployment Package")

    out = Path(args.outputs)
    summary_path = out / "summary.json"
    if not summary_path.exists():
        log("No outputs/summary.json found. Run the pipeline first: "
            "python run_demo.py --no-window", "warn")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        res = build_package(summary, out, name=args.name, make_zip=not args.no_zip)
    except FileNotFoundError as exc:
        log(f"{exc} — run the pipeline first (python run_demo.py --no-window).", "warn")
        return 1

    log(f"Deployment package built -> {res['pkg']}", "ok")
    for p in sorted(Path(res["pkg"]).iterdir()):
        log(f"  · {p.name}  ({p.stat().st_size/1024:.1f} KB)", "info")
    if res["zip"]:
        log(f"Archived -> {res['zip']}  "
            f"({Path(res['zip']).stat().st_size/1024:.1f} KB)", "ok")

    console.print(f"\n  Hand-off ready. On the device run the command in "
                  f"[bold]{(Path(res['pkg'])/'FLASH_INSTRUCTIONS.md').name}[/].")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
