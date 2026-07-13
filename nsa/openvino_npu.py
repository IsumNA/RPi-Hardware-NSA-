"""Intel NPU (AI Boost) runtime via OpenVINO.

Training stays on the host (PyTorch/CPU). This module:
  * points the process at a local Level Zero NPU driver prefix (no root install),
  * compiles ONNX -> OpenVINO IR for ``NPU``,
  * runs timed inference on the Intel NPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .config import project_root

_NPU_PREFIX = project_root() / ".npu_driver" / "prefix" / "usr" / "lib" / "x86_64-linux-gnu"
_RUNTIME_READY = False


def ensure_npu_runtime() -> Path | None:
    """Prepend the local Intel NPU Level Zero libs to ``LD_LIBRARY_PATH``.

    Returns the lib directory when present, else ``None``. Must run before the
    first ``openvino.Core()`` in this process for NPU discovery to work without
    a system-wide driver install.
    """
    global _RUNTIME_READY
    if not _NPU_PREFIX.is_dir():
        return None
    lib = str(_NPU_PREFIX.resolve())
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    if lib not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib + (":" + cur if cur else "")
    # OpenVINO may already be imported; still set env for child procs / dlopen.
    _RUNTIME_READY = True
    return _NPU_PREFIX


def npu_available() -> bool:
    """True when OpenVINO lists an ``NPU`` device."""
    ensure_npu_runtime()
    try:
        from openvino import Core
        return "NPU" in Core().available_devices
    except Exception:
        return False


def npu_device_name() -> str:
    ensure_npu_runtime()
    try:
        from openvino import Core
        core = Core()
        if "NPU" not in core.available_devices:
            return ""
        return str(core.get_property("NPU", "FULL_DEVICE_NAME"))
    except Exception:
        return ""


def compile_onnx_to_ir(onnx_path: Path, xml_path: Path, device: str = "NPU"):
    """Convert ONNX to OpenVINO IR and compile for ``device``.

    Writes ``xml_path`` (+ companion ``.bin``). Returns ``(compiled_model, xml_path)``.
    """
    ensure_npu_runtime()
    from openvino import Core, save_model, convert_model

    onnx_path = Path(onnx_path)
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    model = convert_model(str(onnx_path))
    save_model(model, str(xml_path))
    core = Core()
    if device not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device '{device}' not available "
            f"(have {core.available_devices}). Install Intel NPU Level Zero "
            f"libs under {_NPU_PREFIX} or system-wide."
        )
    compiled = core.compile_model(model, device)
    return compiled, xml_path


def load_compiled(xml_path: Path, device: str = "NPU"):
    ensure_npu_runtime()
    from openvino import Core
    core = Core()
    return core.compile_model(str(xml_path), device)


def infer_nchw(compiled, nchw: np.ndarray, warmup: int = 2,
               repeats: int = 5) -> tuple[np.ndarray, float]:
    """Run NCHW float32 inference; return ``(hwc [0,1], median_ms)``."""
    x = np.ascontiguousarray(nchw, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    infer = compiled.create_infer_request()
    for _ in range(max(0, warmup)):
        infer.infer({0: x})
    times = []
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = infer.infer({0: x})
        times.append((time.perf_counter() - t0) * 1000.0)
        out = next(iter(result.values()))
    y = np.asarray(out)
    if y.ndim == 4:
        y = y[0]
    hwc = np.transpose(y, (1, 2, 0)).astype(np.float32)
    hwc = np.clip(hwc, 0.0, 1.0)
    times.sort()
    return hwc, float(times[len(times) // 2])


def denoise_rgb_on_npu(compiled, rgb: np.ndarray, warmup: int = 1,
                       repeats: int = 3) -> tuple[np.ndarray, float]:
    """Denoise HxWx3 RGB in [0,1] on the compiled NPU model."""
    nchw = np.transpose(np.ascontiguousarray(rgb, dtype=np.float32), (2, 0, 1))[None]
    return infer_nchw(compiled, nchw, warmup=warmup, repeats=repeats)
