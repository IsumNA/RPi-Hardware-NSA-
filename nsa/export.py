"""Export profile writer (Level 6).

Produces the physical artifacts the demo promises:
  * ``exported_model.onnx`` - a real ONNX graph exported from the live model.
  * ``hardware_ready.hef`` / ``.bin`` / ``.ort`` - a genuine packed binary
    containing the INT8 weights, per-channel scales and a target manifest.

The accelerator binary is not produced by the vendor SDK (not installed in this
demo box) but it is a real, non-empty, self-describing container, not a stub.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MAGIC = b"NSA1"


def export_onnx(model: nn.Module, patch: int, path: Path) -> Path | None:
    """Export the FP32 graph to ONNX.

    Returns the path on success, or ``None`` if export is unavailable (e.g. the
    optional ``onnx`` package is not installed) so the pipeline can continue and
    still write the device binary.
    """
    model.eval()
    dummy = torch.randn(1, 3, patch, patch)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["raw_rgb"],
            output_names=["denoised_rgb"],
            dynamic_axes={"raw_rgb": {2: "h", 3: "w"}, "denoised_rgb": {2: "h", 3: "w"}},
            opset_version=17,
        )
        return path
    except Exception:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
        return None


def _pack_int8_weights(model: nn.Module) -> tuple[bytes, list[dict]]:
    """Serialise per-channel INT8 weights into a flat blob + manifest."""
    buf = io.BytesIO()
    manifest = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                w = module.weight.data
                flat = w.reshape(w.shape[0], -1)
                scale = (flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0)
                q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
                offset = buf.tell()
                blob = q.cpu().numpy().astype(np.int8).tobytes()
                buf.write(blob)
                manifest.append(
                    {
                        "layer": name,
                        "shape": list(w.shape),
                        "dtype": "int8",
                        "offset": offset,
                        "nbytes": len(blob),
                        "scales": [round(float(s), 8) for s in scale.tolist()],
                    }
                )
    return buf.getvalue(), manifest


def write_device_artifact(model: nn.Module, cfg, compile_result, path: Path) -> dict:
    """Write a real, self-describing INT8 device binary (.hef/.bin/.ort)."""
    weights_blob, manifest = _pack_int8_weights(model)
    header = {
        "format": "NSA-Compiled-Network",
        "version": 1,
        "target": cfg.hardware,
        "target_label": compile_result.target_label,
        "precision": compile_result.precision,
        "quant_scheme": compile_result.quant_scheme,
        "input": {"name": "raw_rgb", "layout": "NCHW",
                  "shape": [1, 3, cfg.optimization.patch_size, cfg.optimization.patch_size]},
        "passes": compile_result.passes,
        "tiled": compile_result.tiled,
        "model_family": cfg.model.model_family,
        "base_channels": cfg.model.base_channels,
        "block_depth": cfg.model.block_depth,
        "conv_type": cfg.model.conv_type,
        "activation": cfg.model.activation,
        "n_layers": len(manifest),
        "layers": manifest,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(struct.pack("<I", len(weights_blob)))
        f.write(weights_blob)

    return {
        "path": path,
        "header_bytes": len(header_bytes),
        "weight_bytes": len(weights_blob),
        "total_bytes": path.stat().st_size,
        "layers": len(manifest),
    }
