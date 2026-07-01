"""Load and run frozen Hugging Face denoiser weights in the NSA pipeline.

Browse/freeze (``nsa.hub``) pins a commit SHA; this module downloads the snapshot
(if needed) and wires runnable weight files (ONNX / PyTorch) into compile,
live testing, and validation.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig, project_root
from .hub import HubError, LOCK_PATH, STORAGE_DIR, load_lock, model_details
from .models import build_model

RUNNABLE_SUFFIXES = (".onnx", ".pt", ".pth", ".safetensors", ".bin")
_WEIGHT_HINT = re.compile(
    r"(nafnet|dncnn|drunet|restormer|unet|denois|sidd|gopro|restoration)",
    re.I,
)


@dataclass
class HfRunSpec:
    model_id: str
    sha: str
    license: str
    local_dir: Path
    weight_path: Path
    weight_kind: str          # onnx | pytorch | imgutils
    imgutils_variant: str | None = None
    pretrained: bool = True


class OnnxDenoiser(nn.Module):
    """Wrap an ONNX graph so the rest of the stack can call ``model(tensor)``."""

    def __init__(self, onnx_path: Path):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise HubError(
                "running a Hugging Face ONNX model needs onnxruntime "
                "(pip install onnxruntime)."
            ) from exc
        self.onnx_path = Path(onnx_path)
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().float().numpy()
        out = self.session.run([self.output_name], {self.input_name: arr})[0]
        return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)


def _nafnet_dataset_tag(weight_name: str) -> str:
    low = weight_name.lower()
    if "gopro" in low:
        return "GoPro"
    if "reds" in low:
        return "REDS"
    return "SIDD"


class ImgutilsNafnetDenoiser(nn.Module):
    """NAFNet graphs from ``deepghs/image_restoration`` via ``dghs-imgutils``."""

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from imgutils.restore import restore_with_nafnet
        except ImportError as exc:
            raise HubError(
                "NAFNet models from deepghs/image_restoration need "
                "dghs-imgutils (pip install dghs-imgutils)."
            ) from exc
        from PIL import Image
        import numpy as np

        outs = []
        for b in range(x.shape[0]):
            arr = (x[b].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255)
            arr = arr.astype(np.uint8)
            restored = restore_with_nafnet(
                Image.fromarray(arr), model=self.variant, silent=True)
            o = np.array(restored, dtype=np.float32) / 255.0
            outs.append(torch.from_numpy(o).permute(2, 0, 1))
        return torch.stack(outs).to(device=x.device, dtype=x.dtype)


def lock_path(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / LOCK_PATH


def find_lock_entry(model_id: str, root: Path | None = None) -> dict | None:
    for e in load_lock(lock_path(root)):
        if e.get("id") == model_id:
            return e
    return None


def _local_snapshot_dir(model_id: str, root: Path) -> Path:
    return root / STORAGE_DIR / model_id.replace("/", "__")


def ensure_snapshot(
    model_id: str,
    sha: str | None = None,
    *,
    download: bool = True,
    root: Path | None = None,
    lock_file: Path | None = None,
) -> Path:
    """Return the on-disk snapshot directory, downloading when ``download`` is set."""
    root = root or project_root()
    entry = find_lock_entry(model_id, root)
    if entry and entry.get("local_path"):
        p = Path(entry["local_path"])
        if p.is_dir():
            return p

    dest = _local_snapshot_dir(model_id, root)
    if dest.is_dir() and any(dest.iterdir()):
        return dest

    if not download:
        raise HubError(
            f"'{model_id}' is frozen but not downloaded yet. "
            f"Use DOWNLOAD & USE in the GUI or: "
            f"python hf_search.py --freeze {model_id} --download"
        )

    from .hub import freeze_model
    lf = lock_file or lock_path(root)
    e = freeze_model(model_id, revision="main", lock_path=lf,
                     download=True, storage_dir=root / STORAGE_DIR)
    path = e.get("local_path")
    if path and Path(path).is_dir():
        return Path(path)
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    raise HubError(f"download finished but no files found for '{model_id}'.")


def list_runnable_weights(files: list[str]) -> list[str]:
    out: list[str] = []
    for name in files:
        low = name.lower()
        if low.endswith((".onnx", ".pt", ".pth", ".safetensors")):
            out.append(name)
        elif low.endswith(".bin") and "diffusion" not in low:
            out.append(name)
    return out


def pick_weight_file(
    files: list[str],
    *,
    family: str = "",
    hint: str = "",
) -> str | None:
    """Choose the best weight filename from a Hub repo listing."""
    runnable = list_runnable_weights(files)
    if not runnable:
        return None

    def score(name: str) -> tuple[int, str]:
        low = name.lower()
        s = 0
        if low.endswith(".onnx"):
            s += 40
        elif low.endswith((".pt", ".pth")):
            s += 30
        elif low.endswith(".safetensors"):
            s += 20
        if family and family in low:
            s += 25
        if hint and hint.lower() in low:
            s += 15
        if _WEIGHT_HINT.search(low):
            s += 10
        if "deblocking" in low or "grayscale" in low:
            s -= 5
        if "color" in low or "sidd" in low:
            s += 3
        return (-s, name)

    return sorted(runnable, key=score)[0]


def guess_family(model_id: str, weight_name: str) -> str:
    blob = f"{model_id}/{weight_name}".lower()
    for fam in ("nafnet", "drunet", "dncnn", "restormer", "unet", "ffdnet"):
        if fam in blob:
            return fam
    if "dncnn" in blob:
        return "dncnn"
    return "nafnet"


def _load_pytorch_into(model: nn.Module, path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, nn.Module):
        raise HubError(
            f"{path.name} is a full pickled module — export ONNX from the Hub "
            f"or pick a state-dict checkpoint instead."
        )
    if isinstance(ck, dict):
        if "state_dict" in ck:
            sd = ck["state_dict"]
        elif "model" in ck and isinstance(ck["model"], dict):
            sd = ck["model"]
        else:
            sd = ck
    else:
        raise HubError(f"unrecognised checkpoint format in {path.name}")

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing and len(missing) == len(list(model.state_dict())):
        raise HubError(
            f"checkpoint {path.name} did not match the built-in "
            f"{type(model).__name__} graph (all keys missing)."
        )


def load_hf_model(
    model_id: str,
    cfg: ModelConfig,
    *,
    weight: str | None = None,
    download: bool = True,
    root: Path | None = None,
) -> tuple[nn.Module, HfRunSpec]:
    """Download (if needed) and load a frozen Hub model for inference."""
    root = root or project_root()
    entry = find_lock_entry(model_id, root)
    sha = (entry or {}).get("sha")
    lic = (entry or {}).get("license", "?")

    if not entry:
        details = model_details(model_id)
        if details["license"] not in ("apache-2.0", "mit"):
            raise HubError(
                f"refusing to run '{model_id}': license '{details['license']}' "
                f"is not Apache-2.0 / MIT."
            )
        sha = details["sha"]
        lic = details["license"]

    local_dir = ensure_snapshot(model_id, sha, download=download, root=root)
    files = [p.name for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        files = (model_details(model_id).get("files") or [])

    weight_name = weight or pick_weight_file(
        files, family=cfg.model_family, hint=model_id)
    if not weight_name:
        raise HubError(
            f"no runnable ONNX/PyTorch weights found under '{model_id}'. "
            f"Pick a repo that ships .onnx or .pth files "
            f"(e.g. deepghs/image_restoration)."
        )

    weight_path = local_dir / weight_name
    if not weight_path.is_file():
        matches = list(local_dir.rglob(Path(weight_name).name))
        if not matches:
            raise HubError(f"weight file missing after download: {weight_name}")
        weight_path = matches[0]

    low = weight_path.name.lower()
    variant = None
    if "deepghs/image_restoration" in model_id.lower() and "nafnet" in low:
        variant = _nafnet_dataset_tag(weight_path.name)
        model = ImgutilsNafnetDenoiser(variant)
        kind = "imgutils"
    elif low.endswith(".onnx"):
        model = OnnxDenoiser(weight_path)
        kind = "onnx"
    else:
        fam = guess_family(model_id, weight_path.name)
        if fam != cfg.model_family:
            cfg.model_family = fam
        if "width64" in low or "width32" in low:
            m = re.search(r"width(\d+)", low)
            if m:
                w = int(m.group(1))
                if w in (16, 32, 64):
                    cfg.base_channels = w
        model = build_model(cfg)
        _load_pytorch_into(model, weight_path)
        kind = "pytorch"

    spec = HfRunSpec(
        model_id=model_id,
        sha=sha or "",
        license=lic,
        local_dir=local_dir,
        weight_path=weight_path,
        weight_kind=kind,
        imgutils_variant=variant if kind == "imgutils" else None,
    )
    model.eval()
    return model, spec


def copy_hf_onnx(spec: HfRunSpec, dest: Path) -> bool:
    if spec.weight_kind != "onnx":
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.weight_path, dest)
    return dest.is_file()


def is_hf_pretrained(cfg: ModelConfig) -> bool:
    return bool(getattr(cfg, "hf_model", None))


def calibration_steps_for_hf(cfg, default_steps: int) -> int:
    """Pretrained Hub weights should not be re-trained from scratch."""
    if not is_hf_pretrained(cfg):
        return default_steps
    return min(default_steps, 30)
