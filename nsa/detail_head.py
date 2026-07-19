"""Stage B detail refiner: tanh-bounded HF residual on frozen Stage A output."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import BoundaryConsistencyWrapper, ConsistencyStudent
from .models import _NAFBlock


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise isotropic Gaussian blur (sigma in pixels)."""
    if sigma <= 1e-6:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    k = 2 * radius + 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - radius
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    c = x.shape[1]
    kh = g1.view(1, 1, 1, k).expand(c, 1, 1, k)
    kv = g1.view(1, 1, k, 1).expand(c, 1, k, 1)
    x = F.conv2d(x, kh, padding=(0, radius), groups=c)
    x = F.conv2d(x, kv, padding=(radius, 0), groups=c)
    return x


def highpass_noisy(noisy: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """High-pass of the live noisy frame (first 4 packed channels)."""
    return noisy[:, :4] - _gaussian_blur(noisy[:, :4], sigma)


class DetailHead(nn.Module):
    """Few NAF blocks: predict a bounded residual added to Stage A output.

    Input is bilinear-aligned Stage A packed RAW (4ch), optionally concatenated
    with a high-pass of the noisy reference from ``cond``.
    """

    def __init__(
        self,
        out_ch: int = 4,
        base_channels: int = 32,
        block_depth: int = 3,
        *,
        use_noisy_hf: bool = True,
        residual_scale: float = 0.12,
        hf_sigma: float = 1.2,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.use_noisy_hf = bool(use_noisy_hf)
        self.residual_scale = float(residual_scale)
        self.hf_sigma = float(hf_sigma)
        in_ch = out_ch + (out_ch if self.use_noisy_hf else 0)
        c = base_channels
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(block_depth)])
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(
        self,
        stage_a: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = stage_a
        if self.use_noisy_hf and cond is not None:
            x = torch.cat([x, highpass_noisy(cond, self.hf_sigma)], dim=1)
        h = self.body(self.head(x))
        residual = torch.tanh(self.tail(h)) * self.residual_scale
        return torch.clamp(stage_a + residual, 0.0, 1.0)


class StageBRefiner(nn.Module):
    """Train/eval module: frozen Stage A + trainable DetailHead."""

    def __init__(
        self,
        stage_a: BoundaryConsistencyWrapper,
        detail: DetailHead,
    ):
        super().__init__()
        self.stage_a = stage_a
        self.detail = detail
        for p in self.stage_a.parameters():
            p.requires_grad_(False)
        self.stage_a.eval()

    @property
    def in_ch(self) -> int:
        return self.stage_a.in_ch

    @property
    def out_ch(self) -> int:
        return self.stage_a.out_ch

    def stage_a_output(self, cond: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = self.stage_a(cond)
        if a.shape[-2:] != cond.shape[-2:]:
            a = F.interpolate(
                a, size=cond.shape[-2:], mode="bilinear", align_corners=False,
            )
        return a

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.detail(self.stage_a_output(cond), cond)


class StageBDeployWrapper(nn.Module):
    """Deploy I/O: ``forward(packed cond) -> packed denoised`` (Stage A + B)."""

    def __init__(self, refiner: StageBRefiner):
        super().__init__()
        self.stage_a = refiner.stage_a
        self.detail = refiner.detail
        self.in_ch = refiner.in_ch
        self.out_ch = refiner.out_ch

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return refiner_forward(self.stage_a, self.detail, cond)


def refiner_forward(
    stage_a: BoundaryConsistencyWrapper,
    detail: DetailHead,
    cond: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        a = stage_a(cond)
    if a.shape[-2:] != cond.shape[-2:]:
        a = F.interpolate(
            a, size=cond.shape[-2:], mode="bilinear", align_corners=False,
        )
    return detail(a, cond)


def load_stage_a(
    path: Path,
    device: torch.device,
) -> tuple[BoundaryConsistencyWrapper, dict]:
    """Load a ``cfm_student.pt`` / boundary deploy checkpoint."""
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    state = blob["state_dict"]
    gain_channel = bool(meta.get("gain_channel", False))
    temporal = int(meta.get("temporal", 4))
    cond_ch = int(
        meta.get("cond_ch", meta.get("in_ch", 4 * temporal + int(gain_channel))),
    )
    student = ConsistencyStudent(
        cond_ch=cond_ch,
        out_ch=int(meta.get("out_ch", 4)),
        base_channels=int(meta.get("base_channels", 64)),
        block_depth=int(meta.get("block_depth", 6)),
        gain_channel=gain_channel,
    )
    if any(k.startswith("student.") for k in state):
        inner = {k[len("student."):]: v for k, v in state.items()
                 if k.startswith("student.")}
        student.load_state_dict(inner, strict=False)
    elif any(k.startswith("stage_a.") for k in state):
        inner = {k[len("stage_a.student."):]: v for k, v in state.items()
                 if k.startswith("stage_a.student.")}
        student.load_state_dict(inner, strict=False)
    else:
        student.load_state_dict(state, strict=False)
    student.to(device)
    student.eval()
    wrap = BoundaryConsistencyWrapper(student).to(device).eval()
    return wrap, meta


def load_detail_head(
    path: Path,
    device: torch.device,
    *,
    defaults: dict | None = None,
) -> tuple[DetailHead, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = dict(blob.get("model", {}))
    if defaults:
        meta = {**defaults, **meta}
    detail = DetailHead(
        out_ch=int(meta.get("out_ch", 4)),
        base_channels=int(meta.get("base_channels", 32)),
        block_depth=int(meta.get("block_depth", 3)),
        use_noisy_hf=bool(meta.get("use_noisy_hf", True)),
        residual_scale=float(meta.get("residual_scale", 0.12)),
        hf_sigma=float(meta.get("hf_sigma", 1.2)),
    )
    state = blob["state_dict"]
    if any(k.startswith("detail.") for k in state):
        state = {k[len("detail."):]: v for k, v in state.items()
                 if k.startswith("detail.")}
    detail.load_state_dict(state, strict=False)
    detail.to(device)
    return detail, blob


def build_deploy_from_checkpoints(
    stage_a_path: Path,
    detail_path: Path | None,
    device: torch.device,
) -> tuple[nn.Module, dict, int]:
    """Return deploy module (Stage A alone or A+B), meta dict, in_ch."""
    stage_a, ameta = load_stage_a(stage_a_path, device)
    in_ch = int(ameta.get("cond_ch", ameta.get("in_ch", stage_a.in_ch)))
    if detail_path is None or not detail_path.is_file():
        return stage_a, ameta, in_ch
    detail, dblob = load_detail_head(detail_path, device, defaults=ameta)
    refiner = StageBRefiner(stage_a, detail)
    deploy = StageBDeployWrapper(refiner)
    meta = {
        **ameta,
        "stage_b": dblob.get("model", {}),
        "family": "cfm_stage_a_b",
        "stage_a": str(stage_a_path),
        "stage_b_ckpt": str(detail_path),
    }
    return deploy.eval(), meta, in_ch


def fake_quantize_detail_only(
    deploy: StageBDeployWrapper,
) -> StageBDeployWrapper:
    """INT8 fake-quant on DetailHead convs only (Stage A stays FP32)."""
    from .inference import fake_quantize_int8

    q = copy.deepcopy(deploy)
    fake_quantize_int8(q.detail, quant_activations=True)
    q.eval()
    return q
