"""Real, configurable denoising architectures (Level 3 of the stack).

Three families share a common residual-denoising contract: given a noisy RGB
frame they predict a clean RGB frame. Every architectural flag from the config
(``base_channels``, ``block_depth``, ``conv_type``, ``activation``) maps onto a
concrete change in the graph, so the exported ONNX genuinely differs per config.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU(), "silu": nn.SiLU(inplace=True)}[name]


def _conv(in_c: int, out_c: int, conv_type: str, k: int = 3) -> nn.Module:
    """Standard conv, or a depthwise-separable conv when requested."""
    pad = k // 2
    if conv_type == "depthwise" and in_c == out_c:
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, k, padding=pad, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=True),
        )
    if conv_type == "depthwise":
        # Channel count changes -> depthwise on input then pointwise projection.
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, k, padding=pad, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=True),
        )
    return nn.Conv2d(in_c, out_c, k, padding=pad)


class _ConvBlock(nn.Module):
    def __init__(self, c: int, conv_type: str, act: str):
        super().__init__()
        self.conv = _conv(c, c, conv_type)
        self.norm = nn.BatchNorm2d(c)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNDenoiser(nn.Module):
    """DnCNN-style residual denoiser (predicts the noise, subtracts it)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.head = nn.Sequential(_conv(3, c, cfg.conv_type), _act(cfg.activation))
        self.body = nn.Sequential(
            *[_ConvBlock(c, cfg.conv_type, cfg.activation) for _ in range(cfg.block_depth)]
        )
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        noise = self.tail(self.body(self.head(x)))
        return torch.clamp(x - noise, 0.0, 1.0)


class _NAFBlock(nn.Module):
    """Simplified NAFNet block: depthwise conv + SimpleGate + channel attention."""

    def __init__(self, c: int, conv_type: str):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1)
        )
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.dw(self.conv1(x))
        a, b = y.chunk(2, dim=1)         # SimpleGate
        y = a * b
        y = y * self.sca(y)              # simplified channel attention
        y = self.conv2(y)
        return x + y * self.beta


class NAFNetDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(*[_NAFBlock(c, cfg.conv_type) for _ in range(cfg.block_depth)])
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        feat = self.body(self.head(x))
        return torch.clamp(x + self.tail(feat), 0.0, 1.0)


class UNetDenoiser(nn.Module):
    """Compact 2-scale U-Net encoder/decoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        d = max(1, cfg.block_depth // 2)
        self.enc1 = nn.Sequential(_conv(3, c, cfg.conv_type), _act(cfg.activation),
                                  *[_ConvBlock(c, cfg.conv_type, cfg.activation) for _ in range(d)])
        self.down = nn.Conv2d(c, c * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[_ConvBlock(c * 2, cfg.conv_type, cfg.activation) for _ in range(d)])
        self.up = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(*[_ConvBlock(c, cfg.conv_type, cfg.activation) for _ in range(d)])
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        u = self.up(e2) + e1
        out = self.tail(self.dec1(u))
        return torch.clamp(x + out, 0.0, 1.0)


def build_model(cfg: ModelConfig) -> nn.Module:
    families = {"cnn": CNNDenoiser, "unet": UNetDenoiser, "nafnet": NAFNetDenoiser}
    return families[cfg.model_family](cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
