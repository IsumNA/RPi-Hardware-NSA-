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


class NAFNetUNetDenoiser(nn.Module):
    """Multi-scale (U-shaped) NAFNet with a custom encoder/decoder topology.

    Mirrors the official NAFNet layout (intro conv → encoder stages with 2×
    downsamples → middle blocks → decoder stages with PixelShuffle upsamples +
    skip connections → ending conv). The per-stage NAFBlock counts come from the
    config (``nafnet_enc_blocks`` / ``nafnet_middle_blocks`` / ``nafnet_dec_blocks``),
    so the manager can dial in topologies like ``encoders 1 2 2 · middle 4 ·
    decoders 2 1 1`` exactly like denoise-hw's configurable NAFNet.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        enc = list(cfg.nafnet_enc_blocks) or [1, 1, 1]
        dec = list(cfg.nafnet_dec_blocks) or enc[::-1]
        mid = max(1, int(cfg.nafnet_middle_blocks))
        if len(dec) != len(enc):
            dec = enc[::-1]
        self.levels = len(enc)

        self.intro = nn.Conv2d(3, c, 3, padding=1)
        self.ending = nn.Conv2d(c, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = c
        for n in enc:
            self.encoders.append(nn.Sequential(*[_NAFBlock(ch, cfg.conv_type) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        self.middle = nn.Sequential(*[_NAFBlock(ch, cfg.conv_type) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(ch, cfg.conv_type) for _ in range(n)]))

    def forward(self, x):
        # Pad so the input is divisible by 2**levels, then crop back.
        _, _, h, w = x.shape
        mod = 2 ** self.levels
        ph = (mod - h % mod) % mod
        pw = (mod - w % mod) % mod
        xp = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

        feat = self.intro(xp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)
        feat = self.middle(feat)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat) + skip
            feat = dec(feat)
        out = xp + self.ending(feat)
        out = out[..., :h, :w]
        return torch.clamp(out, 0.0, 1.0)


class DnCNNDenoiser(nn.Module):
    """Classic DnCNN — a BatchNorm-free residual conv stack.

    Distinct from ``CNNDenoiser`` (which uses BatchNorm): dropping BN makes the
    graph friendlier to INT8 post-training quantization (no scale/shift folding),
    so it tends to keep more PSNR after the INT8 step. Predicts the noise residual.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        layers = [_conv(3, c, cfg.conv_type), _act(cfg.activation)]
        for _ in range(cfg.block_depth):
            layers += [_conv(c, c, cfg.conv_type), _act(cfg.activation)]
        self.body = nn.Sequential(*layers)
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        noise = self.tail(self.body(x))
        return torch.clamp(x - noise, 0.0, 1.0)


class REDNetDenoiser(nn.Module):
    """RED-Net — residual encoder-decoder with symmetric skip connections.

    A stack of convolutions (encoder) mirrored by transpose-convolutions
    (decoder) at the *same* spatial resolution, with skip connections linking
    matching encoder/decoder layers every two steps. The skips carry image
    detail across the network, which helps recover texture lost to heavy noise.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        n = max(2, cfg.block_depth)
        self.head = nn.Sequential(_conv(3, c, cfg.conv_type), _act(cfg.activation))
        self.encoders = nn.ModuleList(
            [nn.Sequential(_conv(c, c, cfg.conv_type), _act(cfg.activation))
             for _ in range(n)])
        self.decoders = nn.ModuleList(
            [nn.Sequential(nn.ConvTranspose2d(c, c, 3, padding=1), _act(cfg.activation))
             for _ in range(n)])
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        skips = []
        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i % 2 == 0:
                skips.append(feat)
        for i, dec in enumerate(self.decoders):
            feat = dec(feat)
            j = (len(self.decoders) - 1 - i)
            if j % 2 == 0 and skips:
                feat = feat + skips.pop()
        return torch.clamp(x + self.tail(feat), 0.0, 1.0)


class _ChannelAttention(nn.Module):
    """Squeeze-and-excitation style feature attention (RIDNet's EAM gate)."""

    def __init__(self, c: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(4, c // 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(4, c // 4), c, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class _EAM(nn.Module):
    """Enhancement-attention module: residual conv pair + channel attention."""

    def __init__(self, c: int, conv_type: str, act: str):
        super().__init__()
        self.conv1 = _conv(c, c, conv_type)
        self.conv2 = _conv(c, c, conv_type)
        self.act = _act(act)
        self.ca = _ChannelAttention(c)

    def forward(self, x):
        y = self.act(self.conv2(self.act(self.conv1(x))))
        return x + self.ca(y)


class RIDNetDenoiser(nn.Module):
    """RIDNet-style residual-in-residual denoiser with feature attention.

    A head projection feeds a stack of enhancement-attention modules (each a
    residual conv pair gated by channel attention), then a tail projection with
    a global residual connection. The attention lets the network suppress noisy
    channels and emphasise structure — strong quality per parameter.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(
            *[_EAM(c, cfg.conv_type, cfg.activation) for _ in range(cfg.block_depth)])
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
    if cfg.model_family == "nafnet" and list(getattr(cfg, "nafnet_enc_blocks", []) or []):
        return NAFNetUNetDenoiser(cfg)
    families = {
        "cnn": CNNDenoiser,
        "dncnn": DnCNNDenoiser,
        "unet": UNetDenoiser,
        "rednet": REDNetDenoiser,
        "ridnet": RIDNetDenoiser,
        "nafnet": NAFNetDenoiser,
    }
    return families[cfg.model_family](cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
