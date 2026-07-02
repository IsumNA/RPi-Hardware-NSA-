"""Real, configurable denoising architectures (Level 3 of the stack).

Three families share a common residual-denoising contract: given a noisy RGB
frame they predict a clean RGB frame. Every architectural flag from the config
(``base_channels``, ``block_depth``, ``conv_type``, ``activation``) maps onto a
concrete change in the graph, so the exported ONNX genuinely differs per config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _norm(c: int) -> nn.Module:
    """Batch-size-agnostic normalization (GroupNorm).

    Calibration trains on tiny minibatches of one frame's crops, where
    BatchNorm's running statistics are unstable and hurt quality. GroupNorm is
    independent of batch size and quantizes cleanly (no running buffers to fold).
    """
    groups = 8 if c % 8 == 0 else (4 if c % 4 == 0 else 1)
    return nn.GroupNorm(groups, c)


class _ConvBlock(nn.Module):
    def __init__(self, c: int, conv_type: str, act: str):
        super().__init__()
        self.conv = _conv(c, c, conv_type)
        self.norm = _norm(c)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _BNConvBlock(nn.Module):
    """conv → BatchNorm → act (the classic DnCNN-with-BatchNorm unit)."""

    def __init__(self, c: int, conv_type: str, act: str):
        super().__init__()
        self.conv = _conv(c, c, conv_type)
        self.norm = nn.BatchNorm2d(c)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNDenoiser(nn.Module):
    """Classic BatchNorm residual denoiser (predicts the noise, subtracts it).

    Uses BatchNorm — its defining trait vs the GroupNorm ``DnCNNDenoiser``. BN
    depends on batch statistics (noisy for the small on-frame calibration batch)
    and its scale/shift folds into the INT8 graph, so this classic baseline
    trades quality for simplicity and is usually dominated by the GroupNorm /
    modern families — the tool's built-in "why architecture choice matters" foil.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.head = nn.Sequential(_conv(3, c, cfg.conv_type), _act(cfg.activation))
        self.body = nn.Sequential(
            *[_BNConvBlock(c, cfg.conv_type, cfg.activation) for _ in range(cfg.block_depth)]
        )
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        noise = self.tail(self.body(self.head(x)))
        return torch.clamp(x - noise, 0.0, 1.0)


class _NAFBlock(nn.Module):
    """Full NAFNet block: LN → conv/dwconv → SimpleGate → SCA → conv (+shortcut),
    then LN → conv → SimpleGate → conv (+shortcut).

    The earlier version only had the first half and no LayerNorm; adding the
    normalization and the gated feed-forward half is what unlocks NAFNet's
    quality-per-parameter (it is otherwise the leanest, fastest zoo member).
    """

    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        dw = c * expand
        self.norm1 = _LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.dw = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv2 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = _LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, dw, 1)
        self.conv4 = nn.Conv2d(dw // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.dw(self.conv1(self.norm1(x)))
        a, b = y.chunk(2, dim=1)         # SimpleGate
        y = a * b
        y = y * self.sca(y)              # simplified channel attention
        x = x + self.conv2(y) * self.beta
        z = self.conv3(self.norm2(x))
        a, b = z.chunk(2, dim=1)         # SimpleGate (feed-forward)
        z = a * b
        return x + self.conv4(z) * self.gamma


class NAFNetDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(cfg.block_depth)])
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
            self.encoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        self.middle = nn.Sequential(*[_NAFBlock(ch) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))

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
    """Classic DnCNN — conv-norm-act stack that predicts the noise residual.

    Uses GroupNorm (the original paper's BatchNorm, but batch-size-agnostic and
    quantization-friendly): normalization between conv layers is precisely what
    makes DnCNN trainable — without it the plain stack barely denoises.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        layers = [_conv(3, c, cfg.conv_type), _act(cfg.activation)]
        for _ in range(cfg.block_depth):
            layers += [_conv(c, c, cfg.conv_type), _norm(c), _act(cfg.activation)]
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


class FFDNetDenoiser(nn.Module):
    """FFDNet-style denoiser — works at half resolution for speed.

    Pixel-unshuffle folds the image into 12 channels at half resolution, a plain
    conv body cleans it there (4× fewer spatial positions = fast), then
    pixel-shuffle folds it back. Space-to-depth/depth-to-space map cleanly to
    INT8 NPUs, so this is the lean, accelerator-friendly option in the zoo.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        self.down = nn.PixelUnshuffle(2)
        self.head = nn.Sequential(_conv(12, c, cfg.conv_type), _act(cfg.activation))
        self.body = nn.Sequential(
            *[_ConvBlock(c, cfg.conv_type, cfg.activation) for _ in range(cfg.block_depth)])
        self.tail = nn.Conv2d(c, 12, 3, padding=1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        _, _, h, w = x.shape
        ph, pw = h % 2, w % 2
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        d = self.down(xp)
        feat = self.body(self.head(d))
        out = xp + self.up(self.tail(feat))
        out = out[..., :h, :w]
        return torch.clamp(out, 0.0, 1.0)


class _ResBlock(nn.Module):
    def __init__(self, c: int, conv_type: str, act: str):
        super().__init__()
        self.c1 = _conv(c, c, conv_type)
        self.act = _act(act)
        self.c2 = _conv(c, c, conv_type)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class DRUNetDenoiser(nn.Module):
    """DRUNet — a deep, BatchNorm-free 3-scale residual U-Net.

    Deeper and more capacity than the compact ``UNetDenoiser``: residual blocks
    at three resolutions with strided-conv downsamples and transpose-conv
    upsamples, no BatchNorm (quantization-friendly). The flagship quality option
    — heavier on memory, so the suitability matrix flags tiling on small NPUs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        d = max(1, cfg.block_depth // 2)
        ct, ac = cfg.conv_type, cfg.activation
        self.head = _conv(3, c, ct)
        self.enc1 = nn.Sequential(*[_ResBlock(c, ct, ac) for _ in range(d)])
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[_ResBlock(c * 2, ct, ac) for _ in range(d)])
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.mid = nn.Sequential(*[_ResBlock(c * 4, ct, ac) for _ in range(d)])
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(*[_ResBlock(c * 2, ct, ac) for _ in range(d)])
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(*[_ResBlock(c, ct, ac) for _ in range(d)])
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.shape
        ph = (4 - h % 4) % 4
        pw = (4 - w % 4) % 4
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(self.head(xp))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        out = xp + self.tail(d1)
        out = out[..., :h, :w]
        return torch.clamp(out, 0.0, 1.0)


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (Restormer/BCHW style)."""

    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-6)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _MDTA(nn.Module):
    """Multi-Dconv-head Transposed Attention (attention across channels)."""

    def __init__(self, c: int, heads: int):
        super().__init__()
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        hd = c // self.heads
        q = q.reshape(b, self.heads, hd, h * w)
        k = k.reshape(b, self.heads, hd, h * w)
        v = v.reshape(b, self.heads, hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp     # (b, heads, hd, hd)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class _GDFN(nn.Module):
    """Gated-Dconv feed-forward network (Restormer FFN)."""

    def __init__(self, c: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(c * expansion)
        self.project_in = nn.Conv2d(c, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, c, 1)

    def forward(self, x):
        x = self.dw(self.project_in(x))
        a, b = x.chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class _RestormerBlock(nn.Module):
    def __init__(self, c: int, heads: int):
        super().__init__()
        self.norm1 = _LayerNorm2d(c)
        self.attn = _MDTA(c, heads)
        self.norm2 = _LayerNorm2d(c)
        self.ffn = _GDFN(c)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerDenoiser(nn.Module):
    """Restormer-style transformer denoiser (efficient channel attention).

    Uses transposed (channel) self-attention — cost scales with channels², not
    pixels² — plus a gated Dconv FFN and LayerNorm. Highest-quality, but the
    LayerNorm/softmax graph is awkward for INT8 NPUs, so it shines on the Pi 5
    CPU and gets caveats on the accelerators (exactly what the matrix shows).
    Architecture inspired by the published Restormer / HuggingFace variants.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.base_channels
        heads = max(1, c // 16)                  # 16->1, 32->2, 64->4 (divides c)
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(
            *[_RestormerBlock(c, heads) for _ in range(cfg.block_depth)])
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
    from .model_opts import normalize_model_config
    normalize_model_config(cfg)
    if cfg.model_family == "nafnet" and list(getattr(cfg, "nafnet_enc_blocks", []) or []):
        return NAFNetUNetDenoiser(cfg)
    families = {
        "cnn": CNNDenoiser,
        "dncnn": DnCNNDenoiser,
        "unet": UNetDenoiser,
        "rednet": REDNetDenoiser,
        "ridnet": RIDNetDenoiser,
        "nafnet": NAFNetDenoiser,
        "ffdnet": FFDNetDenoiser,
        "drunet": DRUNetDenoiser,
        "restormer": RestormerDenoiser,
    }
    return families[cfg.model_family](cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
