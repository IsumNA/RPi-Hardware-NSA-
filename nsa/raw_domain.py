"""RAW-domain denoising — 'Learning to See in the Dark' style.

We were denoising processed 8-bit RGB, where noise is gamma-stretched, clipped,
and correlated by demosaicing — the hardest possible domain. This module works
on the raw Bayer BEFORE demosaic/gamma, where noise is simple (Poisson-Gaussian,
spatially independent) and all the sensor bits survive.

Pipeline (SID convention):
  * pack_raw:  Bayer -> (H/2, W/2, 4) black-subtracted, white-normalised — the
    2x2 CFA becomes 4 channels, so there's no mosaic pattern masquerading as
    noise and the network runs at quarter cost.
  * denoise the 4-channel packed raw (reuses the normal train/infer loop — the
    tensor helpers are channel-agnostic).
  * packed_to_rgb / unpack: back to a viewable image.

The denoiser (RawDenoiser) reuses NAFNet blocks. Phase 2B feeds 5 channels
(4 packed Bayer + normalised fusion confidence from ``temporal_fusion``) and
predicts a 4-channel packed residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models import _NAFBlock


# --- raw I/O ---------------------------------------------------------------
def load_raw_norm(path: Path) -> tuple[np.ndarray, tuple]:
    """Raw Bayer plane, black-subtracted and white-normalised to ~[0,1]."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        raw = r.raw_image_visible.astype(np.float32)
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
    return (raw - black) / max(white - black, 1.0), (black, white)


def pack_raw(bayer: np.ndarray) -> np.ndarray:
    """Bayer (H,W) -> packed (H/2, W/2, 4) in the sensor's 2x2 order."""
    h, w = bayer.shape[0] // 2 * 2, bayer.shape[1] // 2 * 2
    b = bayer[:h, :w]
    return np.stack([b[0::2, 0::2], b[0::2, 1::2],
                     b[1::2, 0::2], b[1::2, 1::2]], axis=-1).astype(np.float32)


def packed_to_rgb(packed: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Half-res linear RGB for viewing/metrics: R, avg(G1,G2), B (RGGB order).

    ``gain`` brightens the dark linear signal for display only.
    """
    r = packed[..., 0]
    g = 0.5 * (packed[..., 1] + packed[..., 2])
    b = packed[..., 3]
    rgb = np.stack([r, g, b], axis=-1) * gain
    return np.clip(rgb, 0.0, 1.0)


def load_packed(path: Path) -> np.ndarray:
    bayer, _ = load_raw_norm(path)
    return pack_raw(bayer)


def burst_clean(paths, limit: int = 128) -> np.ndarray:
    """Temporal-average GT in the raw domain (packed), from a burst."""
    acc = None
    n = 0
    for p in list(paths)[:limit]:
        pk = load_packed(p)
        acc = pk if acc is None else acc + pk
        n += 1
    return acc / max(n, 1)


def fusion_confidence(
    weight_map: np.ndarray,
    *,
    k_cap: float = 16.0,
) -> np.ndarray:
    """Normalise temporal-fusion sample count to [0, 1] confidence."""
    w = np.asarray(weight_map, dtype=np.float32)
    if w.ndim == 2:
        w = w[..., np.newaxis]
    return np.clip(w / max(float(k_cap), 1e-6), 0.0, 1.0).astype(np.float32)


def stack_fusion_input(
    fused: np.ndarray,
    weight_map: np.ndarray,
    *,
    k_cap: float = 16.0,
) -> np.ndarray:
    """Build 5-channel input: 4 packed Bayer + fusion confidence (H/2, W/2, 5)."""
    fused = np.asarray(fused, dtype=np.float32)[..., :4]
    conf = fusion_confidence(weight_map, k_cap=k_cap)
    return np.concatenate([fused, conf], axis=-1)


def to_fusion_tensor(
    fused: np.ndarray,
    weight_map: np.ndarray,
    *,
    k_cap: float = 16.0,
) -> torch.Tensor:
    """NCHW float tensor for RawDenoiser 5-channel forward pass."""
    x = stack_fusion_input(fused, weight_map, k_cap=k_cap)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


# --- model -----------------------------------------------------------------
class RawDenoiser(nn.Module):
    """NAFNet-blocks denoiser on packed raw (residual).

    Default 4->4 for single-frame packed raw. Phase 2B uses ``in_ch=5`` with
  ``out_ch=4``: fused packed Bayer plus a confidence channel from
    ``fuse_burst_packed``'s weight map (count / k_cap).
    """

    def __init__(
        self,
        base_channels: int = 32,
        block_depth: int = 4,
        in_ch: int = 4,
        out_ch: int | None = None,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = 4 if in_ch == 5 else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        c = base_channels
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(block_depth)])
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)

    def forward(self, x):
        residual = self.tail(self.body(self.head(x)))
        base = x[:, : self.out_ch]
        return torch.clamp(base + residual, 0.0, 1.0)
