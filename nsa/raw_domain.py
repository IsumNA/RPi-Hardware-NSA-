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

The denoiser (RawDenoiser) reuses NAFNet blocks with a 4->4 channel head/tail.
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


# --- model -----------------------------------------------------------------
class RawDenoiser(nn.Module):
    """NAFNet-blocks denoiser on 4-channel packed raw (residual)."""

    def __init__(self, base_channels: int = 32, block_depth: int = 4, in_ch: int = 4):
        super().__init__()
        c = base_channels
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(block_depth)])
        self.tail = nn.Conv2d(c, in_ch, 3, padding=1)

    def forward(self, x):
        return torch.clamp(x + self.tail(self.body(self.head(x))), 0.0, 1.0)
