"""IMX662 Bayer RAW I/O and physically-plausible sensor simulation.

The stack works on demosaiced linear RGB in [0, 1]. This module either:
  * loads a real Bayer RAW frame supplied by the user, or
  * synthesises a plausible IMX662 capture at an extreme analog gain so the
    demo always has a genuinely noisy frame to denoise.

The noise model is a standard photon-transfer model: Poisson shot noise on the
collected electrons plus Gaussian read noise, both scaled by the analog gain.
A clean reference ("ground truth") is produced by temporally averaging many
simulated frames - exactly how a long-exposure / multi-frame reference is built.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .sensors import SensorProfile, get_sensor

BLACK_LEVEL = 0.015  # normalised pedestal


def _synthetic_scene(h: int, w: int, seed: int) -> np.ndarray:
    """A structured low-light test scene with edges, gradients and fine detail.

    Returns clean linear RGB in [0, 1]. Structure (edges, text-like bars, a
    colour chart) makes denoising quality visually obvious in the demo.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = yy / h, xx / w

    # Soft vignetted background gradient.
    img = 0.18 + 0.22 * (1.0 - ((xn - 0.5) ** 2 + (yn - 0.5) ** 2))
    img = np.clip(img, 0, 1)
    img = np.stack([img * 0.9, img, img * 1.05], axis=-1)

    # A colour-checker style patch grid (tests chroma fidelity).
    chart = np.array(
        [
            [0.45, 0.32, 0.27], [0.76, 0.58, 0.50], [0.36, 0.47, 0.61],
            [0.34, 0.42, 0.26], [0.51, 0.50, 0.66], [0.40, 0.74, 0.67],
        ],
        dtype=np.float32,
    )
    ph, pw = h // 6, w // 9
    for i, colour in enumerate(chart):
        r0 = h // 12 + (i // 3) * ph
        c0 = w // 12 + (i % 3) * pw
        img[r0 : r0 + ph - 4, c0 : c0 + pw - 4] = colour

    # High-frequency resolution bars (tests detail preservation).
    bar_region = img[int(h * 0.62) : int(h * 0.92), int(w * 0.55) : int(w * 0.92)]
    bh, bw, _ = bar_region.shape
    freq = np.linspace(2, 30, bw)
    bars = 0.25 + 0.45 * (0.5 + 0.5 * np.sin(np.cumsum(freq) * 0.30))
    bar_region[:] = bars[None, :, None]

    # A couple of bright point sources (tests highlight handling).
    for _ in range(6):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        cv2.circle(img, (int(cx), int(cy)), rng.integers(2, 5), (0.95, 0.95, 0.9), -1)

    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _capture(clean: np.ndarray, gain: int, sensor: SensorProfile,
             rng: np.random.Generator, prnu: np.ndarray) -> np.ndarray:
    """Simulate one noisy read of a clean linear-RGB scene for a given sensor.

    Photon-transfer model: shot noise on the collected electrons (scaled by
    quantum efficiency and full-well capacity) + read noise, with a fixed-pattern
    PRNU gain map and low-frequency chroma cross-talk. Lower-grade sensors (low
    QE / high read noise / high chroma) therefore produce visibly messier frames.
    """
    eff = sensor.full_well * sensor.qe / float(gain)
    photons = np.clip(clean, 0, 1) * eff * prnu
    electrons = rng.poisson(np.maximum(photons, 0.0)).astype(np.float32)
    electrons += rng.normal(0.0, sensor.read_noise, size=clean.shape).astype(np.float32)
    noisy = electrons / eff

    if sensor.chroma_noise > 0:
        cn = rng.normal(0.0, sensor.chroma_noise, size=clean.shape).astype(np.float32)
        cn = cv2.GaussianBlur(cn, (0, 0), 2.2)        # low-frequency splotches
        cn -= cn.mean(axis=2, keepdims=True)          # opponent (chroma-only) noise
        noisy += cn

    noisy += BLACK_LEVEL
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _to_bayer(rgb: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Mosaic an RGB image down to a single-channel Bayer plane."""
    h, w, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer


def _demosaic(bayer: np.ndarray) -> np.ndarray:
    """Demosaic a normalised RGGB Bayer plane back to linear RGB."""
    b16 = np.clip(bayer * 65535.0, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(b16, cv2.COLOR_BAYER_RG2RGB)
    return (rgb.astype(np.float32) / 65535.0)


def load_real_raw(path: str, patch: int) -> np.ndarray:
    """Load a user-supplied RAW/image frame as normalised linear RGB."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 2:
            arr = _demosaic(arr / max(arr.max(), 1e-6))
        arr = arr / max(arr.max(), 1e-6)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read RAW frame: {path}")
        if img.ndim == 2:
            img = _demosaic(img.astype(np.float32) / max(float(img.max()), 1e-6))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = img / max(float(img.max()), 1e-6)
        arr = img
    return _center_crop(arr, patch)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


class Frame:
    """Bundle of the tensors the rest of the stack needs."""

    def __init__(self, noisy_rgb, clean_rgb, bayer, gain, source, sensor):
        self.noisy_rgb = noisy_rgb          # HxWx3 float32 [0,1]  (Panel A)
        self.clean_rgb = clean_rgb          # HxWx3 float32 [0,1]  (Panel B / GT)
        self.bayer = bayer                  # HxW   float32 [0,1]  (the RAW plane)
        self.gain = gain
        self.source = source                # "synthetic" | path
        self.sensor = sensor                # SensorProfile
        self.height, self.width = noisy_rgb.shape[:2]


def build_frame(
    input_raw: str | None,
    gain: int,
    temporal_frames: int,
    patch: int,
    sensor: SensorProfile | str,
    seed: int,
) -> Frame:
    """Produce the (noisy, clean, bayer) frame bundle for the chosen sensor."""
    if isinstance(sensor, str):
        sensor = get_sensor(sensor)
    rng = np.random.default_rng(seed)

    if input_raw:
        clean = load_real_raw(input_raw, patch)
        source = input_raw
    else:
        clean = _center_crop(_synthetic_scene(patch, patch, seed), patch)
        source = "synthetic"

    # Fixed-pattern PRNU gain map: same for every exposure of this sensor, so it
    # does NOT average out in the temporal ground truth (as on real silicon).
    prnu = (1.0 + np.random.default_rng(seed + 777).normal(
        0.0, sensor.prnu, size=clean.shape)).astype(np.float32)

    noisy = _capture(clean, gain, sensor, rng, prnu)

    # Ground truth = temporal average of many independent reads (denoised ref).
    acc = np.zeros_like(clean)
    for _ in range(max(1, temporal_frames)):
        acc += _capture(clean, gain, sensor, rng, prnu)
    gt = acc / max(1, temporal_frames)

    bayer = _to_bayer(noisy, sensor.bayer)
    return Frame(noisy, gt, bayer, gain, source, sensor)
