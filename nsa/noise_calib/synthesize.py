"""Phase 5 — forward noise synthesis from a calibrated model."""

from __future__ import annotations

import numpy as np

from .model import DistributionFit, NoiseModel


def _sample_dist(dist: DistributionFit, shape: tuple, rng: np.random.Generator) -> np.ndarray:
    if dist.kind == "none" or dist.sigma <= 0 and dist.scale <= 0:
        return np.zeros(shape, dtype=np.float32)
    if dist.kind == "gamma" and dist.shape > 0 and dist.scale > 0:
        # centred gamma noise
        samp = rng.gamma(dist.shape, dist.scale, size=shape).astype(np.float32)
        return samp - float(dist.shape * dist.scale)
    sigma = dist.sigma if dist.sigma > 0 else 1e-6
    return rng.normal(dist.mu, sigma, size=shape).astype(np.float32)


def synthesize_noisy(
    clean: np.ndarray,
    model: NoiseModel,
    rng: np.random.Generator,
    *,
    include_shot: bool = True,
    include_read: bool = True,
    include_row: bool = True,
    include_quant: bool = True,
) -> np.ndarray:
    """Add calibrated noise components to a clean linear image in [0, 1].

    1. Poisson shot noise  (variance ≈ a · signal)
    2. Read noise        (per-pixel, fitted distribution)
    3. Row noise         (one draw per row, optional)
    4. Quantisation      (±½ LSB)
    """
    clean = np.clip(clean.astype(np.float32), 0.0, 1.0)
    noisy = clean.copy()

    if include_shot:
        curve = getattr(model, "var_curve", None)
        if curve:
            # Signal-dependent variance from the fitted quadratic TOTAL-variance
            # curve, minus the read floor (added separately) to avoid double
            # counting. Handles the clipped-RGB bend a linear a·μ can't.
            c0, c1, c2 = curve
            total = c0 + c1 * clean + c2 * clean * clean
            read_var = float(model.read_dist.sigma) ** 2 if include_read else 0.0
            shot_std = np.sqrt(np.maximum(total - read_var, 0.0))
            noisy += rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * shot_std
        elif model.shot_a > 0:
            # Gaussian approx to Poisson: σ = sqrt(a·μ) (raw-domain linear fallback)
            shot_std = np.sqrt(np.maximum(model.shot_a * clean, 0.0))
            noisy += rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * shot_std

    if include_read:
        noisy += _sample_dist(model.read_dist, clean.shape, rng)

    if include_row and model.row_dist is not None and model.row_strength > 0:
        h = clean.shape[0]
        if clean.ndim == 2:
            row = _sample_dist(model.row_dist, (h, 1), rng) * model.row_strength
            noisy += np.broadcast_to(row, clean.shape)
        else:
            row = _sample_dist(model.row_dist, (h, 1, 1), rng) * model.row_strength
            noisy += np.broadcast_to(row, (h, 1, clean.shape[2]))

    if include_quant and model.quant_scale > 0:
        noisy += model.quant_scale * rng.uniform(-1.0, 1.0, size=clean.shape).astype(np.float32)

    return np.clip(noisy, 0.0, 1.0)


def synthesize_pair(
    clean: np.ndarray,
    model: NoiseModel,
    seed: int,
    temporal_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, gt) for training — gt is clean or temporal average."""
    rng = np.random.default_rng(seed)
    if temporal_frames <= 1:
        return synthesize_noisy(clean, model, rng), clean.copy()
    acc = np.zeros_like(clean)
    for i in range(temporal_frames):
        acc += synthesize_noisy(clean, model, np.random.default_rng(seed + i))
    gt = acc / temporal_frames
    noisy = synthesize_noisy(clean, model, np.random.default_rng(seed + 9999))
    return noisy, gt
