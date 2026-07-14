"""Phase 2 — extract noise samples from calibration frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_raw_linear, to_luma


def extract_read_samples(bias_paths: list[Path], *, holdout: Path | None = None
                         ) -> tuple[np.ndarray, Path | None]:
    """Read noise: residual = bias_frame − mean(bias stack).

    Returns flattened residual samples and the held-out path (if any).
    """
    paths = [p for p in bias_paths if p != holdout]
    if not paths:
        raise ValueError("Need at least one bias frame (plus optional holdout)")
    stack = np.stack([load_raw_linear(p) for p in paths], axis=0)
    mean_bias = stack.mean(axis=0)
    residuals = []
    for p in paths:
        residuals.append(load_raw_linear(p) - mean_bias)
    samples = np.concatenate([r.ravel() for r in residuals]).astype(np.float32)
    return samples, holdout


def extract_row_samples(dark_paths: list[Path], *, holdout: Path | None = None
                        ) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """Row noise: per-row mean residuals in dark frames (lens capped).

    Returns (row_residual_samples, pixel_residual_samples, holdout_path).
    """
    paths = [p for p in dark_paths if p != holdout]
    if not paths:
        raise ValueError("Need at least one dark frame")
    stack = np.stack([load_raw_linear(p) for p in paths], axis=0)
    mean_dark = stack.mean(axis=0)
    pixel_res = []
    row_res = []
    for p in paths:
        res = load_raw_linear(p) - mean_dark
        pixel_res.append(res.ravel())
        row_means = res.mean(axis=1)
        row_res.append(row_means - row_means.mean())
    return (
        np.concatenate(row_res).astype(np.float32),
        np.concatenate(pixel_res).astype(np.float32),
        holdout,
    )


def extract_shot_points(flat_pairs: list[tuple[Path, Path]],
                        holdout_pair: tuple[Path, Path] | None = None,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Shot noise: (mean signal μ, variance σ²) at each flat-field brightness.

    From each pair: μ = mean((a+b)/2), σ² ≈ var(a−b)/2 (difference removes signal).
    """
    mus, vars_ = [], []
    for pair in flat_pairs:
        if holdout_pair and pair == holdout_pair:
            continue
        a, b = load_raw_linear(pair[0]), load_raw_linear(pair[1])
        mu = 0.5 * (a + b)
        diff = a - b
        mus.append(float(mu.mean()))
        vars_.append(float(np.var(diff) * 0.5))
    if not mus:
        raise ValueError("Need at least one flat-field pair (plus optional holdout)")
    return np.array(mus, dtype=np.float64), np.array(vars_, dtype=np.float64)
