"""Phase 3 — fit statistical parameters from extracted samples."""

from __future__ import annotations

import numpy as np

from .model import DistributionFit, NoiseModel


def _fit_gaussian(samples: np.ndarray) -> DistributionFit:
    mu = float(np.mean(samples))
    sigma = float(np.std(samples))
    return DistributionFit(kind="gaussian", mu=mu, sigma=max(sigma, 1e-12))


def _fit_gamma_moments(samples: np.ndarray) -> DistributionFit:
    """Method-of-moments Gamma fit on |residuals| (noise magnitude)."""
    x = np.abs(samples).astype(np.float64)
    x = x[x > 1e-12]
    if len(x) < 32:
        return _fit_gaussian(samples)
    m = float(np.mean(x))
    v = float(np.var(x))
    if v <= m * m:
        return _fit_gaussian(samples)
    shape = max((m * m) / v, 1e-6)
    scale = max(v / m, 1e-12)
    return DistributionFit(kind="gamma", mu=0.0, sigma=0.0,
                           shape=float(shape), scale=float(scale))


def _ks_gaussian(samples: np.ndarray, mu: float, sigma: float) -> float:
    """Simple KS statistic vs Gaussian CDF (no scipy)."""
    if sigma < 1e-12:
        return 1.0
    try:
        from scipy import stats  # type: ignore
        return float(stats.kstest(samples, "norm", args=(mu, sigma)).statistic)
    except Exception:
        # Normalised mean absolute z-score proxy
        z = (samples - mu) / sigma
        return float(np.mean(np.abs(z) > 3.0))


def _ks_gamma(samples: np.ndarray, shape: float, scale: float) -> float:
    x = np.abs(samples).astype(np.float64)
    x = x[x > 1e-12]
    if len(x) < 32:
        return 1.0
    try:
        from scipy import stats  # type: ignore
        return float(stats.kstest(x, "gamma", args=(shape, 0.0, scale)).statistic)
    except Exception:
        return 0.5


def fit_read_distribution(samples: np.ndarray) -> DistributionFit:
    """Test Gaussian vs Gamma on read-noise residuals; pick lower KS."""
    g = _fit_gaussian(samples)
    gm = _fit_gamma_moments(samples)
    g.ks_stat = _ks_gaussian(samples, g.mu, g.sigma)
    gm.ks_stat = _ks_gamma(samples, gm.shape, gm.scale)
    if gm.ks_stat is not None and g.ks_stat is not None and gm.ks_stat < g.ks_stat:
        return gm
    return g


def fit_row_distribution(row_samples: np.ndarray, pixel_samples: np.ndarray,
                         *, row_sigma_threshold: float = 1e-5) -> tuple[DistributionFit | None, float]:
    """Fit row residual distribution if structural row pattern exists."""
    row_std = float(np.std(row_samples))
    pix_std = float(np.std(pixel_samples))
    if row_std < row_sigma_threshold or row_std < 0.05 * pix_std:
        return None, 0.0
    dist = fit_read_distribution(row_samples)
    strength = row_std / max(pix_std, 1e-12)
    return dist, float(strength)


def fit_shot_poisson(mu: np.ndarray, var: np.ndarray) -> float:
    """Linear regression var = a·μ (+ intercept forced through origin)."""
    mu = np.maximum(mu, 1e-9)
    # a = Σ(μ·σ²) / Σ(μ²)
    a = float(np.sum(mu * var) / max(np.sum(mu * mu), 1e-12))
    return max(a, 0.0)


def fit_variance_curve(mu: np.ndarray, var: np.ndarray) -> list[float] | None:
    """Quadratic least-squares fit of TOTAL variance vs signal: c0 + c1·μ + c2·μ².

    Raw photon-transfer is linear, but in the processed/clipped RGB domain the
    curve bends over (clipping suppresses variance near black and white), so a
    quadratic captures it far better than a line. Returns ``[c0, c1, c2]`` or
    ``None`` if there are too few points to fit.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if mu.size < 3:
        return None
    coeffs = np.polyfit(mu, var, 2)          # highest power first
    return [float(coeffs[2]), float(coeffs[1]), float(coeffs[0])]  # c0, c1, c2


def quant_scale_from_adc(adc_bits: int) -> float:
    """Quantisation noise: ±½ LSB in normalised [0,1] units."""
    levels = max(2 ** int(adc_bits), 2)
    return 0.5 / float(levels - 1)


def build_noise_model(
    *,
    sensor: str,
    gain: int,
    adc_bits: int,
    read_samples: np.ndarray,
    row_samples: np.ndarray,
    pixel_dark_samples: np.ndarray,
    shot_mu: np.ndarray,
    shot_var: np.ndarray,
    n_bias: int,
    n_dark: int,
    n_flat_levels: int,
    temperature_c: float | None = None,
) -> NoiseModel:
    read_dist = fit_read_distribution(read_samples)
    row_dist, row_strength = fit_row_distribution(row_samples, pixel_dark_samples)
    shot_a = fit_shot_poisson(shot_mu, shot_var)
    var_curve = fit_variance_curve(shot_mu, shot_var)
    notes = []
    if read_dist.kind == "gamma":
        notes.append("Read noise: Gamma fit beat Gaussian on bias residuals")
    if row_dist is not None:
        notes.append(f"Row noise detected (strength {row_strength:.3f})")
    if var_curve is not None and var_curve[2] < -1e-6:
        notes.append("Photon-transfer curve bends over (clipping) — quadratic fit used")
    return NoiseModel(
        sensor=sensor,
        gain=gain,
        temperature_c=temperature_c,
        adc_bits=adc_bits,
        shot_a=shot_a,
        var_curve=var_curve,
        read_dist=read_dist,
        row_dist=row_dist,
        row_strength=row_strength,
        quant_scale=quant_scale_from_adc(adc_bits),
        n_bias=n_bias,
        n_dark=n_dark,
        n_flat_levels=n_flat_levels,
        notes=notes,
    )
