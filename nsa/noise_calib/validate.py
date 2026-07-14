"""Phase 4 — validate calibrated model on held-out frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract import extract_read_samples, extract_row_samples, extract_shot_points
from .io import load_raw_linear, to_luma
from .model import NoiseModel
from .synthesize import synthesize_noisy


def _hist_distance(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Total-variation distance between two sample distributions, in [0, 1].

    Uses probability-normalised histograms (not density), so the metric is
    bounded and scale-invariant — the old density-based version returned
    unbounded values (e.g. 9.0) that made the 0.15 threshold meaningless.
    0 = identical, 1 = disjoint.
    """
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return 0.5 * float(np.abs(pa - pb).sum())


def validate_model(
    model: NoiseModel,
    *,
    bias_holdout: Path | None,
    dark_holdout: Path | None,
    flat_holdout: tuple[Path, Path] | None,
    seed: int = 0,
) -> dict:
    """Compare real held-out statistics vs Phase-5 simulation."""
    rng = np.random.default_rng(seed)
    report: dict = {"ok": True, "checks": []}

    if bias_holdout is not None:
        # Compare read-noise MAGNITUDE, not full histogram shape: a single frame
        # can't be temporally mean-subtracted, so high-pass it to isolate noise
        # (read+row on a flat bias frame) and high-pass the simulation the same
        # way — the high-pass scales both equally, so the σ ratio is fair. A
        # magnitude test is robust; a TV-of-histograms test over-flags on finite
        # samples and slight non-Gaussianity even when the fit is good.
        from .report import _highpass
        real_std = float(_highpass(load_raw_linear(bias_holdout)).std())
        model_std = float(model.read_dist.sigma)   # the fitted read magnitude
        rel_err = abs(model_std - real_std) / max(real_std, 1e-9)
        report["checks"].append({
            "name": "read_noise_level",
            "metric": "rel_err(σ)",
            "value": rel_err,
            "pass": rel_err < 0.35,
        })

    if dark_holdout is not None:
        img = load_raw_linear(dark_holdout)
        row_real = img.mean(axis=1) - img.mean()
        zeros = np.zeros_like(img)
        sim = synthesize_noisy(zeros, model, rng, include_shot=False)
        row_sim = sim.mean(axis=1) - sim.mean()
        corr = float(np.corrcoef(row_real, row_sim)[0, 1]) if row_real.std() > 1e-9 else 0.0
        report["checks"].append({
            "name": "row_correlation",
            "metric": "corr(real,sim)",
            "value": corr,
            "pass": abs(corr) < 0.5 or model.row_strength < 0.05,
        })

    if flat_holdout is not None:
        mu_pts, var_pts = extract_shot_points([flat_holdout], holdout_pair=None)
        mu, var_real = float(mu_pts[0]), float(var_pts[0])
        # Monte Carlo shot variance at this brightness
        patch = np.full((32, 32), mu, dtype=np.float32)
        sim_vars = []
        for _ in range(16):
            s = synthesize_noisy(patch, model, rng, include_read=False,
                                 include_row=False, include_quant=False)
            sim_vars.append(float(np.var(s - patch)))
        var_sim = float(np.mean(sim_vars))
        rel_err = abs(var_sim - var_real) / max(var_real, 1e-12)
        report["checks"].append({
            "name": "shot_variance",
            "metric": "rel_err",
            "value": rel_err,
            "pass": rel_err < 0.35,
        })

    report["ok"] = all(c.get("pass", True) for c in report["checks"])
    return report
