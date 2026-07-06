"""Phase 4 — validate calibrated model on held-out frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract import extract_read_samples, extract_row_samples, extract_shot_points
from .io import load_linear, to_luma
from .model import NoiseModel
from .synthesize import synthesize_noisy


def _hist_distance(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.mean(np.abs(ha - hb)))


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
        real_samples, _ = extract_read_samples([bias_holdout], holdout=None)
        # Simulate read+quant on zero signal
        zeros = np.zeros((64, 64), dtype=np.float32)
        sim_stack = np.concatenate([
            synthesize_noisy(zeros, model, rng, include_shot=False).ravel()
            for _ in range(8)
        ])
        hist_d = _hist_distance(real_samples, sim_stack)
        report["checks"].append({
            "name": "read_histogram",
            "metric": "mean_abs_hist_diff",
            "value": hist_d,
            "pass": hist_d < 0.15,
        })

    if dark_holdout is not None:
        img = to_luma(load_linear(dark_holdout))
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
