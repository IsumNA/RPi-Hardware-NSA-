"""Visual calibration report — turns the fitted noise model into proof.

A JSON of {shot_a, read_sigma, ...} is unfalsifiable by eye. This renders the
plots that let you actually judge the fit:

  1. Photon-transfer curve  — measured noise-variance-vs-signal points with the
     fitted line var = a·mu + read_var overlaid (the canonical sensor plot).
  2. Read-noise histogram   — real bias-frame pixels vs the fitted distribution.
  3. Real vs synthetic noise — the money shot: histogram of a REAL frame's noise
     overlaid on noise the calibrated model SYNTHESISES. Overlap = good model.
  4. Amplified noise crops  — real vs synthetic, contrast-stretched so the grain
     is visible; they should look like the same texture.

Written next to the model JSON as ``<name>.report.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_linear, to_luma
from .model import NoiseModel
from .synthesize import synthesize_noisy

_INK, _GRID, _REAL, _SIM, _FIT = "#1A1A1A", "#DDDDDD", "#C41E3A", "#2E7D6B", "#E8912D"


def _amp(residual: np.ndarray) -> np.ndarray:
    """Contrast-stretch a zero-mean noise field to [0,1] (±3σ -> full range)."""
    s = float(residual.std()) or 1e-6
    return np.clip(residual / (6.0 * s) + 0.5, 0.0, 1.0)


def _highpass(img: np.ndarray, k: int = 31) -> np.ndarray:
    """Isolate noise by subtracting a large box-mean (removes a flat panel's
    illumination gradient / lens shading, so it's not counted as 'noise').
    O(N) via an integral image — no scipy/cv2 dependency."""
    a = img.astype(np.float64)
    pad = k // 2
    ap = np.pad(a, pad, mode="reflect")
    ii = np.cumsum(np.cumsum(ap, 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    h, w = a.shape
    s = (ii[k:k + h, k:k + w] - ii[:h, k:k + w]
         - ii[k:k + h, :w] + ii[:h, :w])
    return (a - s / (k * k)).astype(np.float32)


def render_calibration_report(
    model: NoiseModel,
    out_png: Path | str,
    *,
    shot_mu: np.ndarray,
    shot_var: np.ndarray,
    read_samples: np.ndarray,
    validation: dict | None = None,
    real_flat: Path | str | None = None,
    real_pair: "tuple | None" = None,
    seed: int = 662,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    out_png = Path(out_png)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Noise calibration — {model.sensor}  gain {model.gain}   "
        f"(shot a={model.shot_a:.4g}, read σ={model.read_dist.sigma:.4g}, "
        f"row={model.row_strength:.3f})",
        fontsize=14, color=_INK, fontweight="bold")

    # 1) Photon-transfer curve ------------------------------------------------
    a = ax[0, 0]
    read_var = float(model.read_dist.sigma) ** 2
    if len(shot_mu):
        a.scatter(shot_mu, shot_var, s=36, c=_REAL, label="measured", zorder=3)
        xs = np.linspace(0, float(np.max(shot_mu)) * 1.05 + 1e-6, 100)
        a.plot(xs, model.shot_a * xs + read_var, c=_FIT, lw=2,
               label=f"fit: {model.shot_a:.3g}·μ + {read_var:.3g}")
    a.set_title("Photon-transfer curve", color=_INK)
    a.set_xlabel("signal μ"); a.set_ylabel("noise variance")
    a.legend(fontsize=8); a.grid(True, color=_GRID)

    # 2) Read-noise histogram vs fit -----------------------------------------
    a = ax[0, 1]
    if len(read_samples):
        a.hist(read_samples, bins=80, density=True, color=_REAL, alpha=0.55,
               label="real bias pixels")
        mu, sig = float(model.read_dist.mu), max(float(model.read_dist.sigma), 1e-9)
        xs = np.linspace(read_samples.min(), read_samples.max(), 200)
        pdf = np.exp(-((xs - mu) ** 2) / (2 * sig * sig)) / (sig * np.sqrt(2 * np.pi))
        a.plot(xs, pdf, c=_FIT, lw=2, label=f"fitted N(μ,σ={sig:.3g})")
    a.set_title("Read noise: real vs fit", color=_INK)
    a.set_xlabel("residual"); a.legend(fontsize=8); a.grid(True, color=_GRID)

    # 3) Real vs synthesised noise distribution (the key check) --------------
    a = ax[0, 2]
    real_resid = sim_resid = None
    if real_pair is not None:
        # Cleanest check: gt is the clean reference, so noise = noisy - gt
        # directly, and synthetic = synth(gt) - gt. No high-pass needed.
        noisy_rgb, clean_rgb = real_pair
        real = to_luma(np.asarray(noisy_rgb, np.float32))
        clean = to_luma(np.asarray(clean_rgb, np.float32))
        h = min(real.shape[0], clean.shape[0]); w = min(real.shape[1], clean.shape[1])
        real, clean = real[:h, :w], clean[:h, :w]
        real_resid = real - clean
        sim = synthesize_noisy(np.asarray(clean_rgb, np.float32)[:h, :w], model, rng)
        sim_resid = to_luma(sim) - clean
        lo = float(min(real_resid.min(), sim_resid.min()))
        hi = float(max(real_resid.max(), sim_resid.max()))
        a.hist(real_resid.ravel(), bins=80, range=(lo, hi), density=True,
               color=_REAL, alpha=0.5, label=f"REAL noisy-gt (σ={real_resid.std():.4g})")
        a.hist(sim_resid.ravel(), bins=80, range=(lo, hi), density=True,
               color=_SIM, alpha=0.5, label=f"SYNTHETIC (σ={sim_resid.std():.4g})")
    elif real_flat is not None and Path(real_flat).is_file():
        real = to_luma(load_linear(Path(real_flat)))
        mean = float(real.mean())
        # Synthesise noise on a flat patch at the SAME mean level, then compare
        # NOISE to noise: high-pass BOTH with the same filter so the panel's
        # illumination gradient / lens shading isn't mistaken for sensor noise.
        const = np.full(real.shape, mean, dtype=np.float32)
        sim = synthesize_noisy(const, model, rng)   # 2D in -> 2D out
        real_resid = _highpass(real)
        sim_resid = _highpass(sim)
        lo = min(real_resid.min(), sim_resid.min())
        hi = max(real_resid.max(), sim_resid.max())
        a.hist(real_resid.ravel(), bins=80, range=(lo, hi), density=True,
               color=_REAL, alpha=0.5, label=f"REAL (σ={real_resid.std():.4g})")
        a.hist(sim_resid.ravel(), bins=80, range=(lo, hi), density=True,
               color=_SIM, alpha=0.5, label=f"SYNTHETIC (σ={sim_resid.std():.4g})")
    else:
        a.text(0.5, 0.5, "no real flat frame\nfor comparison",
               ha="center", va="center", color="#999")
    a.set_title("Real vs synthetic noise", color=_INK)
    a.set_xlabel("noise value"); a.legend(fontsize=8); a.grid(True, color=_GRID)

    # 4 & 5) Amplified noise crops (real | synthetic) -------------------------
    crop = 220
    for col, (resid, name, c) in enumerate((
            (real_resid, "REAL noise (amplified)", _REAL),
            (sim_resid, "SYNTHETIC noise (amplified)", _SIM))):
        a = ax[1, col]
        if resid is not None:
            h, w = resid.shape
            cy, cx = h // 2, w // 2
            s = min(crop, cy, cx)
            a.imshow(_amp(resid[cy - s:cy + s, cx - s:cx + s]), cmap="gray",
                     vmin=0, vmax=1)
        else:
            a.text(0.5, 0.5, "n/a", ha="center", va="center", color="#999")
        a.set_title(name, color=c, fontsize=10); a.axis("off")

    # 6) Validation verdicts --------------------------------------------------
    a = ax[1, 2]; a.axis("off")
    lines = ["VALIDATION (held-out frames)", ""]
    if validation and validation.get("checks"):
        for ch in validation["checks"]:
            mark = "PASS" if ch.get("pass") else "FAIL"
            lines.append(f"[{mark}]  {ch['name']}")
            lines.append(f"        {ch['metric']} = {ch['value']:.4g}")
        lines.append("")
        lines.append(f"OVERALL: {'PASS' if validation.get('ok') else 'REVIEW'}")
    else:
        lines.append("(no held-out validation)")
    a.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
           fontsize=10, color=_INK, transform=a.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor="white")
    plt.close(fig)
    return out_png
