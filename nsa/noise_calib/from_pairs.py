"""Calibrate a per-gain noise model directly from real noisy/gt PAIRS.

The classic pipeline needs bias/dark/flat calibration frames, and we only have
those at one gain. But we already have real ``noisy.png`` / ``gt.png`` pairs at
EVERY analogue gain — and ``noisy - gt`` IS a noise realisation at that gain.
So fit the Poisson-Gaussian model straight from those residuals:

  * bin pixels by clean intensity -> per-bin variance gives the photon-transfer
    (μ, σ²) points -> shot slope a  (var = a·μ);
  * the darkest bin's residual = read+quant floor -> read distribution;
  * per-row residual means -> row/pattern noise.

Fit one model per gain and you can synthesise arbitrarily large training sets at
the RIGHT noise level for each gain (feed into simulate_dataset.py). Analogue
gain amplifies noise before the ADC, so shot_a and read-σ grow with gain — a
gain-256 profile badly under-noises a gain-512 synthesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nsa.sensors import get_sensor

from .fit import build_noise_model
from .model import NoiseModel


def _luma(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _base_sensor(sensor: str) -> str:
    """SENSORS profile key for a variant tag (imx662h -> imx662 physical sensor)."""
    s = (sensor or "imx662").lower()
    return "imx662" if s.startswith("imx662") else s


def calibrate_from_pairs(pairs, sensor: str, gain: int, *, n_bins: int = 24):
    """Fit a NoiseModel from a list of (noisy_rgb, clean_rgb) float [0,1] pairs.

    Returns ``(model, aux)`` where aux carries the fit inputs for the report:
    ``shot_mu``, ``shot_var``, ``read_samples``.
    """
    if not pairs:
        raise ValueError("calibrate_from_pairs needs at least one (noisy, clean) pair")

    resid_all, clean_all, row_res_all = [], [], []
    for noisy, clean in pairs:
        ln, lc = _luma(np.asarray(noisy, np.float32)), _luma(np.asarray(clean, np.float32))
        h = min(ln.shape[0], lc.shape[0]); w = min(ln.shape[1], lc.shape[1])
        ln, lc = ln[:h, :w], lc[:h, :w]
        r = ln - lc
        resid_all.append(r.ravel())
        clean_all.append(lc.ravel())
        row_res_all.append(r.mean(axis=1))          # per-row mean residual
    resid = np.concatenate(resid_all)
    clean = np.concatenate(clean_all)
    row_samples = np.concatenate(row_res_all)

    # Photon-transfer points: variance of the residual within each intensity bin.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(clean, edges) - 1, 0, n_bins - 1)
    mus, vars = [], []
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) < 512:
            continue
        mus.append(float(clean[m].mean()))
        vars.append(float(resid[m].var()))
    shot_mu = np.array(mus, dtype=np.float64)
    shot_var = np.array(vars, dtype=np.float64)

    # Read+quant floor: residual where the clean signal is darkest.
    dark_cut = float(np.quantile(clean, 0.05))
    read_samples = resid[clean <= max(dark_cut, edges[1])]
    if read_samples.size < 2000:                    # fallback: whole residual
        read_samples = resid
    read_samples = read_samples.astype(np.float64)

    prof = get_sensor(_base_sensor(sensor))
    model = build_noise_model(
        sensor=sensor, gain=gain, adc_bits=prof.bit_depth,
        read_samples=read_samples,
        row_samples=row_samples.astype(np.float64),
        pixel_dark_samples=resid.astype(np.float64),
        shot_mu=shot_mu, shot_var=shot_var,
        n_bias=0, n_dark=0, n_flat_levels=len(shot_mu),
    )
    model.notes.append(f"Fitted from {len(pairs)} real noisy/gt pair(s) at gain {gain}")
    aux = {"shot_mu": shot_mu, "shot_var": shot_var, "read_samples": read_samples}
    return model, aux


def run_pair_calibration(dataset: str, out_json: Path | str, *,
                         sensor: str = "imx662h", gain: int = 512,
                         filter_tokens=None, max_pairs: int = 40,
                         seed: int = 662) -> tuple[NoiseModel, dict]:
    """Fit + save a gain-specific noise model from the dataset's real pairs,
    matching folders whose analogue-gain tag == ``gain`` (e.g. imx662h_ag512).
    Writes the model JSON and a visual report next to it."""
    from nsa.raw_io import load_training_pairs, analog_gain_from_name
    from .model import save_model
    from .report import render_calibration_report

    toks = list(filter_tokens or [sensor])
    named = load_training_pairs(dataset, toks, sensor=_base_sensor(sensor), gain=gain,
                                with_names=True, tile=0, max_side=0)
    pairs = [(n, c) for name, n, c in named
             if analog_gain_from_name(name) == gain]
    if not pairs:
        raise ValueError(f"no {sensor} pairs at gain {gain} under {dataset} "
                         f"(filter {toks})")
    model, aux = calibrate_from_pairs(pairs, sensor, gain)
    out_json = Path(out_json)
    save_model(model, out_json)
    validation = {"n_pairs": len(pairs), "shot_a": model.shot_a,
                  "read_sigma": model.read_dist.sigma}
    try:
        png = render_calibration_report(
            model, out_json.with_suffix(".report.png"),
            shot_mu=aux["shot_mu"], shot_var=aux["shot_var"],
            read_samples=aux["read_samples"], validation=None,
            real_pair=pairs[0], seed=seed)
        if png is not None:
            validation["report_png"] = str(png)
    except Exception as exc:  # noqa: BLE001
        validation["report_error"] = str(exc)
    return model, validation
