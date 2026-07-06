"""Run Phases 1–4 end-to-end on a calibration capture folder."""

from __future__ import annotations

from pathlib import Path

from nsa.sensors import get_sensor

from .extract import extract_read_samples, extract_row_samples, extract_shot_points
from .fit import build_noise_model
from .io import discover_phase1_root
from .model import NoiseModel, save_model
from .validate import validate_model


def run_calibration_pipeline(
    calibration_root: Path | str,
    output_json: Path | str,
    *,
    sensor: str = "imx662",
    gain: int = 256,
    temperature_c: float | None = None,
    holdout: bool = True,
    seed: int = 662,
) -> tuple[NoiseModel, dict]:
    """Phases 1–4: discover → extract → fit → validate → save JSON."""
    root = Path(calibration_root).expanduser().resolve()
    data = discover_phase1_root(root)

    bias_paths: list[Path] = data["bias"]  # type: ignore[assignment]
    dark_paths: list[Path] = data["dark"]  # type: ignore[assignment]
    flat_pairs: list[tuple[Path, Path]] = data["flat_pairs"]  # type: ignore[assignment]

    if len(bias_paths) < 2:
        raise ValueError("Phase 1: need ≥2 bias frames in bias/")
    if len(dark_paths) < 1:
        raise ValueError("Phase 1: need ≥1 dark frame in dark/")
    if len(flat_pairs) < 2:
        raise ValueError("Phase 1: need ≥2 flat-field brightness levels in flat/")

    bias_hold = bias_paths[-1] if holdout and len(bias_paths) >= 3 else None
    dark_hold = dark_paths[-1] if holdout and len(dark_paths) >= 2 else None
    flat_hold = flat_pairs[-1] if holdout and len(flat_pairs) >= 3 else None

    read_samples, _ = extract_read_samples(bias_paths, holdout=bias_hold)
    row_samples, pixel_dark, _ = extract_row_samples(dark_paths, holdout=dark_hold)
    shot_mu, shot_var = extract_shot_points(flat_pairs, holdout_pair=flat_hold)

    prof = get_sensor(sensor)
    model = build_noise_model(
        sensor=sensor,
        gain=gain,
        adc_bits=prof.bit_depth,
        read_samples=read_samples,
        row_samples=row_samples,
        pixel_dark_samples=pixel_dark,
        shot_mu=shot_mu,
        shot_var=shot_var,
        n_bias=len(bias_paths),
        n_dark=len(dark_paths),
        n_flat_levels=len(flat_pairs),
        temperature_c=temperature_c,
    )

    validation = validate_model(
        model,
        bias_holdout=bias_hold,
        dark_holdout=dark_hold,
        flat_holdout=flat_hold,
        seed=seed,
    )

    save_model(model, output_json)
    return model, validation
