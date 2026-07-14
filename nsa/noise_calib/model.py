"""Calibrated noise model — parameter set from Phases 2–3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DistributionFit:
    """Best-fit noise distribution (Gaussian or Gamma)."""

    kind: str                           # gaussian | gamma | none
    mu: float = 0.0
    sigma: float = 0.0                  # gaussian scale
    shape: float = 0.0                  # gamma k
    scale: float = 0.0                  # gamma θ
    ks_stat: float | None = None        # Kolmogorov–Smirnov vs samples (lower=better)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DistributionFit:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class NoiseModel:
    """Phase 3 result: {a, read_dist, row_dist, adc_bits} + metadata."""

    sensor: str = "imx662"
    gain: int = 256
    temperature_c: float | None = None
    adc_bits: int = 12
    shot_a: float = 0.0                 # Poisson scale (variance = a · signal); linear fallback
    # Quadratic TOTAL-variance curve var(μ) = c0 + c1·μ + c2·μ² fitted to the
    # measured (signal, residual-variance) points. In the processed/clipped RGB
    # domain the photon-transfer curve is NOT linear — variance is squeezed to
    # ~0 at both black and white — so a line badly misfits. When present this
    # supersedes shot_a for the signal-dependent noise magnitude.
    var_curve: list | None = None       # [c0, c1, c2]
    read_dist: DistributionFit = field(default_factory=lambda: DistributionFit("gaussian"))
    row_dist: DistributionFit | None = None
    row_strength: float = 0.0           # relative weight of per-row component
    quant_scale: float = 0.0            # ±½ LSB in normalised units
    n_bias: int = 0
    n_dark: int = 0
    n_flat_levels: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["read_dist"] = self.read_dist.to_dict()
        if self.row_dist is not None:
            d["row_dist"] = self.row_dist.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoiseModel:
        rd = DistributionFit.from_dict(d.get("read_dist", {}))
        row_raw = d.get("row_dist")
        row = DistributionFit.from_dict(row_raw) if row_raw else None
        return cls(
            sensor=str(d.get("sensor", "imx662")),
            gain=int(d.get("gain", 256)),
            temperature_c=d.get("temperature_c"),
            adc_bits=int(d.get("adc_bits", 12)),
            shot_a=float(d.get("shot_a", 0.0)),
            var_curve=list(d["var_curve"]) if d.get("var_curve") else None,
            read_dist=rd,
            row_dist=row,
            row_strength=float(d.get("row_strength", 0.0)),
            quant_scale=float(d.get("quant_scale", 0.0)),
            n_bias=int(d.get("n_bias", 0)),
            n_dark=int(d.get("n_dark", 0)),
            n_flat_levels=int(d.get("n_flat_levels", 0)),
            notes=list(d.get("notes", [])),
        )


def save_model(model: NoiseModel, path: Path | str) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write the noise model to {path} (permission denied). The "
            f"target is likely a read-only shared dataset store. Choose an output "
            f"path under a writable location (e.g. this repo's models/noise/)."
        ) from exc


def load_model(path: Path | str) -> NoiseModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return NoiseModel.from_dict(data)
