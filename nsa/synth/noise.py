"""IMX662 noise formation in the packed-Bayer float domain.

Formation model, per Bayer channel c ∈ {R, G1, G2, B}, in **normalized DN** on
the same [0, 1] scale as ``load_packed`` (black-subtracted / white-normalized):

    noisy_c = K_c · Poisson(clean_c / K_c) + read_c + row_c

where
* ``K_c``  is the per-channel system gain (DN per electron, on the normalized
  0–1 scale).  Poisson variance in normalized DN is ``K·mean``.  This is the
  correct ELD / SFRN / Sony-NMIH parameterization.
* ``read_c ~ Normal(0, σ_read_c)`` — signal-independent per-pixel read noise.
  Gaussian is the "Fast" pipeline default; a Tukey-lambda option is exposed
  because the IMX662 has measurably heavy tails (λ ≈ 0.04 at gain 256).
* ``row_c ~ Normal(0, σ_row_c)`` broadcast to a full row — banding noise.

The clean input must be **linear** normalized packed Bayer of shape
``(H, W, 4)`` (as produced by ``nsa.raw_domain.pack_raw``).  Everything stays
float32 end-to-end; there is no 8-bit PNG round-trip anywhere.

Not included in Fast:
* PRNU / DSNU per-pixel maps (would need many bias frames per gain).
* Real dark-frame sampling (adds heavy tails + fixed pattern exactly).
* Dark-current dependence on exposure / temperature.

Fast is validated against real held-out burst frames by ``eval_noise_model``.
Upgrading to dark-frame sampling is a swap of ``_read_noise`` for a sampler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
#  Model dataclass                                                            #
# --------------------------------------------------------------------------- #

# Canonical packed channel order (matches ``nsa.raw_domain.pack_raw`` for RGGB):
#   0 = R  (top-left of the 2×2)
#   1 = G1 (top-right)
#   2 = G2 (bottom-left)
#   3 = B  (bottom-right)
CHANNELS: tuple[str, ...] = ("R", "G1", "G2", "B")


@dataclass
class GainNoiseModel:
    """Per-(sensor, gain, conversion-gain-mode) noise model.

    All noise magnitudes are in **normalized DN** on the [0, 1] scale that
    ``load_packed`` returns.  This keeps ``synthesize_noisy_packed`` composable
    with existing training tensors without any rescaling.

    Attributes
    ----------
    sensor : "imx662" or "imx662h" (HCG variant).
    gain : analog gain (ag128 / ag256 / ag512).
    K : per-channel system gain in normalized DN per electron.  Poisson
        variance = K · signal.  Order matches ``CHANNELS``.
    read_sigma : per-channel read noise σ (normalized DN).
    row_sigma : per-channel row banding σ (normalized DN).  0 disables.
    read_shape : Tukey-lambda shape parameter.  ``None`` or 0.14 = Gaussian.
        Set to e.g. 0.04 to inject the measured heavy tail.
    black_level_norm : residual black-level offset σ (frame-to-frame BLE).
        A single scalar per frame added to every pixel; small — a few LSB.
    n_frames : how many calibration/pair frames the fit used (metadata).
    fit_r2 : PTC fit quality per channel (metadata).
    """

    sensor: str
    gain: int
    K: list[float] = field(default_factory=list)
    read_sigma: list[float] = field(default_factory=list)
    row_sigma: list[float] = field(default_factory=list)
    read_shape: float | None = None
    black_level_norm: float = 0.0
    n_frames: int = 0
    fit_r2: list[float] = field(default_factory=list)
    fit_source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GainNoiseModel":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


def save_gain_model(model: GainNoiseModel, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return p


def load_gain_model(path: str | Path) -> GainNoiseModel:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return GainNoiseModel.from_dict(d)


# --------------------------------------------------------------------------- #
#  Noise sampling primitives                                                  #
# --------------------------------------------------------------------------- #

def _poisson_scaled(clean: np.ndarray, K: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample ``K · Poisson(clean / K)`` for a single channel.

    ``clean`` and the return are in normalized DN.  ``K`` is in DN/electron on
    the same normalized scale.  Follows the standard Sony-NMIH / ELD form: the
    mean is preserved (``E = clean``) and variance is ``K · clean``.
    """
    K = float(max(K, 1e-9))
    lam = np.maximum(clean, 0.0) / K
    # numpy's Poisson clips lam at ~9.22e18; our lams are much smaller.
    sampled = rng.poisson(lam).astype(np.float32)
    return sampled * K


def _tukey_lambda(shape: float, size: tuple[int, ...],
                  rng: np.random.Generator) -> np.ndarray:
    """Standard Tukey-lambda samples (mean 0, unit-ish scale).

    λ = 0.14  ≈ Gaussian.
    λ  < 0.14  → heavier tails (λ = 0 gives logistic, λ < 0 heavier still).
    """
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=size).astype(np.float32)
    if abs(shape) < 1e-9:
        # limit form: logistic
        return np.log(u / (1.0 - u)).astype(np.float32)
    return ((u ** shape - (1.0 - u) ** shape) / shape).astype(np.float32)


def _read_noise(shape_hw: tuple[int, int], sigma: float,
                shape_lambda: float | None,
                rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(shape_hw, dtype=np.float32)
    if shape_lambda is None or abs(shape_lambda - 0.14) < 1e-3:
        return rng.normal(0.0, sigma, size=shape_hw).astype(np.float32)
    raw = _tukey_lambda(shape_lambda, shape_hw, rng)
    # Rescale so the empirical σ ≈ requested σ (Tukey-λ variance depends on λ).
    s = float(raw.std())
    if s > 1e-9:
        raw = raw * (sigma / s)
    return raw.astype(np.float32)


def _row_noise(shape_hw: tuple[int, int], sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(shape_hw, dtype=np.float32)
    h, w = shape_hw
    per_row = rng.normal(0.0, sigma, size=(h, 1)).astype(np.float32)
    return np.broadcast_to(per_row, shape_hw).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #

def synthesize_noisy_packed(
    clean_packed: np.ndarray,
    model: GainNoiseModel,
    rng: np.random.Generator | None = None,
    *,
    n_frames: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """Inject IMX662 noise into a clean packed-Bayer image.

    Parameters
    ----------
    clean_packed : ``(H, W, 4)`` float32 in ``[0, 1]`` — same layout as
        ``nsa.raw_domain.pack_raw`` / ``load_packed``.
    model : ``GainNoiseModel`` for the target gain & conversion-gain mode.
    rng : numpy Generator (deterministic when supplied).
    n_frames : if > 1, return an ``(N, H, W, 4)`` stack of independent noise
        realizations of the same clean image — useful for the temporal input.
    clip : clamp final result to ``[0, 1]``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if clean_packed.ndim != 3 or clean_packed.shape[-1] != 4:
        raise ValueError(f"expected packed (H,W,4), got {clean_packed.shape}")
    if len(model.K) < 4 or len(model.read_sigma) < 4:
        raise ValueError("GainNoiseModel needs per-channel K and read_sigma")

    def _one(clean: np.ndarray) -> np.ndarray:
        out = np.empty_like(clean, dtype=np.float32)
        # per-frame black-level error (single scalar drawn once per frame)
        ble = float(rng.normal(0.0, model.black_level_norm)) \
            if model.black_level_norm > 0 else 0.0
        for c in range(4):
            plane = clean[..., c]
            shot = _poisson_scaled(plane, model.K[c], rng)
            read = _read_noise(plane.shape, float(model.read_sigma[c]),
                               model.read_shape, rng)
            row_sigma = float(model.row_sigma[c]) if model.row_sigma else 0.0
            row = _row_noise(plane.shape, row_sigma, rng)
            out[..., c] = shot + read + row + ble
        if clip:
            np.clip(out, 0.0, 1.0, out=out)
        return out

    if n_frames <= 1:
        return _one(clean_packed)
    return np.stack([_one(clean_packed) for _ in range(n_frames)], axis=0)


def synthesize_temporal_stack(
    clean_packed: np.ndarray,
    model: GainNoiseModel,
    temporal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the exact tensor ``train_stream_to_gt`` uses as ``noisy``.

    Output shape ``(H, W, 4 * T)`` — T independent noise realizations of the
    same clean packed image, concatenated along the channel axis so channels
    ``[0:4]`` are the "current" frame and ``[4:8]``, ``[8:12]``, … are the
    (older) history the network sees.  This matches
    ``train_stream_to_gt._stack_frames``.
    """
    T = max(1, int(temporal))
    stack = synthesize_noisy_packed(clean_packed, model, rng=rng, n_frames=T)
    # stack: (T, H, W, 4) → (H, W, 4*T)
    return np.concatenate([stack[i] for i in range(T)], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Utilities                                                                  #
# --------------------------------------------------------------------------- #

def scale_model_to_gain(base: GainNoiseModel, target_gain: int) -> GainNoiseModel:
    """Rescale a fitted model to a different analog gain, same CG mode.

    Both K and read/row σ scale linearly with analog gain within one
    conversion-gain regime (this is why ELD's calibration only measures at a
    handful of gains and interpolates).  Use this to seed missing gains when
    real data is absent; overwrite with a real fit as soon as one is available.
    """
    if base.gain <= 0 or target_gain <= 0:
        raise ValueError("gains must be positive")
    r = target_gain / base.gain
    scaled = GainNoiseModel(
        sensor=base.sensor,
        gain=int(target_gain),
        K=[float(k * r) for k in base.K],
        read_sigma=[float(s * r) for s in base.read_sigma],
        row_sigma=[float(s * r) for s in base.row_sigma] if base.row_sigma else [],
        read_shape=base.read_shape,
        black_level_norm=float(base.black_level_norm * r),
        n_frames=base.n_frames,
        fit_r2=list(base.fit_r2),
        fit_source=f"scaled_from_ag{base.gain}",
        notes=list(base.notes) + [f"linearly scaled from ag{base.gain}"],
    )
    return scaled
