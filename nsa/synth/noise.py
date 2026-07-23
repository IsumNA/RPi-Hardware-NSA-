"""IMX662 noise formation in the packed-Bayer float domain.

Formation model, per Bayer channel c ∈ {R, G1, G2, B}, in **normalized DN** on
the same [0, 1] scale as ``load_packed`` (black-subtracted / white-normalized):

    noisy_c = K_c · Poisson(clean_c / K_c) + read_c + row_c
              + dark_fpn_c + lf_chroma_c + BLE

where
* ``K_c``  is the per-channel system gain (DN per electron, on the normalized
  0–1 scale).  Poisson variance in normalized DN is ``K·mean``.  This is the
  correct ELD / SFRN / Sony-NMIH parameterization.
* ``read_c ~ Normal(0, σ_read_c)`` — signal-independent per-pixel read noise.
  Gaussian is the "Fast" pipeline default; a Tukey-lambda option is exposed
  because the IMX662 has measurably heavy tails (λ ≈ 0.04 at gain 256).
* ``row_c ~ Normal(0, σ_row_c)`` broadcast to a full row — banding noise.
* ``dark_fpn_c`` — **low-pass residual of a real dark frame** at matching
  gain/HCG (DSNU / fixed-pattern mottle).  HF of the dark is discarded so we
  do not double-count white read.  Shared across a temporal stack.
* ``lf_chroma_c`` — soft synthetic spatially-correlated low-frequency chroma
  blotches (upsample of coarse noise, R/B weighted).  Complements dark FPN
  when the dark library is small.

The clean input must be **linear** normalized packed Bayer of shape
``(H, W, 4)`` (as produced by ``nsa.raw_domain.pack_raw``).  Everything stays
float32 end-to-end; there is no 8-bit PNG round-trip anywhere.

Dark libraries (first hit wins per gain):
  ``datasets/hcg_bottomup/calib_hcg/gain{G}/dark/``
  ``datasets/imx662_project/calibration/imx662h_gain{G}/dark/``  (HCG)
  ``datasets/imx662_project/calibration/imx662_gain{G}/dark/``   (LCG)

Not included: PRNU (signal-multiplicative), exposure/temperature dark current.
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

ROOT = Path(__file__).resolve().parents[2]

# Soft-but-visible defaults: LF chroma ≈ 45% of mean read σ; dark LP ×1.5.
# v6 used 0.30 (wired via dataclass defaults); v6b bumps amplitude so flat-sky
# blotches remain above residual read after demosaic / viewing stretch.
DEFAULT_LF_CHROMA_FRAC = 0.45
DEFAULT_DARK_FPN_SCALE = 1.5
DEFAULT_LF_SCALE_FACTOR = 16  # coarse grid = H/16 × W/16


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
    use_dark_fpn : sample low-pass real-dark residual (FPN/DSNU mottle).
    dark_fpn_scale : scale on the low-pass dark residual.
    use_lf_chroma : add soft synthetic spatially-correlated LF chroma.
    lf_chroma_frac : LF chroma σ as a fraction of mean(read_sigma).
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
    use_dark_fpn: bool = True
    dark_fpn_scale: float = DEFAULT_DARK_FPN_SCALE
    use_lf_chroma: bool = True
    lf_chroma_frac: float = DEFAULT_LF_CHROMA_FRAC

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


def _box_lowpass(arr: np.ndarray, factor: int = 8) -> np.ndarray:
    """Cheap box low-pass via downsample → nearest upsample (keeps LF mottle)."""
    if arr.ndim == 2:
        arr = arr[..., None]
        squeeze = True
    else:
        squeeze = False
    h, w, c = arr.shape
    f = max(2, int(factor))
    hh, ww = (h // f) * f, (w // f) * f
    if hh < f or ww < f:
        return arr.astype(np.float32) if not squeeze else arr[..., 0].astype(np.float32)
    out = np.zeros((h, w, c), dtype=np.float32)
    for ci in range(c):
        p = arr[:hh, :ww, ci]
        d = p.reshape(hh // f, f, ww // f, f).mean(axis=(1, 3)).astype(np.float32)
        u = np.repeat(np.repeat(d, f, axis=0), f, axis=1)
        out[:u.shape[0], :u.shape[1], ci] = u
        # fill remainder by edge replicate
        if u.shape[0] < h:
            out[u.shape[0]:, :u.shape[1], ci] = u[-1:, :]
        if u.shape[1] < w:
            out[:, u.shape[1]:, ci] = out[:, u.shape[1] - 1:u.shape[1], ci]
    return out[..., 0] if squeeze else out


def _lf_chroma_noise(
    shape_hw: tuple[int, int],
    sigma: float,
    rng: np.random.Generator,
    *,
    scale_factor: int = DEFAULT_LF_SCALE_FACTOR,
) -> np.ndarray:
    """Soft spatially-correlated LF chroma blotches ``(H, W, 4)``.

    Coarse Gaussian field (H/sf × W/sf), R/B weighted, plus mild column drift.
    Amplitude set so per-channel σ ≈ ``sigma`` after upsample.
    """
    if sigma <= 0:
        h, w = shape_hw
        return np.zeros((h, w, 4), dtype=np.float32)
    h, w = shape_hw
    sf = max(4, int(scale_factor))
    lh, lw = max(2, h // sf), max(2, w // sf)
    # Shared luma blotch + per-channel chroma; R/B heavier (sky mottle look).
    luma = rng.normal(0.0, 1.0, size=(lh, lw, 1)).astype(np.float32)
    chroma = rng.normal(0.0, 1.0, size=(lh, lw, 4)).astype(np.float32)
    wts = np.asarray([1.35, 0.65, 0.65, 1.35], dtype=np.float32)
    low = 0.45 * luma + chroma * wts.reshape(1, 1, 4)
    # Nearest upsample to full packed size.
    rep_y, rep_x = int(np.ceil(h / lh)), int(np.ceil(w / lw))
    up = np.repeat(np.repeat(low, rep_y, axis=0), rep_x, axis=1)[:h, :w, :]
    # Mild column LF (vertical streaks / shading drift).
    col = rng.normal(0.0, 0.35, size=(1, w, 1)).astype(np.float32)
    # Smooth columns a bit horizontally.
    k = max(3, w // 32)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    col1d = np.convolve(col[0, :, 0], kernel, mode="same")
    up = up + col1d.reshape(1, w, 1) * wts.reshape(1, 1, 4)
    # Rescale to target σ (mean over channels).
    s = float(up.std())
    if s > 1e-9:
        up = up * (float(sigma) / s)
    return up.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Dark-frame residual bank (FPN / DSNU)                                      #
# --------------------------------------------------------------------------- #

class DarkResidualBank:
    """Mean-subtracted dark frames for one (sensor, gain); LP crops = FPN."""

    def __init__(self, residuals: np.ndarray, source: str):
        # residuals: (N, H, W, 4) float32, zero-mean per frame
        self.residuals = residuals.astype(np.float32, copy=False)
        self.source = source
        self.n = int(residuals.shape[0])
        self.H = int(residuals.shape[1])
        self.W = int(residuals.shape[2])

    def sample_lp_crop(
        self,
        shape_hw: tuple[int, int],
        rng: np.random.Generator,
        *,
        scale: float = 1.0,
        lp_factor: int = 8,
    ) -> np.ndarray:
        """Random crop → box low-pass → ``(H, W, 4)`` FPN mottle."""
        th, tw = int(shape_hw[0]), int(shape_hw[1])
        idx = int(rng.integers(0, self.n))
        frame = self.residuals[idx]
        H, W = frame.shape[:2]
        if H < th or W < tw:
            # reflect-pad then crop
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            H, W = frame.shape[:2]
        y = int(rng.integers(0, H - th + 1))
        x = int(rng.integers(0, W - tw + 1))
        crop = frame[y:y + th, x:x + tw].copy()
        if rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        if rng.random() < 0.5:
            crop = crop[::-1, :].copy()
        lp = _box_lowpass(crop, factor=lp_factor)
        return (lp * float(scale)).astype(np.float32)


_DARK_BANK_CACHE: dict[tuple[str, int], DarkResidualBank | None] = {}


def _dark_dirs_for(sensor: str, gain: int) -> list[Path]:
    """Candidate dark folders, preferred first (newer bottom-up HCG first)."""
    g = int(gain)
    dirs: list[Path] = []
    if sensor.endswith("h"):
        dirs.append(ROOT / "datasets" / "hcg_bottomup" / "calib_hcg" / f"gain{g}" / "dark")
        dirs.append(
            ROOT / "datasets" / "imx662_project" / "calibration"
            / f"imx662h_gain{g}" / "dark"
        )
    else:
        dirs.append(
            ROOT / "datasets" / "imx662_project" / "calibration"
            / f"imx662_gain{g}" / "dark"
        )
        # HCG bank can still supply FPN structure at matching gain if LCG missing.
        dirs.append(ROOT / "datasets" / "hcg_bottomup" / "calib_hcg" / f"gain{g}" / "dark")
    return dirs


def load_dark_bank(
    sensor: str,
    gain: int,
    *,
    max_frames: int = 48,
    force_reload: bool = False,
) -> DarkResidualBank | None:
    """Load / cache mean-subtracted packed dark residuals for ``sensor@gain``."""
    key = (str(sensor), int(gain))
    if not force_reload and key in _DARK_BANK_CACHE:
        return _DARK_BANK_CACHE[key]

    from nsa.raw_domain import load_packed  # local import — keeps noise torch-free

    files: list[Path] = []
    source = ""
    for d in _dark_dirs_for(sensor, gain):
        if not d.is_dir():
            continue
        found = sorted(d.glob("dark_*.dng"))
        if not found:
            found = sorted(d.glob("*.dng"))
        if found:
            files = found[:max_frames]
            source = str(d)
            break
    if len(files) < 2:
        _DARK_BANK_CACHE[key] = None
        return None

    stack = np.stack([load_packed(p) for p in files], axis=0).astype(np.float32)
    mean = stack.mean(axis=0, keepdims=True)
    residuals = stack - mean  # zero-mean: drops BLE, keeps FPN + stochastic
    bank = DarkResidualBank(residuals, source=source)
    _DARK_BANK_CACHE[key] = bank
    return bank


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
    dark_bank: DarkResidualBank | None = None,
    shared_dark_fpn: np.ndarray | None = None,
    shared_lf_chroma: np.ndarray | None = None,
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
        Dark FPN is **shared** across the stack (fixed pattern); LF chroma is
        mostly shared with a small per-frame jitter.
    clip : clamp final result to ``[0, 1]``.
    dark_bank : optional preloaded bank; else auto-loaded for model.sensor/gain.
    shared_dark_fpn / shared_lf_chroma : inject pre-sampled fields (temporal).
    """
    if rng is None:
        rng = np.random.default_rng()
    if clean_packed.ndim != 3 or clean_packed.shape[-1] != 4:
        raise ValueError(f"expected packed (H,W,4), got {clean_packed.shape}")
    if len(model.K) < 4 or len(model.read_sigma) < 4:
        raise ValueError("GainNoiseModel needs per-channel K and read_sigma")

    h, w = clean_packed.shape[:2]
    mean_read = float(np.mean(model.read_sigma)) if model.read_sigma else 0.0

    # Resolve dark bank once.
    bank = dark_bank
    if bank is None and bool(getattr(model, "use_dark_fpn", True)):
        bank = load_dark_bank(model.sensor, model.gain)

    # Shared fields for the temporal window (FPN must not flicker frame-to-frame).
    fpn_shared = shared_dark_fpn
    if fpn_shared is None and bank is not None and bool(getattr(model, "use_dark_fpn", True)):
        fpn_shared = bank.sample_lp_crop(
            (h, w), rng,
            scale=float(getattr(model, "dark_fpn_scale", DEFAULT_DARK_FPN_SCALE)),
            lp_factor=8,
        )

    lf_frac = float(getattr(model, "lf_chroma_frac", DEFAULT_LF_CHROMA_FRAC))
    lf_sigma = max(0.0, lf_frac * mean_read)
    lf_shared = shared_lf_chroma
    if lf_shared is None and bool(getattr(model, "use_lf_chroma", True)) and lf_sigma > 0:
        lf_shared = _lf_chroma_noise((h, w), lf_sigma, rng)

    def _one(clean: np.ndarray, *, lf_jitter: float = 0.25) -> np.ndarray:
        out = np.empty_like(clean, dtype=np.float32)
        # per-frame black-level error (single scalar drawn once per frame)
        ble = float(rng.normal(0.0, model.black_level_norm)) \
            if model.black_level_norm > 0 else 0.0
        # Per-frame LF jitter on top of shared blotch (slow-changing mottle).
        lf = None
        if lf_shared is not None:
            if lf_jitter > 0:
                jitter = _lf_chroma_noise(
                    (h, w), lf_sigma * lf_jitter, rng,
                )
                lf = lf_shared + jitter
            else:
                lf = lf_shared
        for c in range(4):
            plane = clean[..., c]
            shot = _poisson_scaled(plane, model.K[c], rng)
            read = _read_noise(plane.shape, float(model.read_sigma[c]),
                               model.read_shape, rng)
            row_sigma = float(model.row_sigma[c]) if model.row_sigma else 0.0
            row = _row_noise(plane.shape, row_sigma, rng)
            plane_out = shot + read + row + ble
            if fpn_shared is not None:
                plane_out = plane_out + fpn_shared[..., c]
            if lf is not None:
                plane_out = plane_out + lf[..., c]
            out[..., c] = plane_out
        if clip:
            np.clip(out, 0.0, 1.0, out=out)
        return out

    if n_frames <= 1:
        return _one(clean_packed, lf_jitter=0.0)
    return np.stack(
        [_one(clean_packed, lf_jitter=0.25) for _ in range(n_frames)],
        axis=0,
    )


def synthesize_temporal_stack(
    clean_packed: np.ndarray,
    model: GainNoiseModel,
    temporal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the exact tensor ``train_stream_to_gt`` uses as ``noisy``.

    Output shape ``(H, W, 4 * T)`` — T noise realizations of the same clean
    packed image, concatenated along the channel axis so channels ``[0:4]`` are
    the "current" frame and ``[4:8]``, ``[8:12]``, … are the (older) history.
    Dark FPN is shared across T (fixed pattern); LF chroma is mostly shared.
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
        use_dark_fpn=bool(getattr(base, "use_dark_fpn", True)),
        dark_fpn_scale=float(getattr(base, "dark_fpn_scale", DEFAULT_DARK_FPN_SCALE)),
        use_lf_chroma=bool(getattr(base, "use_lf_chroma", True)),
        lf_chroma_frac=float(getattr(base, "lf_chroma_frac", DEFAULT_LF_CHROMA_FRAC)),
    )
    return scaled
