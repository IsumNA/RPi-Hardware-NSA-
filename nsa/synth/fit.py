"""Per-gain, per-channel IMX662 noise-model fitting.

Two fit paths, both writing a :class:`GainNoiseModel`:

* :func:`fit_from_calibration`  — the classical PTC fit from lens-capped
  bias + grey-card flat pairs (bias/dark/flat).  Requires a
  ``calibration/<sensor>_gain<g>/`` folder.  Available at gain 256 today.

* :func:`fit_from_bursts`       — fit K and read σ from real static bursts.
  The burst mean is a near-clean GT; residuals of each frame vs. that mean
  give us (μ, σ²) samples over the whole intensity range at the burst's gain.
  Available at every gain a scene was captured at.

Both fits use per-Bayer-channel linear regression **with intercept**::

    σ²(μ) = K · μ + σ_read²

which is the standard photon-transfer form and — critically — separates shot
from read.  The current repo's ``fit_shot_poisson`` forces the intercept to 0
and lets the read-noise floor bleed into ``K`` (3× overestimate on the LCG
gain-256 fit, cf. the fitted JSON).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from ..raw_domain import pack_raw
from .noise import CHANNELS, GainNoiseModel


# --------------------------------------------------------------------------- #
#  Raw loaders (kept here so this module has no dependency on noise_calib)    #
# --------------------------------------------------------------------------- #

def _load_bayer_norm(path: Path) -> np.ndarray:
    """Raw Bayer plane, black-subtracted and white-normalized to ~[0, 1]."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        raw = r.raw_image_visible.astype(np.float32)
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
    return (raw - black) / max(white - black, 1.0)


def _load_packed_norm(path: Path) -> np.ndarray:
    """Half-res packed (H/2, W/2, 4) in ``[0, 1]`` — same as raw_domain.load_packed."""
    return pack_raw(_load_bayer_norm(path))


# --------------------------------------------------------------------------- #
#  Per-channel PTC fit                                                        #
# --------------------------------------------------------------------------- #

def _fit_line_with_intercept(mu: np.ndarray, var: np.ndarray
                             ) -> tuple[float, float, float]:
    """Least-squares fit ``var = K · mu + b``; returns (K, b, R²)."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if mu.size < 2:
        return 0.0, float(var.mean() if var.size else 0.0), 0.0
    A = np.stack([mu, np.ones_like(mu)], axis=1)
    coef, *_ = np.linalg.lstsq(A, var, rcond=None)
    K, b = float(coef[0]), float(coef[1])
    pred = K * mu + b
    ss_res = float(np.sum((var - pred) ** 2))
    ss_tot = float(np.sum((var - var.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return max(K, 0.0), max(b, 0.0), r2


def _per_channel_bins(clean_packed: np.ndarray, resid_packed: np.ndarray,
                      *, n_bins: int = 32, min_per_bin: int = 512,
                      ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bin residual variance vs mean intensity per Bayer channel.

    ``clean_packed`` and ``resid_packed`` are ``(H, W, 4)`` aligned pairs.
    Returns 4 lists (one per channel) of matched (μ, σ²) arrays.
    """
    mus_all: list[np.ndarray] = []
    vars_all: list[np.ndarray] = []
    for c in range(4):
        clean = clean_packed[..., c].ravel()
        resid = resid_packed[..., c].ravel()
        lo, hi = float(np.percentile(clean, 0.5)), float(np.percentile(clean, 99.5))
        if hi <= lo:
            mus_all.append(np.zeros(0))
            vars_all.append(np.zeros(0))
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_idx = np.clip(np.digitize(clean, edges) - 1, 0, n_bins - 1)
        mu_b: list[float] = []
        var_b: list[float] = []
        for k in range(n_bins):
            sel = bin_idx == k
            if int(sel.sum()) < min_per_bin:
                continue
            mu_b.append(float(clean[sel].mean()))
            var_b.append(float(resid[sel].var()))
        mus_all.append(np.array(mu_b, dtype=np.float64))
        vars_all.append(np.array(var_b, dtype=np.float64))
    return mus_all, vars_all


# --------------------------------------------------------------------------- #
#  Calibration-based fit (bias / dark / flat)                                 #
# --------------------------------------------------------------------------- #

def fit_from_calibration(
    calib_root: Path | str,
    sensor: str,
    gain: int,
) -> GainNoiseModel:
    """Fit per-channel K, read σ, row σ from bias/dark/flat frames.

    Layout::

        <calib_root>/
          bias/    *.dng           lens capped, minimum exposure  (≥2)
          dark/    *.dng           lens capped, long exposure     (≥1)
          flat/    level_*/a.dng   grey card, per-level pair       (≥2 levels)
                   level_*/b.dng
    """
    calib_root = Path(calib_root)
    bias_files = sorted(glob.glob(str(calib_root / "bias" / "*.dng")))
    dark_files = sorted(glob.glob(str(calib_root / "dark" / "*.dng")))
    flat_pairs: list[tuple[Path, Path]] = []
    for lv in sorted(glob.glob(str(calib_root / "flat" / "level_*"))):
        files = sorted(glob.glob(str(Path(lv) / "*.dng")))
        if len(files) >= 2:
            flat_pairs.append((Path(files[0]), Path(files[1])))
    if len(bias_files) < 2 or not flat_pairs:
        raise FileNotFoundError(
            f"Need ≥2 bias frames and ≥1 flat pair under {calib_root}")

    # ---- Per-channel K and read intercept from PTC via flat pairs -----------
    K_ch = [0.0] * 4
    b_ch = [0.0] * 4
    r2_ch = [0.0] * 4
    for c in range(4):
        mus: list[float] = []
        vars_: list[float] = []
        for a_path, b_path in flat_pairs:
            a = _load_packed_norm(a_path)[..., c]
            b = _load_packed_norm(b_path)[..., c]
            # discard clipped pixels; the sensor peaks near ~65500 DN16 = ~0.997
            valid = (a < 0.995) & (b < 0.995) & (a > 0.0) & (b > 0.0)
            if valid.mean() < 0.3:
                continue
            mus.append(float(((a[valid] + b[valid]) / 2).mean()))
            vars_.append(float(np.var((a[valid] - b[valid])) / 2.0))
        K_ch[c], b_ch[c], r2_ch[c] = _fit_line_with_intercept(
            np.asarray(mus), np.asarray(vars_))

    # ---- Per-channel read σ from the bias stack ------------------------------
    bias_stack = np.stack([_load_packed_norm(Path(p)) for p in bias_files], axis=0)
    mean_bias = bias_stack.mean(axis=0)
    read_sigma = [
        float(np.std((bias_stack - mean_bias)[..., c]))
        for c in range(4)
    ]

    # ---- Row σ from dark stack (or bias if no dark) --------------------------
    row_sigma = [0.0] * 4
    row_stack_paths = dark_files or bias_files
    if len(row_stack_paths) >= 2:
        stack = np.stack([_load_packed_norm(Path(p)) for p in row_stack_paths], axis=0)
        mean_frame = stack.mean(axis=0)
        for c in range(4):
            resid = stack[..., c] - mean_frame[..., c]
            row_means = resid.mean(axis=2)   # (N_frames, H_rows) — per-frame row means
            row_sigma[c] = float(row_means.std())

    # ---- BLE from per-frame bias means ---------------------------------------
    per_frame_bias = bias_stack.mean(axis=(1, 2, 3))
    ble = float(per_frame_bias.std())

    return GainNoiseModel(
        sensor=sensor,
        gain=int(gain),
        K=[float(k) for k in K_ch],
        read_sigma=read_sigma,
        row_sigma=row_sigma,
        read_shape=None,
        black_level_norm=ble,
        n_frames=int(len(bias_files) + len(dark_files) + 2 * len(flat_pairs)),
        fit_r2=[float(r) for r in r2_ch],
        fit_source=f"calibration:{calib_root}",
        notes=[
            f"flat_pairs={len(flat_pairs)}",
            f"bias_frames={len(bias_files)}",
            f"dark_frames={len(dark_files)}",
        ],
    )


# --------------------------------------------------------------------------- #
#  Burst-based fit (per gain, no lens cap required)                            #
# --------------------------------------------------------------------------- #

_AG_RE = re.compile(r"ag(\d+)$")


def fit_from_burst(
    burst_dir: Path | str,
    sensor: str,
    gain: int,
    *,
    n_burst: int = 128,
    n_gt: int = 128,
    center_crop: int | None = 1024,
    n_bins: int = 32,
    min_per_bin: int = 1024,
) -> GainNoiseModel:
    """Fit K, read σ, row σ from a static burst.

    ``burst_dir`` must contain ≥ ``n_gt + 16`` aligned DNG frames.  GT is the
    temporal mean of the first ``n_gt``; residuals of the next ``n_burst``
    frames vs. that mean are used to build the PTC.

    ``center_crop`` (pixels, post-pack) crops the center of each frame to keep
    memory bounded; use ``None`` for the full frame.
    """
    burst_dir = Path(burst_dir)
    files = sorted(burst_dir.glob("*.dng"))
    if len(files) < n_gt + 8:
        raise FileNotFoundError(
            f"{burst_dir}: need ≥{n_gt + 8} frames, found {len(files)}")

    # Build GT
    gt_stack = [_load_packed_norm(p) for p in files[:n_gt]]
    gt = np.stack(gt_stack, axis=0).mean(axis=0)  # (H, W, 4)
    del gt_stack

    if center_crop is not None:
        H, W, _ = gt.shape
        ch = min(center_crop, H)
        cw = min(center_crop, W)
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        gt = gt[y0:y0 + ch, x0:x0 + cw]

    # Residuals from held-out frames (independent of GT)
    used = files[n_gt:n_gt + n_burst] if len(files) > n_gt else files[:n_burst]
    mus_ch: list[list[float]] = [[] for _ in range(4)]
    vars_ch: list[list[float]] = [[] for _ in range(4)]
    row_sig_ch = [0.0] * 4
    ble_vals: list[float] = []

    # We accumulate binned stats across frames without keeping full stacks in memory.
    lo = np.percentile(gt.reshape(-1, 4), 0.5, axis=0)   # per-channel
    hi = np.percentile(gt.reshape(-1, 4), 99.5, axis=0)
    edges = [np.linspace(float(lo[c]), float(hi[c]), n_bins + 1) for c in range(4)]
    sum_x = [np.zeros(n_bins) for _ in range(4)]
    sum_x2 = [np.zeros(n_bins) for _ in range(4)]
    cnt = [np.zeros(n_bins, dtype=np.int64) for _ in range(4)]
    row_resid_stack: list[np.ndarray] = []

    for p in used:
        frame = _load_packed_norm(p)
        if center_crop is not None:
            frame = frame[y0:y0 + ch, x0:x0 + cw]
        resid = frame - gt
        ble_vals.append(float(resid.mean()))
        # per-frame row means for row noise σ
        row_resid_stack.append(resid.mean(axis=1))  # (H, 4)
        for c in range(4):
            r = resid[..., c].ravel()
            g = gt[..., c].ravel()
            idx = np.clip(np.digitize(g, edges[c]) - 1, 0, n_bins - 1)
            # per-bin residual mean and variance
            np.add.at(sum_x[c], idx, r)
            np.add.at(sum_x2[c], idx, r * r)
            np.add.at(cnt[c], idx, 1)

    K_ch = [0.0] * 4
    b_ch = [0.0] * 4
    r2_ch = [0.0] * 4
    for c in range(4):
        cok = cnt[c] >= min_per_bin
        if int(cok.sum()) < 3:
            # relax if too tight
            cok = cnt[c] >= max(16, min_per_bin // 8)
        counts = cnt[c][cok].astype(np.float64)
        means_r = sum_x[c][cok] / np.maximum(counts, 1.0)
        var_r = sum_x2[c][cok] / np.maximum(counts, 1.0) - means_r ** 2
        centers = 0.5 * (edges[c][:-1] + edges[c][1:])[cok]
        K_ch[c], b_ch[c], r2_ch[c] = _fit_line_with_intercept(centers, var_r)

    # Row σ per channel from stack of per-frame per-row residual means
    row_arr = np.stack(row_resid_stack, axis=0)  # (N, H, 4)
    for c in range(4):
        row_sig_ch[c] = float(row_arr[..., c].std())

    read_sigma = [float(np.sqrt(max(bc, 0.0))) for bc in b_ch]
    ble = float(np.std(ble_vals))

    return GainNoiseModel(
        sensor=sensor,
        gain=int(gain),
        K=[float(k) for k in K_ch],
        read_sigma=read_sigma,
        row_sigma=row_sig_ch,
        read_shape=None,
        black_level_norm=ble,
        n_frames=int(len(used)),
        fit_r2=[float(r) for r in r2_ch],
        fit_source=f"burst:{burst_dir}",
        notes=[f"n_gt={n_gt}", f"n_used={len(used)}", f"center_crop={center_crop}"],
    )


# --------------------------------------------------------------------------- #
#  Directory-scan convenience                                                 #
# --------------------------------------------------------------------------- #

def discover_bursts(bursts_root: Path | str,
                    gains: tuple[int, ...] = (128, 256, 512)) -> list[tuple[str, int, Path]]:
    """Yield ``(scene, gain, burst_dir)`` for every ``bursts/<scene>/ag<g>``."""
    bursts_root = Path(bursts_root)
    out: list[tuple[str, int, Path]] = []
    for scene_dir in sorted(bursts_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for g in gains:
            bd = scene_dir / f"ag{g}"
            if bd.is_dir() and any(bd.glob("*.dng")):
                out.append((scene_dir.name, int(g), bd))
    return out
