"""Genuine GT-quality denoise for static IMX662 / IMX662H scenes.

The only estimator that *equals* a ~100-frame ground truth is a ~100-frame
(or N-frame) average in the same domain the GT was built in. Neural single-frame
regression cannot invent the photons that averaging collects — it converges to a
soft conditional mean.

This module implements that fact:

1. ``merge_burst`` — align + average packed RAW (or linear RGB). This **is** the
   GT method. Use it whenever a burst exists (capture, offline, live accumulate).
2. ``denoise_single_preserve`` — dual-domain single-frame fallback that keeps
   resolution-chart contrast instead of NAFNet-style plastic blur: strong smooth
   on the base layer, SNR-gated detail kept from the noisy frame.
3. ``solve`` — auto: burst folder → merge; single file → preserve path.

Training a network is optional polish on top of (1), not a replacement for it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_BURST_EXTS = {".dng", ".npy", ".tif", ".tiff", ".png"}


def list_burst_files(folder: Path | str) -> list[Path]:
    folder = Path(folder)
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _BURST_EXTS
        and p.stem.lower() not in {"readme", "capture", "noisy", "gt"}
    )
    if not files:
        raise FileNotFoundError(f"No burst frames in {folder}")
    return files


def _load_linear(path: Path) -> np.ndarray:
    from nsa.raw_io import _load_any
    return _load_any(path).astype(np.float32)


def _to_gray01(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _ecc_align(ref_gray: np.ndarray, mov_rgb: np.ndarray,
               warp: np.ndarray | None = None) -> np.ndarray:
    """Euclidean ECC align ``mov_rgb`` to ``ref_gray``; return warped RGB."""
    if warp is None:
        warp = np.eye(2, 3, dtype=np.float32)
    mov_gray = _to_gray01(mov_rgb)
    # ECC wants similar dynamic range
    r8 = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    m8 = cv2.normalize(mov_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try:
        _, warp = cv2.findTransformECC(
            r8, m8, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
        )
        h, w = mov_rgb.shape[:2]
        return cv2.warpAffine(
            mov_rgb, warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error:
        return mov_rgb


def merge_burst(
    frames: list[np.ndarray] | list[Path] | Path,
    *,
    max_frames: int = 0,
    align: bool = True,
    skip_first_for_holdout: bool = False,
) -> np.ndarray:
    """Temporal average → GT-quality clean image (linear RGB [0,1]).

    This is not a neural approximation of the ground truth. For a static scene it
    *is* how the ground truth is defined. ``max_frames=100`` reproduces the
    ~100-frame look the user wants.
    """
    if isinstance(frames, (str, Path)):
        paths = list_burst_files(frames)
        if max_frames and len(paths) > max_frames:
            # Spread across the burst (avoid only using the GT-window head).
            idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
            paths = [paths[i] for i in idx]
        if skip_first_for_holdout and len(paths) > 2:
            paths = paths[1:]
        imgs = [_load_linear(p) for p in paths]
    else:
        imgs = []
        for f in frames:
            if isinstance(f, (str, Path)):
                imgs.append(_load_linear(Path(f)))
            else:
                imgs.append(np.asarray(f, dtype=np.float32))
        if max_frames and len(imgs) > max_frames:
            imgs = imgs[:max_frames]

    if not imgs:
        raise ValueError("merge_burst: no frames")

    ref = imgs[0]
    acc = np.zeros_like(ref, dtype=np.float64)
    ref_gray = _to_gray01(ref)
    n = 0
    warp = np.eye(2, 3, dtype=np.float32)
    for i, img in enumerate(imgs):
        if img.shape[:2] != ref.shape[:2]:
            img = cv2.resize(img, (ref.shape[1], ref.shape[0]),
                            interpolation=cv2.INTER_AREA)
        if align and i > 0:
            img = _ecc_align(ref_gray, img, warp)
        acc += img.astype(np.float64)
        n += 1
    return np.clip(acc / max(n, 1), 0.0, 1.0).astype(np.float32)


def estimate_noise_sigma(rgb: np.ndarray) -> float:
    """MAD estimator on Haar high-frequency (robust to edges)."""
    gray = _to_gray01(rgb)
    hx = gray[:, 1:] - gray[:, :-1]
    hy = gray[1:, :] - gray[:-1, :]
    mad = np.median(np.abs(np.concatenate([hx.ravel(), hy.ravel()])))
    return float(mad / 0.6745 + 1e-8)


def denoise_single_preserve(
    noisy: np.ndarray,
    *,
    base_d: int = 9,
    base_sigma_color: float | None = None,
    base_sigma_space: float = 9.0,
    detail_k: float = 3.5,
) -> np.ndarray:
    """Single-frame denoise that keeps resolution-bar contrast.

    Split into base + detail:
      * base  — bilateral (edge-aware smooth); kills grain in flats
      * detail — noisy − gaussian(noisy); keep coefficients above ``detail_k·σ``
        so real chirp/edge energy survives while sub-threshold grain dies

    This will not equal a 100-frame average (missing photons), but it will not
    plastic-blur fine bars the way L1-trained NAFNet does.
    """
    noisy = np.clip(noisy.astype(np.float32), 0.0, 1.0)
    sigma = estimate_noise_sigma(noisy)
    if base_sigma_color is None:
        # bilateral color sigma in 0-255 units
        base_sigma_color = float(np.clip(30.0 + 500.0 * sigma, 25.0, 90.0))

    img8 = (noisy * 255.0 + 0.5).astype(np.uint8)
    base8 = cv2.bilateralFilter(img8, base_d, base_sigma_color, base_sigma_space)
    # Second pass on flats only-ish: mild NL-means for chroma splotches
    base8 = cv2.fastNlMeansDenoisingColored(base8, None, 3, 3, 7, 21)
    base = base8.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(noisy, (0, 0), 1.2)
    detail = noisy - blur
    thr = detail_k * sigma
    abs_d = np.abs(detail)
    kept = np.sign(detail) * np.maximum(abs_d - thr, 0.0)
    # Only reinject detail near strong base edges (resolution bars / patch borders)
    base_edge = np.abs(base - cv2.GaussianBlur(base, (0, 0), 1.2))
    edge_strength = base_edge.mean(axis=-1, keepdims=True) if base_edge.ndim == 3 else base_edge
    gate = np.clip(edge_strength / (np.percentile(edge_strength, 90) + 1e-6), 0.0, 1.0)
    out = np.clip(base + kept * gate, 0.0, 1.0)
    return out.astype(np.float32)


def merge_burst_motion(
    frames: list[np.ndarray] | list[Path] | Path,
    *,
    max_frames: int = 8,
    ref_index: int = 0,
    flow_scale: float = 0.5,
    consistency: float = 0.02,
) -> np.ndarray:
    """Motion-aware robust merge — for short bursts with camera/subject motion.

    Unlike a blind average (which ghosts under motion), this:
      1. Picks a reference frame (the one you care about — sharp pose)
      2. Optical-flow warps neighbours onto the reference
      3. Photometric-consistency weights: misaligned / moving pixels get ~0 weight
         so they do **not** smear into the result
      4. Falls back to the reference where nothing agrees

    Still not magic for large motion — those pixels stay single-frame. But static
    background gets multi-frame SNR while moving subjects keep the ref's sharpness.
    """
    if isinstance(frames, (str, Path)):
        paths = list_burst_files(frames)
        if max_frames and len(paths) > max_frames:
            # Prefer frames near the reference for small motion
            mid = len(paths) // 2
            lo = max(0, mid - max_frames // 2)
            paths = paths[lo:lo + max_frames]
        imgs = [_load_linear(p) for p in paths]
    else:
        imgs = [
            _load_linear(Path(f)) if isinstance(f, (str, Path)) else np.asarray(f, np.float32)
            for f in frames
        ]
        if max_frames:
            imgs = imgs[:max_frames]

    if len(imgs) == 1:
        return imgs[0]
    ref_index = int(np.clip(ref_index, 0, len(imgs) - 1))
    ref = imgs[ref_index]
    h, w = ref.shape[:2]
    ref_g = _to_gray01(ref)
    # Downscale for flow speed
    small = (max(32, int(w * flow_scale)), max(32, int(h * flow_scale)))

    acc = np.zeros_like(ref, dtype=np.float64)
    wsum = np.zeros(ref.shape[:2], dtype=np.float64)[..., None]

    for i, img in enumerate(imgs):
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if i == ref_index:
            weight = np.ones((h, w, 1), dtype=np.float64)
            warped = img
        else:
            g = _to_gray01(img)
            r_s = cv2.resize(ref_g, small, interpolation=cv2.INTER_AREA)
            g_s = cv2.resize(g, small, interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(
                (r_s * 255).astype(np.uint8), (g_s * 255).astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            # Upscale flow to full res
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= w / small[0]
            flow[..., 1] *= h / small[1]
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            # Photometric consistency vs reference
            err = np.mean(np.abs(warped - ref), axis=-1, keepdims=True)
            weight = np.exp(-err / max(consistency, 1e-6)).astype(np.float64)
        acc += warped.astype(np.float64) * weight
        wsum += weight

    out = acc / np.maximum(wsum, 1e-6)
    # Where almost no support, keep reference (moving subject)
    weak = wsum[..., 0] < 0.35
    out[weak] = ref[weak]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def solve(
    source: Path | str,
    *,
    max_frames: int = 100,
    align: bool = True,
    single: str = "preserve",
    motion: bool = False,
) -> dict:
    """Auto denoise.

    * burst + ``motion=False`` → static merge (lab / tripod only)
    * burst + ``motion=True``  → flow-weighted robust merge (handles motion)
    * file → single-frame preserve (or pass a sharp_single checkpoint separately)
    """
    source = Path(source).expanduser()
    if source.is_dir():
        files = list_burst_files(source)
        if motion:
            n = min(len(files), max_frames if max_frames else 8)
            rgb = merge_burst_motion(source, max_frames=min(n, 8) if max_frames > 8 else n)
            return {"rgb": rgb, "mode": "motion_merge", "frames_used": n,
                    "path": str(source)}
        n = min(len(files), max_frames) if max_frames else len(files)
        rgb = merge_burst(source, max_frames=max_frames, align=align)
        return {"rgb": rgb, "mode": "burst_merge", "frames_used": n,
                "path": str(source)}
    if not source.is_file():
        raise FileNotFoundError(source)
    noisy = _load_linear(source)
    if single == "none":
        out, mode = noisy, "identity"
    else:
        out, mode = denoise_single_preserve(noisy), "single_preserve"
    return {"rgb": out, "mode": mode, "frames_used": 1, "path": str(source)}


def chirp_contrast(rgb: np.ndarray, row_frac: float = 0.78,
                   x0_frac: float = 0.55, x1_frac: float = 0.92) -> float:
    """Peak-to-peak contrast on a horizontal slice through the resolution bars."""
    h, w = rgb.shape[:2]
    y = int(h * row_frac)
    x0, x1 = int(w * x0_frac), int(w * x1_frac)
    line = _to_gray01(rgb)[y, x0:x1]
    return float(line.max() - line.min())


def proof_synthetic(out_dir: Path | str, *, n_burst: int = 100,
                    noise: float = 0.08, seed: int = 0) -> dict:
    """Prove burst-merge recovers chirp contrast; single-preserve beats blur.

    Writes a panel: clean | noisy | bilateral-blurry | preserve | merge-100.
    """
    from nsa.raw_io import _synthetic_scene
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clean = _synthetic_scene(512, 512, seed)
    burst = [
        np.clip(clean + rng.normal(0, noise, clean.shape).astype(np.float32), 0, 1)
        for _ in range(n_burst)
    ]
    noisy = burst[0]
    # Classic over-smooth (what L1 nets converge toward)
    blurry = cv2.GaussianBlur(noisy, (0, 0), 2.5)
    preserve = denoise_single_preserve(noisy)
    merged = merge_burst(burst, align=False)

    c_clean = chirp_contrast(clean)
    metrics = {
        "chirp_clean": c_clean,
        "chirp_noisy": chirp_contrast(noisy),
        "chirp_blurry": chirp_contrast(blurry),
        "chirp_preserve": chirp_contrast(preserve),
        "chirp_merge100": chirp_contrast(merged),
        "psnr_blurry": _psnr(blurry, clean),
        "psnr_preserve": _psnr(preserve, clean),
        "psnr_merge100": _psnr(merged, clean),
    }
    # Recovery ratios vs clean chirp (1.0 = perfect)
    metrics["recovery_blurry"] = metrics["chirp_blurry"] / max(c_clean, 1e-8)
    metrics["recovery_preserve"] = metrics["chirp_preserve"] / max(c_clean, 1e-8)
    metrics["recovery_merge100"] = metrics["chirp_merge100"] / max(c_clean, 1e-8)

    panel = np.concatenate([clean, noisy, blurry, preserve, merged], axis=1)
    _save_rgb(out_dir / "proof_panel.png", panel)
    # Zoom the chirp
    z = slice(int(512 * 0.62), int(512 * 0.92)), slice(int(512 * 0.55), int(512 * 0.92))
    zoom = np.concatenate([clean[z], noisy[z], blurry[z], preserve[z], merged[z]], axis=1)
    _save_rgb(out_dir / "proof_chirp_zoom.png", zoom)
    (out_dir / "proof_metrics.json").write_text(
        __import__("json").dumps(metrics, indent=2))
    return metrics


def _psnr(a, b) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
