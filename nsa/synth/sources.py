"""Clean-image sources for synthetic pair generation.

Two supported source types, both producing packed-Bayer float32 in [0, 1] of
shape ``(H, W, 4)`` — the same domain ``train_stream_to_gt.py`` consumes:

* **Burst averages** (:func:`burst_to_clean_packed`): temporal mean over the
  first N frames of a static burst gives a near-clean GT with the *exact*
  IMX662 optics / CFA / black level.  This is the highest-fidelity source
  we can produce from data on disk today.

* **Unprocessed sRGB** (:func:`unprocess_srgb_to_packed`): a light-weight
  inverse-ISP that maps an ordinary sRGB image to a plausible RGGB Bayer at
  12-bit precision.  Not scene-accurate (no vignetting, no per-pixel PRNU,
  no CCM inversion) — but sufficient to bulk-scale scene diversity for
  training, per Brooks et al. "Unprocessing Images for Learned Raw Denoising"
  (CVPR 19) and the AIM 2025 baseline.

The sources are written to disk as ``.npy`` (or ``.f16.npy`` when ``compress``
is on) under a cache directory.  The dataset class then samples clean frames
from those caches and injects noise on the fly.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
#  Burst → clean packed frame                                                 #
# --------------------------------------------------------------------------- #

def burst_to_clean_packed(
    burst_dir: Path | str,
    *,
    limit: int = 256,
    mode: str = "mean",
    alpha_trim: float = 0.25,
) -> np.ndarray:
    """Temporally reduce a burst folder to a single clean packed frame.

    ``mode`` — "mean" (fast, uses all frames) or "alpha_trim" (drops the
    lightest / darkest fraction per pixel, more robust to occasional bad
    frames).  Empirically the two agree within noise for a static scene.
    """
    from ..raw_domain import (
        burst_clean, burst_clean_alpha_trim, load_packed,
    )
    files = sorted(Path(burst_dir).glob("*.dng"))
    if not files:
        raise FileNotFoundError(f"No .dng under {burst_dir}")
    # limit <= 0 → use every available frame (max averaging).
    use = files if int(limit) <= 0 else files[: int(limit)]
    if mode == "mean":
        return burst_clean(use, limit=len(use))
    if mode == "alpha_trim":
        return burst_clean_alpha_trim(use, limit=len(use), trim=alpha_trim)
    raise ValueError(f"unknown mode={mode!r}")


# --------------------------------------------------------------------------- #
#  Unprocess sRGB → packed Bayer                                              #
# --------------------------------------------------------------------------- #

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Inverse sRGB gamma."""
    a = 0.055
    y = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return y.astype(np.float32)


def _random_white_balance(rng: np.random.Generator) -> np.ndarray:
    """Random per-channel WB gains ≥ 1, mirroring the AIM 2025 baseline."""
    # inverse-camera-WB style: RGB white balance is typically WB^-1 in [1.5, 2.5]
    # so cameras "undo" it — inverting means the raw R and B are darker than the
    # sRGB.  We pick 1 / [wr, 1, wb] with wr, wb ~ U[1.6, 2.4].
    wr = float(rng.uniform(1.6, 2.4))
    wb = float(rng.uniform(1.6, 2.4))
    return np.array([1.0 / wr, 1.0, 1.0 / wb], dtype=np.float32)


def _mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Turn an ``(H, W, 3)`` linear-RGB into packed RGGB ``(H/2, W/2, 4)``.

    We downsample the RGB to half-res and route the channels to the RGGB order
    of ``nsa.raw_domain.pack_raw``:
        packed[..., 0] = R  (top-left of 2×2)
        packed[..., 1] = G  (top-right → G1)
        packed[..., 2] = G  (bottom-left → G2)
        packed[..., 3] = B  (bottom-right)

    This is *packed-Bayer-equivalent* (identical statistics to what a real
    sensor would deliver post-``pack_raw``) without going through a physical
    demosaic → remosaic dance.
    """
    h = rgb.shape[0] // 2 * 2
    w = rgb.shape[1] // 2 * 2
    x = rgb[:h, :w]
    R = 0.25 * (x[0::2, 0::2, 0] + x[0::2, 1::2, 0]
                + x[1::2, 0::2, 0] + x[1::2, 1::2, 0])
    G1 = 0.5 * (x[0::2, 0::2, 1] + x[0::2, 1::2, 1])
    G2 = 0.5 * (x[1::2, 0::2, 1] + x[1::2, 1::2, 1])
    B = 0.25 * (x[0::2, 0::2, 2] + x[0::2, 1::2, 2]
                + x[1::2, 0::2, 2] + x[1::2, 1::2, 2])
    return np.stack([R, G1, G2, B], axis=-1).astype(np.float32)


def unprocess_srgb_to_packed(
    srgb: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    dark_scale: float = 0.35,
) -> np.ndarray:
    """Inverse-ISP an sRGB image to a packed RGGB clean frame in [0, 1].

    Steps (deliberately minimal — Brooks-style but stripped):
      1. cast to float [0, 1]
      2. inverse sRGB gamma (γ→ linear)
      3. inverse white balance (per-channel divide)
      4. scale down to a plausible low-ISO raw range (``dark_scale``)
      5. mosaic to RGGB packed
    """
    if rng is None:
        rng = np.random.default_rng()
    x = srgb.astype(np.float32)
    if x.max() > 1.5:  # 8-bit input
        x = x / 255.0
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    x = np.clip(x, 0.0, 1.0)
    x = _srgb_to_linear(x)
    x = x * _random_white_balance(rng)[None, None, :]
    x = np.clip(x * float(dark_scale), 0.0, 1.0)
    return _mosaic_rggb(x)


def load_srgb(path: Path) -> np.ndarray:
    """Load an sRGB image as float32 ``(H, W, 3)`` in ``[0, 1]``."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PIL/Pillow is required for sRGB source loading") from exc
    with Image.open(str(path)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
#  Building the clean cache                                                   #
# --------------------------------------------------------------------------- #

def _write_npy(arr: np.ndarray, out: Path, *, half: bool = True) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if half:
        np.save(out, arr.astype(np.float16))
    else:
        np.save(out, arr.astype(np.float32))
    return out


def build_burst_clean_cache(
    bursts_root: Path | str,
    cache_root: Path | str,
    *,
    limit: int = 256,
    mode: str = "alpha_trim",
    alpha_trim: float = 0.25,
    tags: Iterable[str] = ("ag1",),
    half: bool = True,
    log: bool = True,
    force_rebuild: bool = False,
) -> list[dict]:
    """Cache one packed-clean ``.npy`` per burst under ``cache_root/``.

    Uses ``ag1`` (lowest-noise) sub-bursts by default because those are what
    average down to the cleanest frames.  Higher-gain bursts *can* be used
    but they contribute more residual noise at every intensity.

    Set ``force_rebuild=True`` to ignore existing ``.npy`` caches (needed when
    raising ``limit`` / tightening ``alpha_trim`` for cleaner GT).
    """
    bursts_root = Path(bursts_root)
    cache_root = Path(cache_root)
    tag_set = set(tags)
    manifest: list[dict] = []
    for scene_dir in sorted(bursts_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for tag in sorted(scene_dir.iterdir()):
            if not tag.is_dir() or tag.name not in tag_set:
                continue
            n_dng = len(list(tag.glob("*.dng")))
            if n_dng < 1:
                continue
            out = cache_root / f"{scene_dir.name}__{tag.name}.npy"
            n_use = n_dng if int(limit) <= 0 else min(int(limit), n_dng)
            if out.exists() and not force_rebuild:
                if log:
                    print(f"  keep  {out.relative_to(cache_root)}", flush=True)
                arr_shape = np.load(out, mmap_mode="r").shape
                manifest.append({"path": str(out), "shape": list(arr_shape),
                                 "source": "burst", "scene": scene_dir.name,
                                 "burst_tag": tag.name,
                                 "n_frames_used": n_use, "mode": mode})
                continue
            if log:
                print(f"  build {out.relative_to(cache_root)}  "
                      f"(mode={mode}, N={n_use}/{n_dng}, trim={alpha_trim})",
                      flush=True)
            gt = burst_to_clean_packed(tag, limit=n_use, mode=mode,
                                       alpha_trim=alpha_trim)
            _write_npy(gt, out, half=half)
            manifest.append({"path": str(out), "shape": list(gt.shape),
                             "source": "burst", "scene": scene_dir.name,
                             "burst_tag": tag.name,
                             "n_frames_used": n_use,
                             "mode": mode,
                             "alpha_trim": float(alpha_trim)})
    return manifest


def build_srgb_clean_cache(
    srgb_root: Path | str,
    cache_root: Path | str,
    *,
    seed: int = 662,
    dark_scale: float = 0.35,
    tile: int | None = 1024,
    half: bool = True,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    log: bool = True,
    max_images: int | None = None,
) -> list[dict]:
    """Unprocess every image under ``srgb_root`` to a packed-Bayer ``.npy``.

    ``tile`` — if set, crop to (tile × tile) sRGB centered → (tile/2)² packed.
    Set to None to keep native resolution (bigger files, less variety).
    """
    srgb_root = Path(srgb_root)
    cache_root = Path(cache_root)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    files = [Path(p) for e in exts
             for p in glob.glob(str(srgb_root / f"**/*{e}"), recursive=True)]
    files = sorted(files)
    if max_images is not None:
        files = files[:max_images]
    for i, src in enumerate(files):
        rel = src.relative_to(srgb_root)
        out = cache_root / rel.with_suffix(".npy")
        if out.exists():
            arr_shape = np.load(out, mmap_mode="r").shape
            manifest.append({"path": str(out), "shape": list(arr_shape),
                             "source": "srgb", "src": str(rel)})
            continue
        try:
            rgb = load_srgb(src)
        except Exception as exc:
            if log:
                print(f"  skip  {rel}: {exc}", flush=True)
            continue
        if tile is not None:
            h, w = rgb.shape[:2]
            if h < tile or w < tile:
                continue
            y0 = (h - tile) // 2
            x0 = (w - tile) // 2
            rgb = rgb[y0:y0 + tile, x0:x0 + tile]
        packed = unprocess_srgb_to_packed(rgb, rng, dark_scale=dark_scale)
        _write_npy(packed, out, half=half)
        if log and (i % 25 == 0 or i == len(files) - 1):
            print(f"  [{i+1}/{len(files)}] {rel}  → {out.relative_to(cache_root)}",
                  flush=True)
        manifest.append({"path": str(out), "shape": list(packed.shape),
                         "source": "srgb", "src": str(rel),
                         "dark_scale": dark_scale})
    return manifest


def write_manifest(path: Path | str, entries: list[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    return p


def load_manifest(path: Path | str) -> list[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(d.get("entries", []))
