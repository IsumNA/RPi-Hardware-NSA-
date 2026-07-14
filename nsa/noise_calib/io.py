"""Load calibration frames (Phase 1 layout + I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nsa.raw_io import IMAGE_EXTS, SUPPORTED_EXTS, _load_any

FRAME_EXTS = SUPPORTED_EXTS


def load_linear(path: Path) -> np.ndarray:
    """Decode a frame to linear float32 (H×W or H×W×3) in [0, 1]."""
    return _load_any(path).astype(np.float32)


def load_raw_linear(path: Path) -> np.ndarray:
    """Load a frame in the TRUE sensor domain for noise calibration.

    For a DNG this returns the raw Bayer plane, black-level subtracted and
    white-level normalised — linear, NOT demosaiced, NOT gamma-mapped, NOT
    clipped at black. This is the only domain where read noise is a symmetric
    zero-mean distribution; ``rawpy.postprocess`` (what ``load_linear`` uses)
    applies a tone curve that ~2x-inflates shadow noise and clips the negative
    half at black, which makes the read/shot fits physically wrong.

    Non-raw inputs (PNG/JPG) have no recoverable raw domain, so fall back to
    processed luma with a warning left to the caller.
    """
    p = Path(path)
    if p.suffix.lower() in (".dng", ".raw", ".arw", ".nef", ".cr2"):
        import rawpy
        with rawpy.imread(str(p)) as r:
            raw = r.raw_image_visible.astype(np.float32)
            black = float(np.mean(r.black_level_per_channel))
            white = float(r.white_level)
        return (raw - black) / max(white - black, 1.0)
    return to_luma(load_linear(p))


def is_raw(path: Path) -> bool:
    return Path(path).suffix.lower() in (".dng", ".raw", ".arw", ".nef", ".cr2")


def to_luma(img: np.ndarray) -> np.ndarray:
    """Single-channel signal for statistics (ITU-R BT.601 on RGB)."""
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def list_frames(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in FRAME_EXTS
    )


def discover_phase1_root(root: Path) -> dict[str, list[Path] | list[tuple[Path, Path]]]:
    """Discover Phase-1 calibration captures under ``root``.

    Expected layout::

        <root>/
          bias/          *.dng|png   (lens capped, zero / minimal exposure)
          dark/          *.dng|png   (lens capped, normal exposure)
          flat/
            level_01/    a.* b.*     (uniform light pair)
            level_02/    ...
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration root not found: {root}")

    bias = list_frames(root / "bias")
    dark = list_frames(root / "dark")

    flat_pairs: list[tuple[Path, Path]] = []
    flat_root = root / "flat"
    if flat_root.is_dir():
        level_dirs = sorted(d for d in flat_root.iterdir() if d.is_dir())
        if not level_dirs:
            # loose pairs: flat/a_01.png flat/b_01.png naming
            files = list_frames(flat_root)
            for i in range(0, len(files) - 1, 2):
                flat_pairs.append((files[i], files[i + 1]))
        else:
            for lv in level_dirs:
                files = list_frames(lv)
                if len(files) >= 2:
                    flat_pairs.append((files[0], files[1]))

    return {"bias": bias, "dark": dark, "flat_pairs": flat_pairs}
