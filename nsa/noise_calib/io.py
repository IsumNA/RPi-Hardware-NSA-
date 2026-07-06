"""Load calibration frames (Phase 1 layout + I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nsa.raw_io import IMAGE_EXTS, SUPPORTED_EXTS, _load_any

FRAME_EXTS = SUPPORTED_EXTS


def load_linear(path: Path) -> np.ndarray:
    """Decode a frame to linear float32 (H×W or H×W×3) in [0, 1]."""
    return _load_any(path).astype(np.float32)


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
