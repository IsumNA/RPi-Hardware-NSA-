"""Build clean ground-truth images from multi-frame RAW bursts.

Temporal averaging is the standard way to obtain GT for noise calibration and
synthetic dataset generation: independent read noise cancels while static scene
detail is preserved.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nsa.raw_io import SUPPORTED_EXTS, _load_any


def list_burst_frames(folder: Path) -> list[Path]:
    """Sorted image/RAW files in a burst folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Burst folder not found: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        and p.stem.lower() not in ("readme", "capture")
    )
    if not files:
        raise ValueError(f"No image files in burst folder: {folder}")
    return files


def temporal_average_gt(
    frame_paths: list[Path],
    *,
    min_frames: int = 8,
    max_side: int = 0,
    align: bool = True,
) -> np.ndarray:
    """Average a burst into a single clean linear RGB image in [0, 1].

  Parameters
  ----------
  frame_paths:
      Paths to burst frames (DNG/PNG/…).
  min_frames:
      Minimum frames required.
  max_side:
      Downscale so longest side <= N before averaging (0 = native).
  align:
      ECC-align frames to the first before averaging (helps minor tripod drift).
    """
    if len(frame_paths) < min_frames:
        raise ValueError(
            f"Need at least {min_frames} burst frames, got {len(frame_paths)}"
        )

    acc: np.ndarray | None = None
    ref_gray: np.ndarray | None = None
    warp = np.eye(2, 3, dtype=np.float32)
    n = 0

    for path in frame_paths:
        img = _load_any(path).astype(np.float32)
        if max_side and max(img.shape[:2]) > max_side:
            h, w = img.shape[:2]
            s = max_side / float(max(h, w))
            img = cv2.resize(
                img,
                (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                interpolation=cv2.INTER_AREA,
            )
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        gray = cv2.cvtColor(
            (np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY,
        )
        if ref_gray is None:
            ref_gray = gray
        elif align:
            try:
                _, warp = cv2.findTransformECC(
                    ref_gray, gray, warp, cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
                )
                h, w = img.shape[:2]
                img = cv2.warpAffine(
                    img, warp, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                )
            except cv2.error:
                pass  # keep unaligned frame

        if acc is None:
            acc = np.zeros_like(img, dtype=np.float64)
        acc += img.astype(np.float64)
        n += 1

    assert acc is not None and n > 0
    return np.clip(acc / n, 0.0, 1.0).astype(np.float32)


def write_gt_png(path: Path, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not write {path}")


def burst_folder_to_gt(
    burst_dir: Path | str,
    output_path: Path | str,
    *,
    min_frames: int = 8,
    max_side: int = 0,
    align: bool = True,
) -> dict:
    """Average a burst folder and write a GT image. Returns a small manifest."""
    burst_dir = Path(burst_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    frames = list_burst_frames(burst_dir)
    gt = temporal_average_gt(
        frames, min_frames=min_frames, max_side=max_side, align=align,
    )
    write_gt_png(output_path, gt)
    return {
        "burst": str(burst_dir),
        "output": str(output_path),
        "frames_used": len(frames),
        "width": int(gt.shape[1]),
        "height": int(gt.shape[0]),
    }
