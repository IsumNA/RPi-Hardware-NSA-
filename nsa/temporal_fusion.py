"""Motion-aware temporal fusion for packed RAW bursts (Phase 2A).

For high-gain / low-light static scenes the right answer is a large-N mean
(256–512+ frames). Motion gating is optional and must NOT cap the effective
sample count below ``n_frames`` for static cabinets — that was why early
demos barely denoised (12 frames, k_cap=16).

Adapted from live.TemporalDenoiser motion gating, operating on packed raw
channels instead of demosaiced RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from nsa.raw_domain import load_packed, packed_to_rgb


@dataclass
class FusionConfig:
    n_frames: int = 512         # high-gain needs hundreds of frames
    tau: float = 0.04           # motion gate width in normalised packed units
    floor_quantile: float = 0.20
    k_cap: float = 512.0        # must track n_frames for static large-N averaging
    ref_index: int = 0          # reference frame for misalignment fallback
    mode: str = "mean"          # "mean" (static) | "motion" (gated)


def _motion_gate(
    frame: np.ndarray,
    acc: np.ndarray,
    *,
    tau: float,
    floor_quantile: float,
) -> np.ndarray:
    """Per-pixel gate in [0,1]; 1 = static, 0 = motion."""
    diff = np.abs(frame - acc).mean(axis=-1, keepdims=True)
    floor = min(float(np.quantile(diff, floor_quantile)), tau)
    excess = np.maximum(diff - floor, 0.0)
    return np.exp(-(excess / max(tau, 1e-6)) ** 2).astype(np.float32)


def fuse_burst_packed(
    frames: Sequence[np.ndarray],
    cfg: FusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse an ordered packed-raw burst. Returns (fused, weight_map).

    ``weight_map`` is the per-pixel effective sample count (H/2, W/2, 1).
    Default ``mode="mean"`` is a uniform average — correct for static
    high-gain cabinets. ``mode="motion"`` applies the noise-floor gate.
    """
    cfg = cfg or FusionConfig()
    if not frames:
        raise ValueError("fuse_burst_packed needs at least one frame")
    n_use = min(len(frames), max(1, cfg.n_frames))
    stack = [np.asarray(f, dtype=np.float32) for f in frames[:n_use]]
    k_cap = max(float(cfg.k_cap), float(n_use))

    if cfg.mode == "mean" or n_use == 1:
        acc = np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)
        count = np.full(acc.shape[:2] + (1,), float(n_use), dtype=np.float32)
        return np.clip(acc, 0.0, 1.0), count

    acc = stack[cfg.ref_index % len(stack)].copy()
    count = np.ones(acc.shape[:2] + (1,), np.float32)

    for i, frame in enumerate(stack):
        if i == cfg.ref_index % len(stack) and i == 0:
            continue
        gate = _motion_gate(frame, acc, tau=cfg.tau,
                            floor_quantile=cfg.floor_quantile)
        count = np.minimum(gate * count + 1.0, k_cap)
        acc += (frame - acc) / count

    return np.clip(acc, 0.0, 1.0), count


def fuse_paths(
    paths: Sequence[Path | str],
    cfg: FusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load DNG paths as packed raw and fuse."""
    cfg = cfg or FusionConfig()
    limit = max(1, cfg.n_frames)
    frames = [load_packed(Path(p)) for p in list(paths)[:limit]]
    return fuse_burst_packed(frames, cfg)


class BurstFusionAccumulator:
    """Streaming fusion — one packed frame at a time (live preview path)."""

    def __init__(self, cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        self.acc: np.ndarray | None = None
        self.count: np.ndarray | None = None
        self.seen = 0

    def reset(self) -> None:
        self.acc = None
        self.count = None
        self.seen = 0

    def step(self, packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        packed = np.asarray(packed, dtype=np.float32)
        if self.acc is None:
            self.acc = packed.copy()
            self.count = np.ones(packed.shape[:2] + (1,), np.float32)
            self.seen = 1
            return self.acc.copy(), self.count.copy()

        gate = _motion_gate(
            packed, self.acc,
            tau=self.cfg.tau,
            floor_quantile=self.cfg.floor_quantile,
        )
        self.count = np.minimum(gate * self.count + 1.0, self.cfg.k_cap)
        self.acc += (packed - self.acc) / self.count
        self.seen += 1
        return np.clip(self.acc, 0.0, 1.0).copy(), self.count.copy()


def resolve_burst_dir(pair_dir: Path, dataset_root: Path | None = None) -> Path | None:
    """Map imx662h pair folder → burst directory if it exists."""
    from nsa.raw_io import _find_burst_dir

    root = dataset_root or pair_dir.parents[2]  # …/PI_RAW/Data → PI_RAW
    return _find_burst_dir(root, pair_dir)


def compare_fusion(
    fused: np.ndarray,
    naive: np.ndarray,
    ref: np.ndarray,
    display_gain: float = 8.0,
) -> dict[str, float]:
    """PSNR of fused vs naive against a reference (packed or RGB)."""
    from nsa.inference import psnr

    def to_rgb(x):
        return packed_to_rgb(x, display_gain) if x.shape[-1] == 4 else x

    fr, nr, rr = to_rgb(fused), to_rgb(naive), to_rgb(ref)
    h = min(fr.shape[0], nr.shape[0], rr.shape[0])
    w = min(fr.shape[1], nr.shape[1], rr.shape[1])
    fr, nr, rr = fr[:h, :w], nr[:h, :w], rr[:h, :w]
    return {
        "psnr_fused": psnr(fr, rr),
        "psnr_naive": psnr(nr, rr),
        "gain_db": psnr(fr, rr) - psnr(nr, rr),
    }
