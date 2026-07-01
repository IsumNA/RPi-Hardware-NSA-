"""denoise-hw / PI_RAW dataset helpers.

The training pipeline at https://github.com/davidplowman/denoise-hw expects paired
``noisy.dng`` + ``gt.dng`` folders under a ``PI_RAW/Data/…`` tree (not shipped in
that repo — usually at ``/opt/datasets/PI_RAW`` on a Pi). NSA auto-detects that
layout and falls back to ``datasets/PI_RAW`` in this project.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nsa.raw_io import find_paired_folders, list_frames

# Same test folder denoise-hw's run.sh uses (test.py cabinet_D50_100/imx219_ag12_test).
DEFAULT_TEST_REL = "Data/cabinet_D50_100/imx219_ag12_test"
DEFAULT_FILTER = ["imx219", "ag12"]

PI_RAW_SEARCH = (
    Path("datasets/PI_RAW"),
    Path("../datasets/PI_RAW"),
    Path("/opt/datasets/PI_RAW"),
    Path(os.environ.get("PI_RAW", "")) if os.environ.get("PI_RAW") else None,
)


def resolve_pi_raw(explicit: str | Path | None = None) -> Path | None:
    """Return the first usable PI_RAW root (folder with paired captures)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(p for p in PI_RAW_SEARCH if p)
    seen: set[str] = set()
    for root in candidates:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        if not root.exists():
            continue
        if find_paired_folders(str(root)):
            return root.resolve()
    return None


def normalize_dataset_root(path: str | Path) -> Path:
    """Accept PI_RAW root, PI_RAW/Data, a scene folder, or denoise-hw clone parent."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return p
    if find_paired_folders(str(p)):
        return p
    # denoise-hw clone sitting next to datasets/PI_RAW
    sibling = p.parent / "datasets" / "PI_RAW"
    if sibling.exists() and find_paired_folders(str(sibling)):
        return sibling.resolve()
    for sub in (p / "datasets" / "PI_RAW", p / "PI_RAW", p / "Data"):
        if sub.exists() and find_paired_folders(str(sub if sub.name != "Data" else p)):
            return (p if sub.name == "Data" and find_paired_folders(str(p))
                    else sub).resolve()
    return p


def dataset_summary(root: Path | None) -> dict:
    """Quick manifest for CLI / GUI."""
    if root is None or not root.exists():
        return {"root": None, "paired_folders": 0, "samples": []}
    pairs = find_paired_folders(str(root))
    frames = list_frames(str(root), limit=8)
    return {
        "root": str(root),
        "paired_folders": len(pairs),
        "samples": [f["name"] for f in frames],
    }


def apply_auto_dataset(cfg) -> bool:
    """If config has no real dataset, enable PI_RAW when found. Returns True if set."""
    root = resolve_pi_raw(cfg.sensor.dataset_path)
    if root is None:
        return False
    if not cfg.sensor.dataset_path:
        cfg.sensor.dataset_path = str(root)
    if not cfg.sensor.real_capture:
        cfg.sensor.real_capture = True
    if not cfg.sensor.filter:
        # Prefer the canonical denoise-hw test scene when present.
        test_dir = root / DEFAULT_TEST_REL
        if test_dir.is_dir():
            cfg.sensor.filter = list(DEFAULT_FILTER)
    return True


def clone_denoise_hw(dest: Path) -> Path:
    """Shallow-clone davidplowman/denoise-hw (reference code + tunings)."""
    dest = dest.resolve()
    if (dest / "dataset.py").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "nafnet",
         "https://github.com/davidplowman/denoise-hw.git", str(dest)],
        check=True,
    )
    return dest


def build_sample_pi_raw(out_root: Path, seed: int = 219) -> int:
    """Write denoise-hw-style paired PNG folders when no real PI_RAW is available."""
    import cv2
    import numpy as np

    from nsa.raw_io import _synth_noisy_gt, _synthetic_scene
    from nsa.sensors import get_sensor

    scenes = (
        ("Data/cabinet_D50_100/imx219_ag12_test", "imx219", 512),
        ("Data/cabinet_D50_100/imx662_ag12_test", "imx662", 512),
        ("Data/cabinet_H_2/imx219_ag1_test", "imx219", 256),
    )
    patch = 512
    n = 0
    for rel, sensor_key, gain in scenes:
        folder = out_root / rel
        folder.mkdir(parents=True, exist_ok=True)
        noisy_p, gt_p = folder / "noisy.png", folder / "gt.png"
        if noisy_p.exists() and gt_p.exists():
            n += 1
            continue
        sensor = get_sensor(sensor_key)
        clean = _synthetic_scene(patch * 2, patch * 2, seed + n)
        noisy, gt = _synth_noisy_gt(clean, gain, sensor, temporal_frames=64,
                                    seed=seed + n + 100)
        for arr, path in ((noisy, noisy_p), (gt, gt_p)):
            img8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
        n += 1
    return n


def ensure_project_dataset(project_root: Path | None = None) -> Path:
    """Guarantee datasets/PI_RAW exists (sample PNGs or symlink to system PI_RAW)."""
    root = project_root or Path(__file__).resolve().parents[1]
    local = root / "datasets" / "PI_RAW"
    system = resolve_pi_raw("/opt/datasets/PI_RAW")
    if system and system != local.resolve():
        if not local.exists():
            try:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.symlink_to(system, target_is_directory=True)
                return local
            except OSError:
                pass
    if not find_paired_folders(str(local)):
        build_sample_pi_raw(local)
    return local.resolve()
