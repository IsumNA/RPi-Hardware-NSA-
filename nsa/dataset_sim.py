"""Build PI_RAW / denoise-hw training datasets from clean source images.

Workflow (5-phase calibrated noise)
-----------------------------------
  Phase 1–4  ``calibrate_noise.py`` on bias/dark/flat captures → noise model JSON
  Phase 5    this module — clean images + calibrated model → PI_RAW pairs

Legacy shortcut: pass ``sensor`` + ``gain`` to use the datasheet photon-transfer
model (``nsa.raw_io._synth_noisy_gt``) when no calibration JSON exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from nsa.raw_io import IMAGE_EXTS, _load_any, _synth_noisy_gt
from nsa.sensors import SENSOR_KEYS, get_sensor, with_noise_std

if TYPE_CHECKING:
    from nsa.noise_calib.model import NoiseModel

SIM_MARKER = ".nsa_simulated_dataset"
_SKIP_STEMS = frozenset({"noisy", "gt", "clean", "reference"})


@dataclass
class SimJob:
    scene: str
    image: Path
    out_dir: Path


@dataclass
class SimResult:
    scene: str
    image: str
    out_dir: str
    width: int
    height: int


def _safe_name(text: str) -> str:
    """Filesystem-safe token for folder names."""
    s = re.sub(r"[^\w.\-]+", "_", text.strip())
    return s.strip("_") or "frame"


def test_folder_name(
    sensor: str,
    gain: int,
    *,
    ag_tag: str | None = None,
    image_stem: str | None = None,
    multi_in_scene: bool = False,
) -> str:
    """denoise-hw style folder, e.g. ``imx662_ag12_test`` or ``imx662_ag12_study01_test``."""
    tag = ag_tag if ag_tag else f"ag{gain}"
    base = f"{sensor}_{tag}"
    if multi_in_scene and image_stem:
        base = f"{base}_{_safe_name(image_stem)}"
    return f"{base}_test"


def discover_clean_images(
    input_root: Path,
    *,
    layout: str = "auto",
    scene: str = "scene_001",
    recursive: bool = True,
) -> list[tuple[str, Path]]:
    """Map input tree → ``(scene_name, clean_image_path)`` pairs.

    * **auto** — subfolders that contain images become scenes; else flat ``scene``
    * **flat** — every image under ``input_root`` shares one ``scene``
    * **scenes** — only immediate child folders are scenes (non-recursive)
    """
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")

    def is_clean_image(p: Path) -> bool:
        return (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.stem.lower() not in _SKIP_STEMS
        )

    def images_in(folder: Path) -> list[Path]:
        if recursive:
            files = [p for p in folder.rglob("*") if is_clean_image(p)]
        else:
            files = [p for p in folder.iterdir() if is_clean_image(p)]
        return sorted(files)

    mode = layout.lower()
    if mode == "auto":
        child_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        scene_dirs = [d for d in child_dirs if images_in(d)]
        if scene_dirs:
            out: list[tuple[str, Path]] = []
            for sd in scene_dirs:
                for img in images_in(sd):
                    out.append((sd.name, img))
            return out
        return [(scene, p) for p in images_in(root)]

    if mode == "flat":
        return [(scene, p) for p in images_in(root)]

    if mode == "scenes":
        out = []
        for sd in sorted(d for d in root.iterdir() if d.is_dir()):
            for img in images_in(sd):
                out.append((sd.name, img))
        if out:
            return out
        raise ValueError(
            f"No scene subfolders with images under {root}. "
            "Use --layout flat or put clean images in child folders."
        )

    raise ValueError(f"Unknown layout {layout!r} — use auto, flat, or scenes")


def plan_jobs(
    input_root: Path,
    output_root: Path,
    *,
    sensor: str,
    gain: int,
    layout: str = "auto",
    scene: str = "scene_001",
    ag_tag: str | None = None,
    recursive: bool = True,
) -> list[SimJob]:
    """Assign each clean image an output PI_RAW test folder."""
    pairs = discover_clean_images(input_root, layout=layout, scene=scene,
                                  recursive=recursive)
    if not pairs:
        raise ValueError(f"No clean images found under {input_root}")

    counts: dict[str, int] = {}
    for sc, _ in pairs:
        counts[sc] = counts.get(sc, 0) + 1

    jobs: list[SimJob] = []
    for sc, img in pairs:
        multi = counts[sc] > 1
        folder = test_folder_name(
            sensor, gain, ag_tag=ag_tag, image_stem=img.stem, multi_in_scene=multi)
        out_dir = output_root / "Data" / sc / folder
        jobs.append(SimJob(scene=sc, image=img, out_dir=out_dir))
    return jobs


def _write_rgb_png(path: Path, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not write {path}")


def simulate_one(
    clean_path: Path,
    out_dir: Path,
    *,
    sensor: str,
    gain: int,
    temporal_frames: int = 64,
    seed: int = 0,
    noise_std: float | None = None,
    max_side: int = 0,
    noise_model: NoiseModel | None = None,
) -> SimResult:
    """Load one clean image, inject noise (Phase 5), write noisy.png + gt.png."""
    clean = _load_any(clean_path)
    if max_side and max(clean.shape[:2]) > max_side:
        h, w = clean.shape[:2]
        s = max_side / float(max(h, w))
        clean = cv2.resize(
            clean,
            (max(1, int(round(w * s))), max(1, int(round(h * s)))),
            interpolation=cv2.INTER_AREA,
        )

    if noise_model is not None:
        from nsa.noise_calib.synthesize import synthesize_pair
        noisy, gt = synthesize_pair(clean, noise_model, seed, temporal_frames)
    else:
        if sensor not in SENSOR_KEYS:
            raise ValueError(f"Unknown sensor {sensor!r}. Choose from: {list(SENSOR_KEYS)}")
        prof = with_noise_std(get_sensor(sensor), noise_std)
        noisy, gt = _synth_noisy_gt(clean, gain, prof, temporal_frames, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb_png(out_dir / "noisy.png", noisy)
    _write_rgb_png(out_dir / "gt.png", gt)
    return SimResult(
        scene=out_dir.parent.name,
        image=str(clean_path),
        out_dir=str(out_dir),
        width=int(gt.shape[1]),
        height=int(gt.shape[0]),
    )


def build_dataset(
    input_root: Path | str,
    output_root: Path | str,
    *,
    sensor: str = "imx662",
    gain: int = 256,
    layout: str = "auto",
    scene: str = "scene_001",
    ag_tag: str | None = None,
    temporal_frames: int = 64,
    seed: int = 662,
    noise_std: float | None = None,
    max_side: int = 0,
    recursive: bool = True,
    overwrite: bool = False,
    calibration: Path | str | None = None,
) -> dict:
    """Run the full input-folder → PI_RAW dataset pipeline (Phase 5)."""
    inp = Path(input_root).expanduser().resolve()
    out = Path(output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    noise_model = None
    if calibration:
        from nsa.noise_calib import load_model
        noise_model = load_model(calibration)
        sensor = noise_model.sensor
        gain = noise_model.gain

    jobs = plan_jobs(
        inp, out, sensor=sensor, gain=gain, layout=layout, scene=scene,
        ag_tag=ag_tag, recursive=recursive,
    )

    results: list[SimResult] = []
    for i, job in enumerate(jobs):
        noisy_p, gt_p = job.out_dir / "noisy.png", job.out_dir / "gt.png"
        if not overwrite and noisy_p.is_file() and gt_p.is_file():
            continue
        results.append(simulate_one(
            job.image, job.out_dir,
            sensor=sensor, gain=gain, temporal_frames=temporal_frames,
            seed=seed + i, noise_std=noise_std, max_side=max_side,
            noise_model=noise_model,
        ))

    manifest = {
        "kind": "nsa_simulated_dataset",
        "workflow": "calibrated_5phase" if noise_model else "datasheet_photon_transfer",
        "created": datetime.now(timezone.utc).isoformat(),
        "input": str(inp),
        "output": str(out),
        "calibration": str(calibration) if calibration else None,
        "sensor": sensor,
        "gain": gain,
        "ag_tag": ag_tag,
        "temporal_frames": temporal_frames,
        "noise_std": noise_std,
        "layout": layout,
        "pairs_written": len(results),
        "pairs_planned": len(jobs),
        "scenes": sorted({j.scene for j in jobs}),
        "results": [asdict(r) for r in results],
    }
    if noise_model is not None:
        manifest["noise_model"] = noise_model.to_dict()
    (out / SIM_MARKER).write_text(
        "Simulated PI_RAW dataset (clean images + injected sensor noise).\n"
        f"Generated {manifest['created']}\n",
        encoding="utf-8",
    )
    (out / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest
