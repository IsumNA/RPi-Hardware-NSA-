"""Burst alignment for Pi CTT cache → denoise-hw PI_RAW pairs.

Promotes the scratchpad HCG recovery workflow to a reusable library. The Pi
syncs its full CTT workspace (``imx662/``) to ``Pi_Unique_Cache`` on the AI
server; this module sorts flat/raw-cache DNGs into per-gain burst folders and
builds ``noisy.dng`` + ``gt.tif`` test pairs under ``PI_RAW/Data/``.

Expected inputs
---------------
**Pi_Unique_Cache** (from ``scripts/sync_pi_burst_cache.sh``)::

    Pi_Unique_Cache/
      project.json          # CTT capture metadata (see below)
      imx662_5000k_5l_….dng  # flat DNG files at sync root

Override path with env ``PI_UNIQUE_CACHE`` (default
``/opt/datasets/PI_RAW/Pi_Unique_Cache``).

**project.json** structure (Pi CTT server workspace)::

    {
      "captures": [
        {
          "filename": "imx662_5000k_5l_00042.dng",
          "captured_at": "2026-07-10T09:12:34Z",
          "controls": {"gain": 3.24, "exposure": 1200, "lux": 5}
        },
        …
      ]
    }

Per-frame ``controls.gain`` is the *measured* analogue gain (not the requested
tag). HCG scenes are reconstructed by mapping rounded gain → ``ag2``…``ag512``
(``ag1`` is unreachable in HCG on this sensor).

**hcg_sort_manifest.json** (optional intermediate)::

    {
      "cabinet_H_2": {"ag2": ["imx662_5000k_5l_….dng", …], …},
      "cabinet_D_10": {…},
      "cabinet_F_5": {…}
    }

Built automatically from ``project.json`` unless you pass ``--manifest``.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from nsa.dataset_layout import HCG_ILLUM_SCENES, resolve_layout

# Measured HCG gain (rounded to 2 dp) → denoise-hw ag folder tag.
# ag1 is not reachable in HCG mode on this rig.
HCG_GAIN_TO_TAG: dict[float, str] = {
    3.24: "ag2",
    3.98: "ag4",
    7.94: "ag8",
    15.85: "ag16",
    31.62: "ag32",
    63.1: "ag64",
    125.89: "ag128",
    251.19: "ag256",
    501.19: "ag512",
}

HCG_SENSOR_PREFIX = "imx662h"
LCG_SENSOR_PREFIX = "imx662"
HCG_FRAME_PREFIX = "imx662_5000k_"

DEFAULT_PI_UNIQUE_CACHE = Path(
    os.environ.get("PI_UNIQUE_CACHE", "/opt/datasets/PI_RAW/Pi_Unique_Cache"),
)
DEFAULT_GT_FRAMES = 256
DEFAULT_LCG_NOISY_PICK = 300
DEFAULT_HCG_NOISY_PICK = 60
DEFAULT_MIN_BURST_FRAMES = 10

LCG_GAIN_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
HCG_GAIN_SWEEP: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class SceneRule:
    """How to pick frames for one HCG illuminant scene from project.json."""

    scene: str
    filename_prefix: str
    # Optional extra filter on capture dicts (e.g. cabinet_F_5 time window).
    frame_filter: Callable[[dict[str, Any]], bool] | None = None


def _f5_time_window(capture: dict[str, Any]) -> bool:
    """Isolate cabinet_F_5 inside the entangled ``1l`` bucket (2026-07-10 ~10:14–10:31 UTC)."""
    ts = capture.get("captured_at") or ""
    return any(
        needle in ts
        for needle in ("2026-07-10T10:1", "2026-07-10T10:2", "2026-07-10T10:3")
    )


DEFAULT_HCG_SCENE_RULES: tuple[SceneRule, ...] = (
    SceneRule("cabinet_H_2", "imx662_5000k_5l_"),
    SceneRule("cabinet_D_10", "imx662_5000k_398l_"),
    SceneRule("cabinet_F_5", "imx662_5000k_1l_", frame_filter=_f5_time_window),
)


@dataclass
class AlignResult:
    """Summary returned by :func:`align_pi_cache`."""

    manifest_path: Path | None = None
    sorted_bursts: dict[str, dict[str, int]] = field(default_factory=dict)
    pairs_built: list[dict[str, Any]] = field(default_factory=list)
    h2_split: dict[str, int] | None = None
    skipped: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pi_unique_cache() -> Path:
    return DEFAULT_PI_UNIQUE_CACHE.expanduser()


def default_project_root() -> Path:
    return repo_root() / "datasets" / "imx662_project"


def project_json_in_cache(cache_root: Path | str) -> Path:
    """``project.json`` lives at the sync root (``imx662/project.json`` on the Pi)."""
    cache_root = Path(cache_root).expanduser()
    for cand in (cache_root / "project.json", cache_root / "imx662" / "project.json"):
        if cand.is_file():
            return cand
    return cache_root / "project.json"


def load_project_json(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"project.json not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"project.json root must be an object, got {type(data).__name__}")
    if "captures" not in data:
        raise ValueError("project.json missing required key 'captures'")
    if not isinstance(data["captures"], list):
        raise ValueError("project.json 'captures' must be a list")
    return data


def frames_with_prefix(
    captures: Sequence[dict[str, Any]],
    prefix: str,
    *,
    frame_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Sorted capture records whose filename starts with *prefix*."""
    out: list[dict[str, Any]] = []
    for capture in captures:
        fn = capture.get("filename", "")
        if not fn.startswith(prefix):
            continue
        if frame_filter is not None and not frame_filter(capture):
            continue
        ctrl = capture.get("controls") or {}
        out.append({
            "filename": fn,
            "gain": ctrl.get("gain"),
            "exposure": ctrl.get("exposure"),
            "lux": ctrl.get("lux"),
            "captured_at": capture.get("captured_at"),
        })
    out.sort(key=lambda item: item.get("captured_at") or "")
    return out


def assign_ag_tags(
    items: Sequence[dict[str, Any]],
    *,
    gain_round: int = 2,
    gain_map: dict[float, str] | None = None,
) -> dict[str, list[str]]:
    """Map measured per-frame gain → ``ag*`` tag → list of filenames."""
    mapping = gain_map or HCG_GAIN_TO_TAG
    by_tag: dict[str, list[str]] = {}
    for item in items:
        gain = item.get("gain")
        if gain is None:
            continue
        tag = mapping.get(round(float(gain), gain_round))
        if tag is None:
            continue
        by_tag.setdefault(tag, []).append(str(item["filename"]))
    return by_tag


def build_hcg_sort_manifest(
    project_json: Path | str,
    *,
    scene_rules: Sequence[SceneRule] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Build scene → ag_tag → [filenames] from Pi ``project.json`` metadata."""
    data = load_project_json(project_json)
    captures = data["captures"]
    rules = scene_rules or DEFAULT_HCG_SCENE_RULES
    manifest: dict[str, dict[str, list[str]]] = {}
    for rule in rules:
        items = frames_with_prefix(
            captures, rule.filename_prefix, frame_filter=rule.frame_filter,
        )
        manifest[rule.scene] = assign_ag_tags(items)
    return manifest


def manifest_file_list(manifest: dict[str, dict[str, list[str]]]) -> list[str]:
    """Unique sorted filenames referenced by a sort manifest."""
    files: set[str] = set()
    for tags in manifest.values():
        for names in tags.values():
            files.update(names)
    return sorted(files)


def write_manifest(manifest: dict[str, Any], path: Path | str) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path | str) -> dict[str, dict[str, list[str]]]:
    path = Path(path).expanduser()
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def resolve_cache_file(cache_root: Path, filename: str) -> Path | None:
    """Locate a DNG inside a flat or nested Pi cache tree."""
    cache_root = cache_root.expanduser()
    direct = cache_root / filename
    if direct.is_file():
        return direct
    nested = cache_root / "imx662" / filename
    if nested.is_file():
        return nested
    hits = list(cache_root.rglob(filename))
    return hits[0] if hits else None


def sort_cache_into_bursts(
    cache_root: Path | str,
    bursts_root: Path | str,
    manifest: dict[str, dict[str, list[str]]],
    *,
    copy: bool = True,
) -> dict[str, dict[str, int]]:
    """Copy/sync manifest-listed DNGs into ``bursts/<scene>/<ag_tag>/``."""
    cache_root = Path(cache_root).expanduser()
    bursts_root = Path(bursts_root).expanduser()
    counts: dict[str, dict[str, int]] = {}

    for scene, tags in manifest.items():
        counts[scene] = {}
        for ag_tag, filenames in tags.items():
            dest_dir = bursts_root / scene / ag_tag
            dest_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for fn in filenames:
                src = resolve_cache_file(cache_root, fn)
                if src is None:
                    continue
                dst = dest_dir / Path(fn).name
                if dst.exists():
                    moved += 1
                    continue
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(str(src), str(dst))
                moved += 1
            counts[scene][ag_tag] = moved
    return counts


def list_burst_dngs(burst_dir: Path) -> list[Path]:
    return sorted(p for p in burst_dir.glob("*.dng") if p.is_file())


def demosaic_mean(files: Sequence[Path | str], limit: int) -> np.ndarray:
    """Temporal-average raw Bayer over ``files[:limit]``, then demosaic once.

    Returns uint16 RGB (linear, no gamma) — pixel-compatible with
    ``nsa.raw_io._load_any``'s DNG path.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError(
            "demosaic_mean requires rawpy — pip install rawpy"
        ) from exc

    acc: np.ndarray | None = None
    n = 0
    black: float | None = None
    white: float | None = None
    for f in files[:limit]:
        with rawpy.imread(str(f)) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            if black is None:
                black = float(np.mean(raw.black_level_per_channel))
                white = float(raw.white_level)
        acc = bayer if acc is None else acc + bayer
        n += 1
    mean_bayer = acc / max(n, 1)
    norm = np.clip((mean_bayer - black) / max(white - black, 1.0), 0.0, 1.0)
    b16 = (norm * 65535.0).astype(np.uint16)
    return cv2.cvtColor(b16, cv2.COLOR_BAYER_RG2RGB)


def _update_gain_json(
    dest: Path,
    *,
    requested_gain: int,
    hcg: bool,
    source: str,
) -> None:
    gj = dest / "gain.json"
    existing: dict[str, Any] = {}
    if gj.is_file():
        try:
            existing = json.loads(gj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("requested_gain", requested_gain)
    if hcg:
        existing["hcg_enabled"] = True
    existing["source"] = source
    gj.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def convert_burst_to_pair(
    burst_dir: Path | str,
    dest_dir: Path | str,
    *,
    gain_num: int,
    sensor_prefix: str = HCG_SENSOR_PREFIX,
    gt_frames: int = DEFAULT_GT_FRAMES,
    noisy_pick: int = DEFAULT_HCG_NOISY_PICK,
    min_frames: int = DEFAULT_MIN_BURST_FRAMES,
    source: str = "recovered_from_pi_ctt_cache",
    hcg: bool = True,
) -> int | None:
    """Build ``noisy.dng`` + ``gt.tif`` in *dest_dir* from a burst folder."""
    burst_dir = Path(burst_dir)
    dest_dir = Path(dest_dir)
    files = list_burst_dngs(burst_dir)
    if len(files) < min_frames:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    noisy_src = files[min(noisy_pick, len(files) - 1)]
    shutil.copy2(noisy_src, dest_dir / "noisy.dng")

    rgb16 = demosaic_mean(files, min(gt_frames, len(files)))
    cv2.imwrite(str(dest_dir / "gt.tif"), cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))

    for stale in ("noisy.png", "gt.png"):
        stale_path = dest_dir / stale
        if stale_path.is_file():
            stale_path.unlink()

    _update_gain_json(dest_dir, requested_gain=gain_num, hcg=hcg, source=source)
    return len(files)


def build_hcg_pairs_from_bursts(
    bursts_root: Path | str,
    pi_raw_data: Path | str,
    manifest: dict[str, dict[str, list[str]]],
    *,
    gt_frames: int = DEFAULT_GT_FRAMES,
    noisy_pick: int = DEFAULT_HCG_NOISY_PICK,
    min_frames: int = DEFAULT_MIN_BURST_FRAMES,
    source: str = "recovered_from_pi_ctt_cache",
    scene_burst_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert sorted HCG bursts into ``imx662h_ag*_test`` PI_RAW folders."""
    bursts_root = Path(bursts_root)
    pi_raw_data = Path(pi_raw_data)
    burst_map = scene_burst_map or {}
    built: list[dict[str, Any]] = []

    for scene, tags in manifest.items():
        burst_scene = burst_map.get(scene, scene)
        for ag_tag in sorted(tags, key=lambda t: int(t.replace("ag", "") or "0")):
            gain_num = int(ag_tag.replace("ag", ""))
            burst_dir = bursts_root / burst_scene / ag_tag
            dest = pi_raw_data / scene / f"{HCG_SENSOR_PREFIX}_{ag_tag}_test"
            n = convert_burst_to_pair(
                burst_dir, dest,
                gain_num=gain_num,
                sensor_prefix=HCG_SENSOR_PREFIX,
                gt_frames=gt_frames,
                noisy_pick=noisy_pick,
                min_frames=min_frames,
                source=source,
                hcg=True,
            )
            if n:
                built.append({
                    "scene": scene,
                    "ag_tag": ag_tag,
                    "frames": n,
                    "dest": str(dest),
                    "burst_dir": str(burst_dir),
                })
    return built


def build_lcg_pairs_from_bursts(
    bursts_root: Path | str,
    pi_raw_data: Path | str,
    *,
    scenes: Iterable[str] | None = None,
    gains: Sequence[int] = LCG_GAIN_SWEEP,
    gt_frames: int = DEFAULT_GT_FRAMES,
    noisy_pick: int = DEFAULT_LCG_NOISY_PICK,
    min_frames: int = DEFAULT_MIN_BURST_FRAMES,
) -> list[dict[str, Any]]:
    """Convert LCG ``bursts/<scene>/ag<N>/burst_*.dng`` trees to imx662 pairs."""
    bursts_root = Path(bursts_root)
    pi_raw_data = Path(pi_raw_data)
    if scenes is None:
        scenes = sorted(
            d.name for d in bursts_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    built: list[dict[str, Any]] = []
    for scene in scenes:
        for gain in gains:
            ag_tag = f"ag{gain}"
            burst_dir = bursts_root / scene / ag_tag
            if not burst_dir.is_dir():
                continue
            dest = pi_raw_data / scene / f"{LCG_SENSOR_PREFIX}_{ag_tag}_test"
            n = convert_burst_to_pair(
                burst_dir, dest,
                gain_num=gain,
                sensor_prefix=LCG_SENSOR_PREFIX,
                gt_frames=gt_frames,
                noisy_pick=noisy_pick,
                min_frames=min_frames,
                source="burst_temporal_average",
                hcg=False,
            )
            if n:
                built.append({
                    "scene": scene,
                    "ag_tag": ag_tag,
                    "frames": n,
                    "dest": str(dest),
                    "burst_dir": str(burst_dir),
                })
    return built


def split_hcg_from_mixed_burst(
    bursts_root: Path | str,
    *,
    mixed_scene: str = "cabinet_H_2",
    hcg_scene: str = "cabinet_H_2_hcg",
    gains: Sequence[int] = HCG_GAIN_SWEEP,
    hcg_filename_prefix: str = HCG_FRAME_PREFIX,
) -> dict[str, int]:
    """Move Pi-cache HCG frames out of a shared LCG burst tree.

    ``cabinet_H_2`` is used for both LCG (``burst_*.dng``) and HCG
    (``imx662_5000k_*``) captures. This splits HCG frames into
    ``bursts/cabinet_H_2_hcg/ag<N>/`` so LCG and HCG pairs stay clean.
    """
    bursts_root = Path(bursts_root)
    moved_by_gain: dict[str, int] = {}

    for gain in gains:
        ag_tag = f"ag{gain}"
        mixed = bursts_root / mixed_scene / ag_tag
        clean_hcg = bursts_root / hcg_scene / ag_tag
        if not mixed.is_dir():
            continue
        clean_hcg.mkdir(parents=True, exist_ok=True)
        moved = 0
        for frame in list(mixed.glob(f"{hcg_filename_prefix}*.dng")):
            dst = clean_hcg / frame.name
            if dst.exists():
                frame.unlink()
            else:
                shutil.move(str(frame), str(dst))
            moved += 1
        moved_by_gain[ag_tag] = moved
    return moved_by_gain


def align_pi_cache(
    *,
    cache_root: Path | str | None = None,
    project_root: Path | str | None = None,
    project_json: Path | str | None = None,
    manifest_path: Path | str | None = None,
    manifest: dict[str, dict[str, list[str]]] | None = None,
    sort: bool = True,
    build_pairs: bool = True,
    fix_h2_contamination: bool = True,
    copy_cache_files: bool = True,
) -> AlignResult:
    """End-to-end: manifest → sort cache → (optional H2 split) → HCG pairs."""
    cache_root = Path(cache_root or default_pi_unique_cache()).expanduser()
    project_root = Path(project_root or default_project_root()).expanduser()
    _, pi_raw_root = resolve_layout(project_root)
    bursts_root = project_root / "bursts"
    pi_raw_data = pi_raw_root / "Data"

    result = AlignResult()

    if manifest is None and manifest_path and Path(manifest_path).is_file():
        manifest = read_manifest(manifest_path)
        result.manifest_path = Path(manifest_path)

    if manifest is None and (sort or project_json is not None):
        pj = Path(project_json) if project_json else project_json_in_cache(cache_root)
        if not pj.is_file():
            if sort:
                result.skipped.append(f"project.json missing: {pj}")
            elif build_pairs:
                result.skipped.append(
                    f"no manifest and project.json missing: {pj} — "
                    "pass --manifest or run --write-manifest first"
                )
        else:
            manifest = build_hcg_sort_manifest(pj)
            out_manifest = (
                Path(manifest_path)
                if manifest_path
                else project_root / "hcg_sort_manifest.json"
            )
            write_manifest(manifest, out_manifest)
            result.manifest_path = out_manifest

    if build_pairs and manifest is None:
        result.skipped.append("cannot build pairs without manifest")
        return result

    if sort:
        if not cache_root.is_dir():
            result.skipped.append(f"cache missing: {cache_root}")
        else:
            result.sorted_bursts = sort_cache_into_bursts(
                cache_root, bursts_root, manifest, copy=copy_cache_files,
            )

    if fix_h2_contamination and bursts_root.is_dir():
        result.h2_split = split_hcg_from_mixed_burst(bursts_root)

    if build_pairs:
        scene_burst_map = {}
        if fix_h2_contamination:
            scene_burst_map["cabinet_H_2"] = "cabinet_H_2_hcg"
        result.pairs_built = build_hcg_pairs_from_bursts(
            bursts_root,
            pi_raw_data,
            manifest,
            scene_burst_map=scene_burst_map or None,
            source="recovered_from_pi_ctt_cache",
        )

    return result


def cache_readiness(
    cache_root: Path | str,
    manifest: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """Report how many manifest files are present in the Pi cache (for progress)."""
    cache_root = Path(cache_root).expanduser()
    wanted = manifest_file_list(manifest)
    present = [fn for fn in wanted if resolve_cache_file(cache_root, fn) is not None]
    return {
        "cache_root": str(cache_root),
        "wanted_files": len(wanted),
        "present_files": len(present),
        "fraction": len(present) / max(len(wanted), 1),
        "missing_sample": [fn for fn in wanted if fn not in present][:12],
    }
