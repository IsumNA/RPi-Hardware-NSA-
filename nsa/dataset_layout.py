"""IMX662 dataset layout — matches the real denoise-hw PI_RAW tree on disk.

Your manager's captures already live here::

    PI_RAW/Data/
      cabinet_D50_100/
        imx219_ag2_test/   noisy.dng  noisy.png  gt.dng  gt.png
        imx662_ag12_test/  …
      cabinet_F11_25/ …
      cabinet_H_10/ …
      colour_stripes/ …

The **noise synthesis pipeline** (unchanged) adds a separate calibration tree and
writes new ``imx662_ag*_test`` folders — it does not replace the existing data.

    <project>/calibration/imx662_gain256/   bias/ dark/ flat/   → noise model JSON
    <project>/clean_scenes/<scene>/         GT for synthesis (from bursts OR copy gt.* from PI_RAW)
    PI_RAW/Data/<scene>/imx662_ag24_test/   synthesized noisy+gt pairs
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsa.noise_calib.io import discover_phase1_root, list_frames
from nsa.raw_io import IMAGE_EXTS, SUPPORTED_EXTS, _pair_in_folder, find_paired_folders

DEFAULT_FLAT_LEVELS = 12

# Scenes from the manager's PI_RAW dataset (exact folder names).
MANAGER_SCENES: tuple[str, ...] = (
    "cabinet_D50_100",
    "cabinet_F11_25",
    "cabinet_H_10",
    "colour_stripes",
)

# denoise-hw analogue-gain tags already shot for IMX219 / legacy work.
LEGACY_AG_TAGS: tuple[str, ...] = ("ag1", "ag2", "ag4", "ag8", "ag12")

# IMX662 is low-light — synthesize extra night-vision folders beyond ag12.
# Folder tag (ag24) is a denoise-hw path label; calibration JSON uses sensor gain 256/512.
IMX662_TARGET_AG_TAGS: tuple[str, ...] = ("ag12", "ag24", "ag48")

_TEST_FOLDER_RE = re.compile(
    r"^(?P<sensor>imx219|imx662|imxng)_(?P<ag>ag\d+)_test$", re.IGNORECASE,
)

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"


@dataclass
class TestFolderInfo:
    scene: str
    folder_name: str
    sensor: str
    ag_tag: str
    rel_path: str
    has_pair: bool
    files: dict[str, str]   # noisy_dng, noisy_png, gt_dng, gt_png → rel paths
    source: str = "manager"  # manager | synthesized


@dataclass
class SlotSpec:
    slot_id: str
    rel_path: str
    title: str
    section: str
    purpose: str
    how_to_capture: str
    min_count: int
    count_label: str
    example_files: list[str] = field(default_factory=list)
    optional: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def parse_test_folder(name: str) -> tuple[str, str] | None:
    m = _TEST_FOLDER_RE.match(name)
    if not m:
        return None
    return m.group("sensor").lower(), m.group("ag").lower()


def resolve_layout(root: Path | str) -> tuple[Path, Path]:
    """Return ``(project_root, pi_raw_root)``.

    Accepts either the PI_RAW root (…/PI_RAW) or a parent project folder that
    contains PI_RAW/ and calibration/.
    """
    root = Path(root).expanduser().resolve()
    if (root / "Data").is_dir() and root.name.upper() == "PI_RAW":
        return root.parent, root
    if (root / "PI_RAW" / "Data").is_dir():
        return root, root / "PI_RAW"
    if (root / "Data").is_dir():
        return root.parent, root
    return root, root / "PI_RAW"


def _files_in_test_folder(folder: Path, pi_raw_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not folder.is_dir():
        return out
    for f in folder.iterdir():
        if not f.is_file():
            continue
        stem, ext = f.stem.lower(), f.suffix.lower()
        if stem == "noisy" and ext in (".dng", ".raw", ".png", ".jpg", ".jpeg", ".tif"):
            key = "noisy_dng" if ext == ".dng" else f"noisy{ext.replace('.', '_')}"
            if ext == ".png":
                key = "noisy_png"
            out[key] = str(f.relative_to(pi_raw_root))
        elif stem in ("gt", "clean", "reference"):
            key = "gt_dng" if ext == ".dng" else f"gt{ext.replace('.', '_')}"
            if ext == ".png":
                key = "gt_png"
            out[key] = str(f.relative_to(pi_raw_root))
    return out


def scan_pi_raw(pi_raw_root: Path | str) -> dict[str, Any]:
    """Inventory the manager's denoise-hw tree under ``PI_RAW/Data``."""
    pi_raw_root = Path(pi_raw_root).expanduser().resolve()
    data_root = pi_raw_root / "Data" if (pi_raw_root / "Data").is_dir() else pi_raw_root

    scenes: list[dict[str, Any]] = []
    all_tests: list[TestFolderInfo] = []
    sensors_seen: set[str] = set()
    ag_tags_seen: set[str] = set()

    if data_root.is_dir():
        for scene_dir in sorted(d for d in data_root.iterdir() if d.is_dir()):
            scene_name = scene_dir.name
            tests: list[dict[str, Any]] = []
            for test_dir in sorted(d for d in scene_dir.iterdir() if d.is_dir()):
                parsed = parse_test_folder(test_dir.name)
                if parsed is None:
                    continue
                sensor, ag = parsed
                sensors_seen.add(sensor)
                ag_tags_seen.add(ag)
                files = _files_in_test_folder(test_dir, pi_raw_root)
                pr = _pair_in_folder(test_dir)
                info = TestFolderInfo(
                    scene=scene_name,
                    folder_name=test_dir.name,
                    sensor=sensor,
                    ag_tag=ag,
                    rel_path=str(test_dir.relative_to(pi_raw_root)),
                    has_pair=pr is not None,
                    files=files,
                    source="synthesized" if (test_dir / ".nsa_simulated").is_file() else "manager",
                )
                all_tests.append(info)
                tests.append(asdict(info))
            scenes.append({
                "name": scene_name,
                "path": str(scene_dir.relative_to(pi_raw_root)),
                "test_count": len(tests),
                "tests": tests,
            })

    paired = sum(1 for t in all_tests if t.has_pair)
    return {
        "pi_raw_root": str(pi_raw_root),
        "data_root": str(data_root),
        "scenes": scenes,
        "scene_names": [s["name"] for s in scenes],
        "total_test_folders": len(all_tests),
        "paired_folders": paired,
        "sensors": sorted(sensors_seen),
        "ag_tags": sorted(ag_tags_seen, key=lambda x: int(x.replace("ag", "") or "0")),
        "tests": [asdict(t) for t in all_tests],
    }


def export_clean_gt_from_pi_raw(
    pi_raw_root: Path | str,
    clean_root: Path | str,
    *,
    scenes: tuple[str, ...] | None = None,
    prefer_sensor: str = "imx219",
    prefer_ag: str = "ag12",
) -> list[dict[str, str]]:
    """Copy existing ``gt.*`` from PI_RAW into ``clean_scenes/<scene>/`` for synthesis.

    Uses the preferred sensor/ag test folder when present, else the first paired
    folder in the scene that has a gt file.
    """
    pi_raw_root = Path(pi_raw_root).expanduser().resolve()
    clean_root = Path(clean_root).expanduser().resolve()
    inv = scan_pi_raw(pi_raw_root)
    scene_names = scenes or tuple(s["name"] for s in inv["scenes"])
    written: list[dict[str, str]] = []

    for scene in scene_names:
        scene_tests = [
            t for t in inv["tests"]
            if t["scene"] == scene and t.get("has_pair")
        ]
        if not scene_tests:
            continue
        pick = None
        for t in scene_tests:
            if t["sensor"] == prefer_sensor and t["ag_tag"] == prefer_ag:
                pick = t
                break
        if pick is None:
            pick = scene_tests[0]
        src_folder = pi_raw_root / pick["rel_path"]
        gt_path = None
        for name in ("gt.dng", "gt.png", "gt.tif", "gt.jpg"):
            p = src_folder / name
            if p.is_file():
                gt_path = p
                break
        if gt_path is None:
            pr = _pair_in_folder(src_folder)
            if pr:
                gt_path = pr[1]
        if gt_path is None:
            continue
        out_dir = clean_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = gt_path.suffix.lower()
        out_path = out_dir / f"gt_from_{pick['folder_name']}{ext}"
        shutil.copy2(gt_path, out_path)
        written.append({
            "scene": scene,
            "from": str(gt_path.relative_to(pi_raw_root)),
            "to": str(out_path),
        })
    return written


def _calibration_specs(gain: int, flat_levels: int) -> list[SlotSpec]:
    cal = f"calibration/imx662_gain{gain}"
    specs: list[SlotSpec] = [
        SlotSpec(
            slot_id="cal_root", rel_path=cal,
            title=f"Noise calibration (gain {gain}×)",
            section="noise_pipeline",
            purpose="NEW captures for the 5-phase noise model — separate from PI_RAW scenes.",
            how_to_capture=(
                "Shoot on the Pi with IMX662 at fixed analogue gain (e.g. 256× for night). "
                "This folder is NOT inside PI_RAW/Data — it lives beside PI_RAW.\n\n"
                "Subfolders: bias/ (lens cap, min exposure), dark/ (lens cap, normal exposure), "
                "flat/level_XX/ (uniform light pairs a+b)."
            ),
            min_count=1, count_label="folder",
        ),
        SlotSpec(
            slot_id="bias", rel_path=f"{cal}/bias",
            title="bias/ — read noise",
            section="noise_pipeline",
            purpose="Lens capped, minimal exposure. Measures read noise + ADC offset.",
            how_to_capture=(
                "Lens cap ON · shortest exposure · 5+ frames · bias_00.dng …"
            ),
            min_count=2, count_label="frames",
            example_files=["bias_00.dng", "bias_01.dng"],
        ),
        SlotSpec(
            slot_id="dark", rel_path=f"{cal}/dark",
            title="dark/ — row noise",
            section="noise_pipeline",
            purpose="Lens capped, normal exposure at the IMX662 night gain.",
            how_to_capture=(
                f"Lens cap ON · analogue gain {gain}× · normal exposure · 3+ frames"
            ),
            min_count=1, count_label="frames",
            example_files=["dark_00.dng"],
        ),
        SlotSpec(
            slot_id="flat_root", rel_path=f"{cal}/flat",
            title="flat/ — shot noise curve",
            section="noise_pipeline",
            purpose="10–15 brightness levels; each level_XX/ holds a.dng + b.dng.",
            how_to_capture="Uniform grey card · two frames per level · same exposure/gain.",
            min_count=2, count_label="levels",
        ),
    ]
    for i in range(1, flat_levels + 1):
        lv = f"{i:02d}"
        specs.append(SlotSpec(
            slot_id=f"flat_{lv}", rel_path=f"{cal}/flat/level_{lv}",
            title=f"flat/level_{lv}/",
            section="noise_pipeline",
            purpose=f"Flat-field pair at brightness step {i}.",
            how_to_capture="a.dng and b.dng at the same light level.",
            min_count=2, count_label="files",
            example_files=["a.dng", "b.dng"],
        ))
    return specs


def _imx662_target_specs(
    scenes: tuple[str, ...],
    ag_tags: tuple[str, ...],
    pi_raw_root: Path,
) -> list[SlotSpec]:
    """Slots for synthesized IMX662 folders we still need under PI_RAW."""
    specs: list[SlotSpec] = []
    inv = scan_pi_raw(pi_raw_root) if pi_raw_root.is_dir() else {"tests": []}
    existing = {
        (t["scene"], t["sensor"], t["ag_tag"])
        for t in inv.get("tests", [])
    }
    for scene in scenes:
        for ag in ag_tags:
            folder = f"imx662_{ag}_test"
            rel = f"Data/{scene}/{folder}"
            present = (scene, "imx662", ag) in existing
            specs.append(SlotSpec(
                slot_id=f"imx662_{scene}_{ag}",
                rel_path=rel,
                title=f"{scene}/{folder}",
                section="imx662_targets",
                purpose=(
                    f"Synthesized night-vision pair for scene '{scene}' at tag {ag}. "
                    "Created by Noise Dataset Wizard / simulate_dataset.py — "
                    "NOT shot on camera."
                ),
                how_to_capture=(
                    "1. Calibrate noise model from calibration/ folder.\n"
                    "2. Put clean GT in clean_scenes/ (copy from existing PI_RAW gt.* "
                    "with 'USE EXISTING GT' in Studio, or temporal burst).\n"
                    f"3. Run synthesis → writes noisy.* + gt.* here.\n\n"
                    f"Note: '{ag}' is a denoise-hw folder tag (manager used "
                    f"{', '.join(LEGACY_AG_TAGS)} for IMX219). IMX662 low-light may "
                    "need ag24/ag48 folders even though calibration uses gain 256/512."
                ),
                min_count=1 if present else 2,
                count_label="pair",
                example_files=["noisy.dng", "gt.dng", "noisy.png", "gt.png"],
                optional=(ag == "ag12"),
                meta={"scene": scene, "ag_tag": ag, "on_disk": present},
            ))
    return specs


def _slot_status(spec: SlotSpec, project_root: Path, pi_raw_root: Path) -> dict[str, Any]:
    if spec.section == "imx662_targets":
        folder = pi_raw_root / spec.rel_path
    else:
        folder = project_root / spec.rel_path

    files: list[str] = []
    found = 0
    req = spec.min_count

    if spec.section == "imx662_targets":
        pr = _pair_in_folder(folder) if folder.is_dir() else None
        if folder.is_dir():
            files = _files_in_test_folder(folder, pi_raw_root)
            files = list(files.values())
        found = 1 if pr else (len(files) // 2 if files else 0)
        req = 1
    elif spec.slot_id.startswith("flat_"):
        files_p = list_frames(folder) if folder.is_dir() else []
        found = len(files_p)
        files = [str(f.relative_to(project_root)) for f in files_p]
    elif spec.slot_id == "flat_root":
        levels = 0
        if folder.is_dir():
            for d in sorted(folder.iterdir()):
                if d.is_dir() and len(list_frames(d)) >= 2:
                    levels += 1
        found = levels
    elif spec.slot_id == "cal_root":
        found = 1 if folder.is_dir() else 0
    else:
        files_p = list_frames(folder) if folder.is_dir() else []
        found = len(files_p)
        files = [str(f.relative_to(project_root)) for f in files_p]

    if spec.optional and not folder.is_dir():
        status = STATUS_MISSING
    elif spec.section == "imx662_targets":
        status = STATUS_COMPLETE if folder.is_dir() and _pair_in_folder(folder) else STATUS_MISSING
    elif found >= req:
        status = STATUS_COMPLETE
    elif found > 0:
        status = STATUS_PARTIAL
    else:
        status = STATUS_MISSING

    return {
        "slot_id": spec.slot_id,
        "rel_path": spec.rel_path,
        "title": spec.title,
        "section": spec.section,
        "status": status,
        "found": found,
        "required": req,
        "files": files[:48],
        "purpose": spec.purpose,
        "how_to_capture": spec.how_to_capture,
        "optional": spec.optional,
        "meta": spec.meta,
    }


def _existing_pi_raw_slots(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn scan results into tree slots for the GUI."""
    slots: list[dict[str, Any]] = []
    for scene in inventory.get("scenes", []):
        slots.append({
            "slot_id": f"scene_{scene['name']}",
            "rel_path": scene["path"],
            "title": scene["name"],
            "section": "on_disk",
            "status": STATUS_COMPLETE if scene["test_count"] else STATUS_MISSING,
            "found": scene["test_count"],
            "required": 1,
            "files": [],
            "purpose": (
                f"Manager scene folder — {scene['test_count']} sensor test folder(s) inside."
            ),
            "how_to_capture": (
                "Already captured by your team. Each subfolder is named "
                "`<sensor>_ag<N>_test` and holds noisy.* + gt.* (DNG and/or PNG).\n\n"
                f"Legacy IMX219 tags on disk: {', '.join(LEGACY_AG_TAGS)}."
            ),
            "optional": False,
            "meta": {"scene": scene["name"]},
            "children": scene.get("tests", []),
        })
    return slots


def audit_project(
    root: Path | str,
    *,
    gain: int = 256,
    imx662_ag_tags: tuple[str, ...] | None = None,
    scenes: tuple[str, ...] | None = None,
    flat_levels: int = DEFAULT_FLAT_LEVELS,
) -> dict[str, Any]:
    """Full audit: what's ON DISK (manager PI_RAW) + what the noise pipeline NEEDS."""
    project_root, pi_raw_root = resolve_layout(root)
    scene_tuple = scenes or MANAGER_SCENES
    ag_tuple = imx662_ag_tags or IMX662_TARGET_AG_TAGS

    inventory = scan_pi_raw(pi_raw_root) if pi_raw_root.is_dir() else {
        "scenes": [], "scene_names": [], "total_test_folders": 0,
        "paired_folders": 0, "sensors": [], "ag_tags": [], "tests": [],
        "pi_raw_root": str(pi_raw_root),
    }

    on_disk = _existing_pi_raw_slots(inventory)
    cal_specs = _calibration_specs(gain, flat_levels)
    target_specs = _imx662_target_specs(scene_tuple, ag_tuple, pi_raw_root)

    noise_slots = [_slot_status(s, project_root, pi_raw_root) for s in cal_specs]
    target_slots = [_slot_status(s, project_root, pi_raw_root) for s in target_specs]

    cal_path = project_root / f"calibration/imx662_gain{gain}"
    cal_validation: dict[str, Any] | None = None
    if cal_path.is_dir():
        try:
            discovered = discover_phase1_root(cal_path)
            cal_validation = {
                "bias": len(discovered["bias"]),
                "dark": len(discovered["dark"]),
                "flat_levels": len(discovered["flat_pairs"]),
                "ready": (
                    len(discovered["bias"]) >= 2
                    and len(discovered["dark"]) >= 1
                    and len(discovered["flat_pairs"]) >= 2
                ),
            }
        except Exception as exc:  # noqa: BLE001
            cal_validation = {"error": str(exc), "ready": False}

    imx662_on_disk = [
        t for t in inventory.get("tests", [])
        if t.get("sensor") == "imx662" and t.get("has_pair")
    ]
    imx662_missing = [
        s for s in target_slots
        if s["status"] != STATUS_COMPLETE and not s.get("optional")
    ]

    by_section = {
        "on_disk": on_disk,
        "noise_pipeline": noise_slots,
        "imx662_targets": target_slots,
    }

    return {
        "project_root": str(project_root),
        "pi_raw_root": str(pi_raw_root),
        "exists": project_root.is_dir() or pi_raw_root.is_dir(),
        "sensor": "imx662",
        "calibration_gain": gain,
        "imx662_ag_tags": list(ag_tuple),
        "legacy_ag_tags": list(LEGACY_AG_TAGS),
        "manager_scenes": list(scene_tuple),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "pi_raw_inventory": inventory,
        "summary": {
            "paired_on_disk": inventory.get("paired_folders", 0),
            "test_folders_on_disk": inventory.get("total_test_folders", 0),
            "scenes_on_disk": len(inventory.get("scenes", [])),
            "imx662_pairs_on_disk": len(imx662_on_disk),
            "imx662_targets_missing": len(imx662_missing),
            "calibration_ready": bool(cal_validation and cal_validation.get("ready")),
        },
        "calibration_pipeline": cal_validation,
        "slots": on_disk + noise_slots + target_slots,
        "by_section": by_section,
    }


# Backward-compatible alias for scaffold / old callers
def imx662_slot_specs(**kwargs: Any) -> list[SlotSpec]:
    gain = int(kwargs.get("gain", 256))
    scenes = kwargs.get("scenes", MANAGER_SCENES)
    ag_tags = kwargs.get("imx662_ag_tags", IMX662_TARGET_AG_TAGS)
    flat_levels = int(kwargs.get("flat_levels", DEFAULT_FLAT_LEVELS))
    return (
        _calibration_specs(gain, flat_levels)
        + _imx662_target_specs(tuple(scenes), tuple(ag_tags), Path("."))
    )


def scaffold_imx662_project(
    root: Path | str,
    *,
    gain: int = 256,
    imx662_ag_tags: tuple[str, ...] = IMX662_TARGET_AG_TAGS,
    scenes: tuple[str, ...] = MANAGER_SCENES,
    flat_levels: int = DEFAULT_FLAT_LEVELS,
    overwrite_readme: bool = False,
) -> Path:
    """Create calibration/ + clean_scenes/ beside an existing or new PI_RAW tree."""
    project_root, pi_raw_root = resolve_layout(root)
    project_root.mkdir(parents=True, exist_ok=True)
    (pi_raw_root / "Data").mkdir(parents=True, exist_ok=True)

    readme = project_root / "README.md"
    if overwrite_readme or not readme.exists():
        readme.write_text(_PROJECT_README.format(
            scenes=", ".join(scenes),
            ag_tags=", ".join(imx662_ag_tags),
            legacy_ag=", ".join(LEGACY_AG_TAGS),
        ), encoding="utf-8")

    gt_guide = project_root / "GT_CAPTURE.md"
    if overwrite_readme or not gt_guide.exists():
        gt_guide.write_text(_GT_CAPTURE_GUIDE, encoding="utf-8")

    for spec in _calibration_specs(gain, flat_levels):
        folder = project_root / spec.rel_path
        folder.mkdir(parents=True, exist_ok=True)
        md = folder / "CAPTURE.md"
        if not md.exists():
            md.write_text(
                f"# {spec.title}\n\n{spec.purpose}\n\n{spec.how_to_capture}\n",
                encoding="utf-8",
            )

    for scene in scenes:
        (project_root / "clean_scenes" / scene).mkdir(parents=True, exist_ok=True)
        (project_root / "bursts" / scene / "take01").mkdir(parents=True, exist_ok=True)

    manifest = audit_project(project_root, gain=gain, imx662_ag_tags=imx662_ag_tags,
                             scenes=scenes)
    (project_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return project_root


_PROJECT_README = """# IMX662 noise synthesis project

## What is already on disk (manager dataset)

Your team's real captures live in **PI_RAW/Data/** — do not delete or move them::

    PI_RAW/Data/
      cabinet_D50_100/   imx219_ag1_test … imx219_ag12_test  (noisy.* + gt.*)
      cabinet_F11_25/
      cabinet_H_10/
      colour_stripes/

Each test folder contains up to four files: ``noisy.dng``, ``noisy.png``, ``gt.dng``, ``gt.png``.

## What you add for the noise pipeline

| Folder | Purpose |
|--------|---------|
| ``calibration/imx662_gain256/`` | NEW bias/dark/flat shoots → noise model JSON |
| ``clean_scenes/<scene>/`` | Clean GT for synthesis (copy from PI_RAW gt.* or burst average) |
| ``PI_RAW/Data/<scene>/imx662_ag24_test/`` | **Generated** pairs (night-vision tags: {ag_tags}) |

Legacy IMX219 tags: {legacy_ag}. IMX662 low-light likely needs higher tags ({ag_tags}).

Open **Dataset Studio** in the NSA GUI and point **PI_RAW root** at this folder.

Scenes: {scenes}
"""

_GT_CAPTURE_GUIDE = """# Ground truth for IMX662 synthesis

## Option A — reuse manager GT (fastest)

In Dataset Studio click **USE EXISTING GT** — copies ``gt.dng``/``gt.png`` from each
scene's best PI_RAW folder (e.g. imx219_ag12_test) into ``clean_scenes/<scene>/``.

## Option B — temporal burst (best quality)

1. Tripod, static scene, 32–128 RAW frames → ``bursts/<scene>/take01/``
2. ``python capture_gt.py --burst bursts/<scene>/take01 --output clean_scenes/<scene>/gt_01.png``

## Then synthesize

1. ``python calibrate_noise.py -i calibration/imx662_gain256``
2. ``python simulate_dataset.py -i clean_scenes -o PI_RAW --calibration models/noise/….json``
"""


def default_project_roots() -> list[Path]:
    from nsa.denoise_hw_data import SYSTEM_PI_RAW, desktop_pi_raw_path

    here = Path(__file__).resolve().parents[1]
    candidates = [
        SYSTEM_PI_RAW,
        SYSTEM_PI_RAW.parent if SYSTEM_PI_RAW.parent.exists() else None,
        here / "datasets" / "PI_RAW",
        desktop_pi_raw_path(),
        here / "datasets" / "imx662_project",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        if c is None:
            continue
        try:
            p = c.expanduser().resolve()
            if str(p) not in seen:
                seen.add(str(p))
                out.append(p)
        except OSError:
            continue
    return out


def find_best_project_root() -> Path | None:
    for root in default_project_roots():
        if not root.is_dir():
            continue
        _, pi = resolve_layout(root)
        if (pi / "Data").is_dir() or find_paired_folders(str(pi)):
            return pi if pi.name.upper() == "PI_RAW" else root
        if (root / "calibration").is_dir():
            return root
    return default_project_roots()[0] if default_project_roots() else None
