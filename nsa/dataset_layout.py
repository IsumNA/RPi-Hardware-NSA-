"""IMX662 dataset layout: templates, audit, and scaffold.

Three cooperating areas on disk::

    <root>/
      calibration/imx662_gain256/   bias/ dark/ flat/level_XX/  → noise model
      clean_scenes/<scene>/         temporally-averaged GT stills → synthesis input
      PI_RAW/Data/<scene>/imx662_ag12_test/  noisy.png + gt.png → training pairs

Use :func:`audit_project` to see what is present vs missing, and
:func:`scaffold_imx662_project` to create the folder tree with capture guides.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nsa.noise_calib.io import discover_phase1_root, list_frames
from nsa.raw_io import IMAGE_EXTS, SUPPORTED_EXTS, find_paired_folders

# Default flat-field brightness levels (10–15 recommended for calibration).
DEFAULT_FLAT_LEVELS = 12

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"


@dataclass
class SlotSpec:
    """One required folder or capture role in the IMX662 workflow."""

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


def imx662_slot_specs(
    *,
    gain: int = 256,
    ag_tag: str = "ag12",
    flat_levels: int = DEFAULT_FLAT_LEVELS,
    scenes: tuple[str, ...] = ("cabinet_D50_100", "colour_strips", "study"),
) -> list[SlotSpec]:
    """Return the full checklist of folders/files expected for IMX662."""
    cal = f"calibration/imx662_gain{gain}"
    specs: list[SlotSpec] = [
        SlotSpec(
            slot_id="cal_root",
            rel_path=cal,
            title=f"Calibration root (gain {gain}×)",
            section="calibration",
            purpose="Parent folder for Phase-1 bias/dark/flat captures.",
            how_to_capture=(
                "On the Pi with IMX662: fix camera on a tripod. Use the SAME analogue "
                "gain and temperature for every frame in this folder (e.g. gain 256×). "
                "Shoot RAW/DNG when possible; PNG is accepted."
            ),
            min_count=1,
            count_label="folder",
        ),
        SlotSpec(
            slot_id="bias",
            rel_path=f"{cal}/bias",
            title="Bias frames",
            section="calibration",
            purpose="Read noise + ADC offset (lens capped, minimal exposure).",
            how_to_capture=(
                "1. Put the lens cap on (or cover the sensor).\n"
                "2. Set the shortest exposure / lowest analogue gain the driver allows.\n"
                "3. Capture at least 5 identical frames (10+ is better).\n"
                "4. Save as bias_00.dng, bias_01.dng, … in this folder."
            ),
            min_count=2,
            count_label="frames",
            example_files=["bias_00.dng", "bias_01.dng", "bias_02.dng"],
        ),
        SlotSpec(
            slot_id="dark",
            rel_path=f"{cal}/dark",
            title="Dark frames",
            section="calibration",
            purpose="Row-fixed-pattern noise (lens capped, normal exposure).",
            how_to_capture=(
                "1. Lens cap ON.\n"
                "2. Use the SAME analogue gain you will use for flat-field and scenes "
                f"(e.g. {gain}×).\n"
                "3. Normal exposure time (not minimal).\n"
                "4. Capture 3+ frames: dark_00.dng, dark_01.dng, …"
            ),
            min_count=1,
            count_label="frames",
            example_files=["dark_00.dng", "dark_01.dng"],
        ),
        SlotSpec(
            slot_id="flat_root",
            rel_path=f"{cal}/flat",
            title="Flat-field levels",
            section="calibration",
            purpose="Photon-transfer (shot noise) at many brightness levels.",
            how_to_capture=(
                "Point the capped-off lens at a uniform grey card or integrating sphere. "
                "For each brightness level, capture TWO frames (a + b) at the SAME light "
                "level — used to measure signal-dependent noise. Use 10–15 levels from "
                "dark grey to nearly clipping."
            ),
            min_count=2,
            count_label="levels",
        ),
    ]
    for i in range(1, flat_levels + 1):
        lv = f"{i:02d}"
        specs.append(SlotSpec(
            slot_id=f"flat_level_{lv}",
            rel_path=f"{cal}/flat/level_{lv}",
            title=f"Flat level {lv}",
            section="calibration",
            purpose=f"Uniform-light pair at brightness step {i}/{flat_levels}.",
            how_to_capture=(
                f"Level {lv}: adjust light so the card fills the frame evenly. "
                "Capture a.dng and b.dng (or a.png / b.png) back-to-back without "
                "changing exposure or gain."
            ),
            min_count=2,
            count_label="files (a+b pair)",
            example_files=["a.dng", "b.dng"],
        ))

    specs.append(SlotSpec(
        slot_id="clean_root",
        rel_path="clean_scenes",
        title="Clean ground-truth library",
        section="clean_scenes",
        purpose="Noise-free scene images used as GT for Phase-5 synthesis.",
        how_to_capture=(
            "Each scene subfolder holds temporally-averaged stills (see GT_CAPTURE.md). "
            "These are NOT noisy camera photos — they are the clean reference the "
            "simulator adds IMX662 noise onto."
        ),
        min_count=1,
        count_label="scene folder",
    ))
    for scene in scenes:
        specs.append(SlotSpec(
            slot_id=f"scene_{scene}",
            rel_path=f"clean_scenes/{scene}",
            title=f"Scene: {scene}",
            section="clean_scenes",
            purpose=f"Clean GT stills for the '{scene}' environment.",
            how_to_capture=(
                f"1. Mount IMX662 on a tripod; scene must be static.\n"
                f"2. Shoot a burst of 32–128 short-exposure RAW frames into "
                f"bursts/{scene}/<take>/.\n"
                f"3. Run: python capture_gt.py --burst bursts/{scene}/take01 "
                f"--output clean_scenes/{scene}/gt_01.png\n"
                "4. Repeat for more viewpoints (gt_02.png, …)."
            ),
            min_count=1,
            count_label="GT image",
            example_files=["gt_01.png", "gt_02.dng"],
        ))

    specs.append(SlotSpec(
        slot_id="burst_root",
        rel_path="bursts",
        title="RAW burst staging (optional)",
        section="clean_scenes",
        purpose="Temporary folder for multi-frame bursts before GT averaging.",
        how_to_capture=(
            "While capturing GT: save sequential RAW frames here (one subfolder per take). "
            "After averaging with capture_gt.py, the result moves to clean_scenes/."
        ),
        min_count=0,
        count_label="burst folders",
        optional=True,
    ))

    specs.append(SlotSpec(
        slot_id="pi_raw_root",
        rel_path="PI_RAW",
        title="PI_RAW training dataset",
        section="pi_raw",
        purpose="denoise-hw layout: noisy + gt pairs for compile / extended training.",
        how_to_capture=(
            "Usually GENERATED by the Noise Dataset Wizard or simulate_dataset.py — "
            "not shot manually. Each test folder needs noisy.png and gt.png."
        ),
        min_count=1,
        count_label="paired folder",
    ))
    for scene in scenes:
        specs.append(SlotSpec(
            slot_id=f"pair_{scene}",
            rel_path=f"PI_RAW/Data/{scene}/imx662_{ag_tag}_test",
            title=f"Training pair: {scene}",
            section="pi_raw",
            purpose=f"Synthesized (or real) noisy/gt pair for scene '{scene}'.",
            how_to_capture=(
                "Generated by Phase 5 synthesis from clean_scenes/ + calibration JSON. "
                "Must contain noisy.png and gt.png (or .dng)."
            ),
            min_count=2,
            count_label="files (noisy+gt)",
            example_files=["noisy.png", "gt.png"],
        ))
    return specs


@dataclass
class SlotStatus:
    slot_id: str
    rel_path: str
    title: str
    section: str
    status: str
    found: int
    required: int
    files: list[str]
    purpose: str
    how_to_capture: str
    optional: bool = False


def _count_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return list_frames(folder)


def _slot_status(spec: SlotSpec, root: Path) -> SlotStatus:
    path = root / spec.rel_path
    files: list[Path] = []
    found = 0

    if spec.slot_id.startswith("flat_level_"):
        if path.is_dir():
            files = _count_images(path)
            found = len(files)
        req = spec.min_count
    elif spec.slot_id == "flat_root":
        flat_root = path
        levels = 0
        if flat_root.is_dir():
            for d in sorted(flat_root.iterdir()):
                if d.is_dir() and len(_count_images(d)) >= 2:
                    levels += 1
            files = _count_images(flat_root)
        found = levels
        req = spec.min_count
    elif spec.slot_id == "cal_root":
        found = 1 if path.is_dir() else 0
        req = 1
    elif spec.slot_id in ("pi_raw_root", "clean_root", "burst_root"):
        if spec.slot_id == "pi_raw_root" and path.is_dir():
            pairs = find_paired_folders(str(path))
            found = len(pairs)
        elif path.is_dir():
            child_imgs = [
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                and p.stem.lower() not in ("noisy",)
            ]
            found = len({p.parent for p in child_imgs}) or (1 if child_imgs else 0)
        req = spec.min_count
        files = _count_images(path) if path.is_dir() else []
    elif spec.slot_id.startswith("pair_"):
        from nsa.raw_io import _pair_in_folder
        pr = _pair_in_folder(path) if path.is_dir() else None
        found = 2 if pr else 0
        req = spec.min_count
        if pr:
            files = [pr[0], pr[1]]
    elif spec.slot_id.startswith("scene_"):
        files = _count_images(path) if path.is_dir() else []
        # GT images: anything that isn't a README/marker
        files = [f for f in files if f.stem.lower() not in ("readme",)]
        found = len(files)
        req = spec.min_count
    else:
        files = _count_images(path) if path.is_dir() else []
        found = len(files)
        req = spec.min_count

    if spec.optional and found == 0:
        status = STATUS_MISSING
    elif found >= req:
        status = STATUS_COMPLETE
    elif found > 0:
        status = STATUS_PARTIAL
    else:
        status = STATUS_MISSING

    return SlotStatus(
        slot_id=spec.slot_id,
        rel_path=spec.rel_path,
        title=spec.title,
        section=spec.section,
        status=status,
        found=found,
        required=req,
        files=[str(f.relative_to(root)) if f.is_relative_to(root) else str(f)
               for f in files[:48]],
        purpose=spec.purpose,
        how_to_capture=spec.how_to_capture,
        optional=spec.optional,
    )


def audit_project(
    root: Path | str,
    *,
    gain: int = 256,
    ag_tag: str = "ag12",
    scenes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Scan ``root`` and return structured status for GUI / CLI."""
    root = Path(root).expanduser().resolve()
    scene_tuple = scenes or ("cabinet_D50_100", "colour_strips", "study")
    specs = imx662_slot_specs(gain=gain, ag_tag=ag_tag, scenes=scene_tuple)
    slots = [_slot_status(s, root) for s in specs]

    cal_path = root / f"calibration/imx662_gain{gain}"
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

    by_section: dict[str, list[dict]] = {}
    for sl in slots:
        by_section.setdefault(sl.section, []).append(asdict(sl))

    required = [s for s in slots if not s.optional]
    complete = sum(1 for s in required if s.status == STATUS_COMPLETE)
    partial = sum(1 for s in required if s.status == STATUS_PARTIAL)

    return {
        "root": str(root),
        "exists": root.is_dir(),
        "sensor": "imx662",
        "gain": gain,
        "ag_tag": ag_tag,
        "scenes": list(scene_tuple),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "complete": complete,
            "partial": partial,
            "missing": len(required) - complete - partial,
            "total_required": len(required),
            "percent": round(100.0 * complete / max(1, len(required)), 1),
        },
        "calibration_pipeline": cal_validation,
        "slots": [asdict(s) for s in slots],
        "by_section": by_section,
    }


def _write_capture_md(path: Path, spec: SlotSpec) -> None:
    path.mkdir(parents=True, exist_ok=True)
    md = path / "CAPTURE.md"
    if md.exists():
        return
    examples = "\n".join(f"  - `{n}`" for n in spec.example_files) or "  - (see parent README)"
    md.write_text(
        f"# {spec.title}\n\n"
        f"**Purpose:** {spec.purpose}\n\n"
        f"## How to capture\n\n{spec.how_to_capture}\n\n"
        f"## Expected files ({spec.min_count}+ {spec.count_label})\n\n{examples}\n\n"
        f"Supported formats: {', '.join(sorted(IMAGE_EXTS | {'.dng', '.raw'}))}\n",
        encoding="utf-8",
    )


def scaffold_imx662_project(
    root: Path | str,
    *,
    gain: int = 256,
    ag_tag: str = "ag12",
    scenes: tuple[str, ...] = ("cabinet_D50_100", "colour_strips", "study"),
    flat_levels: int = DEFAULT_FLAT_LEVELS,
    overwrite_readme: bool = False,
) -> Path:
    """Create the IMX662 folder tree with CAPTURE.md guides in every slot."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    specs = imx662_slot_specs(
        gain=gain, ag_tag=ag_tag, scenes=scenes, flat_levels=flat_levels,
    )

    readme = root / "README.md"
    if overwrite_readme or not readme.exists():
        readme.write_text(
            "# IMX662 dataset project\n\n"
            "This tree was scaffolded by NSA. Three stages:\n\n"
            "1. **calibration/** — bias, dark, flat captures → noise model JSON\n"
            "2. **clean_scenes/** — temporally-averaged ground-truth stills\n"
            "3. **PI_RAW/** — synthesized (or real) noisy+gt training pairs\n\n"
            "Open **Dataset Studio** in the NSA GUI to see what is missing.\n\n"
            "Quick GT from a burst:\n"
            "```bash\n"
            "python capture_gt.py --burst bursts/cabinet_D50_100/take01 \\\n"
            "    --output clean_scenes/cabinet_D50_100/gt_01.png\n"
            "```\n",
            encoding="utf-8",
        )

    gt_guide = root / "GT_CAPTURE.md"
    if overwrite_readme or not gt_guide.exists():
        gt_guide.write_text(_GT_CAPTURE_GUIDE, encoding="utf-8")

    for spec in specs:
        folder = root / spec.rel_path
        if spec.slot_id.startswith("pair_"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".placeholder").write_text(
                "noisy.png and gt.png will be written here by synthesis.\n",
                encoding="utf-8",
            )
        else:
            _write_capture_md(folder, spec)
            (folder / ".gitkeep").touch(exist_ok=True)

    # Burst staging example
    for scene in scenes:
        burst = root / "bursts" / scene / "take01"
        burst.mkdir(parents=True, exist_ok=True)
        hint = burst / "PUT_RAW_FRAMES_HERE.txt"
        if not hint.exists():
            hint.write_text(
                "Save 32–128 sequential RAW/DNG frames of a STATIC scene here,\n"
                "then run:\n"
                f"  python capture_gt.py --burst {burst.relative_to(root)} "
                f"--output clean_scenes/{scene}/gt_01.png\n",
                encoding="utf-8",
            )

    manifest = audit_project(root, gain=gain, ag_tag=ag_tag, scenes=scenes)
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return root


_GT_CAPTURE_GUIDE = """# Ground truth capture guide (IMX662)

The clean_scenes/ folder must contain **noise-free** reference images. The NSA
synthesizer adds IMX662 sensor noise on top of these — they are not ordinary photos.

## Recommended method: temporal averaging

1. **Tripod** — camera and scene must not move.
2. **Short exposure** — avoid motion blur and saturation; use analogue gain for brightness.
3. **Burst** — capture 32–128 RAW frames of the same scene (lens cap OFF).
4. **Average in linear light** — run `capture_gt.py` on the burst folder.

```bash
python capture_gt.py --burst bursts/cabinet_D50_100/take01 \\
    --output clean_scenes/cabinet_D50_100/gt_01.png --min-frames 16
```

The tool loads each frame to linear float, aligns if needed, and averages. Random
read noise averages toward zero; static scene detail remains → true ground truth.

## What NOT to use as GT

- A single noisy frame (even at low gain)
- A frame denoised by another network
- JPEG phone photos with compression artifacts
- Images that do not match your target IMX662 colour / FOV

## After GT capture

1. Calibrate noise: `calibrate_noise.py -i calibration/imx662_gain256`
2. Synthesize pairs: `simulate_dataset.py -i clean_scenes -o PI_RAW --calibration …`
3. Train: `run_demo.py --real --dataset PI_RAW --extended-train`
"""


def default_project_roots() -> list[Path]:
    """Candidate roots to probe (AI server, project, desktop)."""
    from nsa.denoise_hw_data import SYSTEM_PI_RAW, desktop_pi_raw_path

    here = Path(__file__).resolve().parents[1]
    candidates = [
        SYSTEM_PI_RAW.parent if SYSTEM_PI_RAW.exists() else None,
        SYSTEM_PI_RAW,
        here / "datasets" / "imx662_project",
        here / "datasets" / "PI_RAW",
        desktop_pi_raw_path().parent,
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        if c is None:
            continue
        try:
            p = c.expanduser().resolve()
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
        except OSError:
            continue
    return out


def find_best_project_root() -> Path | None:
    """Return the first existing root that looks like an IMX662 project."""
    for root in default_project_roots():
        if not root.is_dir():
            continue
        if (root / "calibration").is_dir() or (root / "clean_scenes").is_dir():
            return root
        if find_paired_folders(str(root)):
            return root
    return default_project_roots()[0] if default_project_roots() else None
