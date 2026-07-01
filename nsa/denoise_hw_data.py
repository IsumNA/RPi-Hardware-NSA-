"""denoise-hw / PI_RAW dataset helpers.

The training pipeline at https://github.com/davidplowman/denoise-hw expects paired
``noisy.dng`` + ``gt.dng`` folders under a ``PI_RAW/Data/…`` tree (not shipped in
that repo — usually at ``/opt/datasets/PI_RAW`` on a Pi). NSA auto-detects that
layout and falls back to ``datasets/PI_RAW`` in this project.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from nsa.raw_io import find_paired_folders, list_frames

# Same test folder denoise-hw's run.sh uses (test.py cabinet_D50_100/imx219_ag12_test).
DEFAULT_TEST_REL = "Data/cabinet_D50_100/imx219_ag12_test"
DEFAULT_FILTER = ["imx219", "ag12"]
DEFAULT_DATASET_PATH = "datasets/PI_RAW"
DEFAULT_REMOTE_PI_RAW = "/opt/datasets/PI_RAW"
SYNTHETIC_MARKER = ".nsa_synthetic_sample"


def desktop_pi_raw_path() -> Path:
    """``~/Desktop/PI_RAW`` (Windows or macOS/Linux)."""
    base = os.environ.get("USERPROFILE") or os.environ.get("HOME") or str(Path.home())
    return Path(base).expanduser() / "Desktop" / "PI_RAW"


def _pi_raw_search_paths() -> tuple[Path | None, ...]:
    return (
        Path("datasets/PI_RAW"),
        Path("../datasets/PI_RAW"),
        desktop_pi_raw_path(),
        Path("/opt/datasets/PI_RAW"),
        Path(os.environ.get("PI_RAW", "")) if os.environ.get("PI_RAW") else None,
    )


PI_RAW_SEARCH = _pi_raw_search_paths()


def resolve_pi_raw(explicit: str | Path | None = None,
                   project_root: Path | None = None) -> Path | None:
    """Return the first usable PI_RAW root (folder with paired captures)."""
    root = project_root or Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        candidates.append(p)
    for rel in PI_RAW_SEARCH:
        if rel is None:
            continue
        p = rel if rel.is_absolute() else (root / rel)
        candidates.append(p)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if find_paired_folders(str(candidate)):
            return candidate.resolve()
    return None


def _is_default_dataset_path(path: Path, project_root: Path) -> bool:
    norm = str(path).replace("\\", "/")
    if norm.rstrip("/") == DEFAULT_DATASET_PATH:
        return True
    try:
        return path.resolve() == (project_root / DEFAULT_DATASET_PATH).resolve()
    except OSError:
        return False


SYSTEM_PI_RAW = Path("/opt/datasets/PI_RAW")


def prefer_system_pi_raw() -> Path | None:
    """Real PI_RAW on the AI machine (/opt/datasets/PI_RAW) wins over bundled samples."""
    if SYSTEM_PI_RAW.exists() and find_paired_folders(str(SYSTEM_PI_RAW)):
        return SYSTEM_PI_RAW.resolve()
    return None


def prefer_desktop_pi_raw() -> Path | None:
    """``~/Desktop/PI_RAW`` from ``fetch_pi_raw.ps1`` wins over bundled samples."""
    desktop = desktop_pi_raw_path()
    if desktop.exists() and find_paired_folders(str(desktop)):
        if not is_synthetic_sample_dataset(desktop):
            return desktop.resolve()
    return None


def has_real_dng(root: Path | str) -> bool:
    return any(Path(root).rglob("*.dng"))


def is_synthetic_sample_dataset(root: Path | str | None) -> bool:
    """True for repo-bundled PNG placeholders (not real camera PI_RAW)."""
    if not root:
        return False
    p = Path(root)
    if not p.exists():
        return False
    if (p / SYNTHETIC_MARKER).exists():
        return True
    if has_real_dng(p):
        return False
    pairs = find_paired_folders(str(p))
    if not pairs:
        return False
    png_only = all(
        any(Path(folder).glob("noisy.png")) and not any(Path(folder).glob("noisy.dng"))
        for folder in pairs
    )
    return png_only and len(pairs) <= 8


def _apply_default_filter(cfg, root: Path) -> None:
    if not cfg.sensor.filter:
        test_dir = root / DEFAULT_TEST_REL
        if test_dir.is_dir():
            cfg.sensor.filter = list(DEFAULT_FILTER)


def finalize_dataset_config(cfg, project_root: Path | None = None) -> bool:
    """Pick the dataset root after YAML + CLI — never stomp a custom ``dataset_path``."""
    root_dir = project_root or Path(__file__).resolve().parents[1]
    if not cfg.sensor.real_capture:
        return False

    if cfg.sensor.dataset_path:
        cfg.sensor.dataset_path = str(
            resolve_data_path(cfg.sensor.dataset_path, root_dir) or cfg.sensor.dataset_path
        )

    explicit = Path(cfg.sensor.dataset_path) if cfg.sensor.dataset_path else None

    # Custom path from yaml or ``--dataset`` — use it if it has paired data.
    if explicit and explicit.exists() and not _is_default_dataset_path(explicit, root_dir):
        resolved = resolve_pi_raw(explicit, root_dir) or normalize_dataset_root(explicit)
        if find_paired_folders(str(resolved)):
            cfg.sensor.dataset_path = str(resolved)
            _apply_default_filter(cfg, Path(resolved))
            return True

    # Default / bundled path — prefer real PI_RAW (system, then Desktop fetch).
    default_local = (root_dir / DEFAULT_DATASET_PATH).resolve()
    if explicit is None or _is_default_dataset_path(explicit, root_dir):
        for prefer in (prefer_system_pi_raw, prefer_desktop_pi_raw):
            real_root = prefer()
            if real_root is not None:
                cfg.sensor.dataset_path = str(real_root)
                _apply_default_filter(cfg, real_root)
                return True

    if not find_paired_folders(str(default_local)):
        ensure_project_dataset(root_dir)
    resolved = resolve_pi_raw(default_local, root_dir) or default_local
    cfg.sensor.dataset_path = str(resolved)
    _apply_default_filter(cfg, Path(resolved))
    return True


def resolve_data_path(path: str | Path | None, project_root: Path | None = None) -> Path | None:
    """Anchor relative data paths to the project root."""
    if not path:
        return None
    root = project_root or Path(__file__).resolve().parents[1]
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


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
        return {"root": None, "paired_folders": 0, "samples": [], "kind": "missing"}
    pairs = find_paired_folders(str(root))
    frames = list_frames(str(root), limit=8)
    if is_synthetic_sample_dataset(root):
        kind = "synthetic_sample"
    elif has_real_dng(root):
        kind = "real_dng"
    else:
        kind = "real_png"
    return {
        "root": str(root),
        "paired_folders": len(pairs),
        "samples": [f["name"] for f in frames],
        "kind": kind,
    }


def dataset_quality_notice(root: Path | str | None) -> str | None:
    """Short user-facing warning when captures are not real camera RAW."""
    if not root:
        return None
    p = Path(root)
    if is_synthetic_sample_dataset(p):
        return ("Bundled synthetic PNG samples (demo only). Fetch real PI_RAW: "
                "fetch_pi_raw.ps1 or setup_denoise_hw_data.py --fetch user@host:/opt/datasets/PI_RAW")
    if not has_real_dng(p):
        return ("No DNG files in dataset — PNG previews only. "
                "Real PI_RAW DNGs give much better denoising.")
    return None


def apply_auto_dataset(cfg, project_root: Path | None = None) -> bool:
    """Backward-compatible alias — prefer ``finalize_dataset_config`` after CLI overrides."""
    return finalize_dataset_config(cfg, project_root)


def parse_remote_spec(spec: str) -> tuple[str, str]:
    """``user@host:/opt/datasets/PI_RAW`` -> (``user@host``, ``/opt/datasets/PI_RAW``)."""
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(
            f"Remote spec must look like user@host:/opt/datasets/PI_RAW (got {spec!r})"
        )
    host, path = spec.rsplit(":", 1)
    if not host or not path:
        raise ValueError(f"Invalid remote spec: {spec!r}")
    return host, path


def fetch_pi_raw(remote_spec: str, dest: Path, *, full: bool = False) -> Path:
    """Copy PI_RAW from a remote machine with ``scp -r``.

    By default only the canonical denoise-hw test folder is copied (small).
    Pass ``full=True`` to mirror the entire remote tree (can be very large).
    """
    host, remote_root = parse_remote_spec(remote_spec)
    remote_root = remote_root.rstrip("/")
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("scp") is None:
        raise RuntimeError("scp not found — install OpenSSH client")

    if full:
        log_cmd = ["scp", "-r", f"{host}:{remote_root}/.", str(dest)]
    else:
        remote_folder = f"{remote_root}/{DEFAULT_TEST_REL}"
        local_parent = dest / Path(DEFAULT_TEST_REL).parent
        local_parent.mkdir(parents=True, exist_ok=True)
        log_cmd = ["scp", "-r", f"{host}:{remote_folder}",
                   str(local_parent / Path(DEFAULT_TEST_REL).name)]

    subprocess.run(log_cmd, check=True)
    if not find_paired_folders(str(dest)):
        raise RuntimeError(f"Fetch finished but no paired folders under {dest}")
    return dest


def patch_config_dataset(cfg_path: Path, dataset_root: Path) -> None:
    """Point ``config.yaml`` at a PI_RAW root."""
    text = cfg_path.read_text(encoding="utf-8")
    ds = str(dataset_root).replace("\\", "/")
    text = re.sub(r"real_capture:\s*\w+", "real_capture: true", text)
    text = re.sub(r"dataset_path:\s*\S+", f"dataset_path: {ds}", text)
    if "filter:" in text and "imx219" not in text:
        text = re.sub(r"filter:\s*\[\]", "filter: [imx219, ag12]", text)
    cfg_path.write_text(text, encoding="utf-8")


def link_or_point_dataset(local: Path, src: Path, *, prefer_symlink: bool = True) -> Path:
    """Symlink ``local`` -> ``src`` when possible; otherwise just return ``src``."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory: {src}")
    if not find_paired_folders(str(src)):
        raise ValueError(f"No paired noisy/gt folders under {src}")
    if prefer_symlink and local.resolve() != src:
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() or local.is_symlink():
            if local.is_symlink():
                local.unlink()
            elif local.is_dir() and not any(local.iterdir()):
                local.rmdir()
            else:
                raise FileExistsError(f"Remove or rename {local} first")
        try:
            local.symlink_to(src, target_is_directory=True)
            return local.resolve()
        except OSError:
            pass
    return src


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


def build_sample_pi_raw(out_root: Path, seed: int = 219, force: bool = False) -> int:
    """Write denoise-hw-style paired PNG folders when no real PI_RAW is available."""
    import cv2
    import numpy as np

    from nsa.raw_io import _synth_noisy_gt, _synthetic_scene
    from nsa.sensors import get_sensor

    scenes = (
        ("Data/cabinet_D50_100/imx219_ag12_test", "imx219", 256),
        ("Data/cabinet_D50_100/imx662_ag12_test", "imx662", 256),
        ("Data/cabinet_H_2/imx219_ag1_test", "imx219", 256),
    )
    patch = 1024
    n = 0
    for rel, sensor_key, gain in scenes:
        folder = out_root / rel
        folder.mkdir(parents=True, exist_ok=True)
        noisy_p, gt_p = folder / "noisy.png", folder / "gt.png"
        if not force and noisy_p.exists() and gt_p.exists():
            n += 1
            continue
        sensor = get_sensor(sensor_key)
        clean = _synthetic_scene(patch, patch, seed + n)
        noisy, gt = _synth_noisy_gt(clean, gain, sensor, temporal_frames=128,
                                    seed=seed + n + 100)
        for arr, path in ((noisy, noisy_p), (gt, gt_p)):
            img8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
        n += 1
    (out_root / SYNTHETIC_MARKER).write_text(
        "Bundled synthetic PNG pairs for offline demo.\n"
        "Replace with real PI_RAW: setup_denoise_hw_data.py --fetch …\n",
        encoding="utf-8",
    )
    return n


def ensure_project_dataset(project_root: Path | None = None) -> Path:
    """Pick the best local PI_RAW root (system, desktop, or synthetic samples)."""
    root = project_root or Path(__file__).resolve().parents[1]
    local = root / "datasets" / "PI_RAW"
    for candidate in (Path("/opt/datasets/PI_RAW"), desktop_pi_raw_path()):
        if candidate.exists() and find_paired_folders(str(candidate)):
            if is_synthetic_sample_dataset(candidate):
                continue
            try:
                return link_or_point_dataset(local, candidate)
            except (FileExistsError, OSError):
                return candidate.resolve()
    if find_paired_folders(str(local)) and not is_synthetic_sample_dataset(local):
        return local.resolve()
    build_sample_pi_raw(local, force=not find_paired_folders(str(local)))
    return local.resolve()
