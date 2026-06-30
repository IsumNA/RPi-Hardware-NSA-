"""Persistent run history for the NSA stack.

Every compile (and architecture sweep) is snapshotted into
``outputs/history/<timestamp>_<tag>/`` together with its summary + key artifacts,
and a compact one-line record is appended to ``outputs/history/index.jsonl``.

That way past results *and* the trained models are kept around — you can browse
previous runs, reuse a model on the live camera, or reload a configuration
without ever re-running the test.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

HISTORY_DIR = Path("outputs/history")
INDEX_NAME = "index.jsonl"

# Files copied verbatim from outputs/ into each run's snapshot folder.
_COPY_COMPILE = ("summary.json", "validation_panel.png", "model.pt",
                 "exported_model.onnx")


def _safe(text) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "-" for c in str(text))


def _index_path(history_dir: Path) -> Path:
    return Path(history_dir) / INDEX_NAME


def _append(rec: dict, history_dir: Path = HISTORY_DIR) -> None:
    Path(history_dir).mkdir(parents=True, exist_ok=True)
    with _index_path(history_dir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _profile(summary: dict) -> str:
    m = summary.get("model", {}) or {}
    return (f"{str(m.get('family', '')).upper()} "
            f"{m.get('base_channels', '')}ch x {m.get('block_depth', '')} · "
            f"{m.get('conv_type', '')} · {m.get('activation', '')} · "
            f"{summary.get('precision', '')}").strip()


def record_run(summary: dict, out_dir, history_dir: Path = HISTORY_DIR) -> dict:
    """Snapshot a finished compile + append it to the history index.

    Returns the stored record (also useful for logging).
    """
    out_dir = Path(out_dir)
    ts = time.localtime()
    stamp = time.strftime("%Y%m%d-%H%M%S", ts)
    m = summary.get("model", {}) or {}
    fam = m.get("family", "model")
    hw = summary.get("hardware", "")
    slug = f"{stamp}_{_safe(fam)}_{_safe(hw)}"
    run_dir = Path(history_dir) / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    for name in _COPY_COMPILE:
        src = out_dir / name
        if src.exists():
            try:
                shutil.copy2(src, run_dir / name)
                copied[name] = str((run_dir / name).resolve())
            except Exception:  # noqa: BLE001
                pass
    for src in out_dir.glob("hardware_ready.*"):       # extension varies by chip
        try:
            shutil.copy2(src, run_dir / src.name)
            copied[src.name] = str((run_dir / src.name).resolve())
        except Exception:  # noqa: BLE001
            pass
    zip_path = summary.get("package_zip")
    if zip_path and Path(zip_path).exists():
        try:
            dest = run_dir / Path(zip_path).name
            shutil.copy2(zip_path, dest)
            copied["package_zip"] = str(dest.resolve())
        except Exception:  # noqa: BLE001
            pass

    rec = {
        "id": slug,
        "kind": "compile",
        "time": time.strftime("%Y-%m-%d %H:%M:%S", ts),
        "epoch": time.mktime(ts),
        "family": fam,
        "profile": _profile(summary),
        "hardware": hw,
        "hardware_name": summary.get("hardware_name", hw),
        "sensor": summary.get("sensor", ""),
        "sensor_key": summary.get("sensor_key", ""),
        "gain": summary.get("gain"),
        "precision": summary.get("precision", ""),
        "params": m.get("params"),
        "psnr_in": summary.get("psnr_in"),
        "psnr_out": summary.get("psnr_out"),
        "psnr_gain": summary.get("psnr_gain"),
        "latency_ms": summary.get("latency_ms"),
        "fps": summary.get("fps"),
        "fitness": summary.get("fitness"),
        "grade": summary.get("grade"),
        "run_mode": summary.get("run_mode"),
        "dir": str(run_dir.resolve()),
        "summary": copied.get("summary.json"),
        "panel": copied.get("validation_panel.png"),
        "model_pt": copied.get("model.pt"),
        "package_zip": copied.get("package_zip"),
        "model": m,                                    # full config for reload
    }
    _append(rec, history_dir)
    return rec


def record_sweep(payload: dict, out_dir, history_dir: Path = HISTORY_DIR) -> dict:
    """Snapshot a finished architecture sweep + append it to the history index."""
    out_dir = Path(out_dir)
    ts = time.localtime()
    stamp = time.strftime("%Y%m%d-%H%M%S", ts)
    hw = payload.get("target", "")
    slug = f"{stamp}_sweep_{_safe(hw)}"
    run_dir = Path(history_dir) / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    src = out_dir / "pareto.json"
    if src.exists():
        try:
            shutil.copy2(src, run_dir / "pareto.json")
            copied["pareto"] = str((run_dir / "pareto.json").resolve())
        except Exception:  # noqa: BLE001
            pass

    win = payload.get("winner", {}) or {}
    profile = (f"{str(win.get('family', '')).upper()} "
               f"{win.get('base_channels', '')}ch x {win.get('block_depth', '')}")
    rec = {
        "id": slug,
        "kind": "sweep",
        "time": time.strftime("%Y-%m-%d %H:%M:%S", ts),
        "epoch": time.mktime(ts),
        "family": win.get("family", ""),
        "profile": profile.strip(),
        "hardware": hw,
        "hardware_name": payload.get("target_label", hw),
        "sensor": payload.get("sensor", ""),
        "sensor_key": payload.get("sensor", ""),
        "gain": payload.get("gain"),
        "precision": "",
        "params": win.get("params"),
        "psnr_out": win.get("psnr"),
        "latency_ms": win.get("latency_ms"),
        "fitness": win.get("fitness"),
        "grade": win.get("grade"),
        "n_evaluated": payload.get("n_evaluated"),
        "all_sensors": payload.get("all_sensors"),
        "dir": str(run_dir.resolve()),
        "pareto": copied.get("pareto"),
        "winner": win,
    }
    _append(rec, history_dir)
    return rec


def load_history(history_dir: Path = HISTORY_DIR, limit: int | None = None) -> list[dict]:
    """Return saved run records, newest first."""
    idx = _index_path(history_dir)
    rows: list[dict] = []
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    rows.sort(key=lambda r: r.get("epoch", 0), reverse=True)
    return rows[:limit] if limit else rows
