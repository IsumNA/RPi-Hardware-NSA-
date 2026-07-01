#!/usr/bin/env python3
"""Prepare denoise-hw test images for NSA.

The denoise-hw repo (https://github.com/davidplowman/denoise-hw) does not ship
RAW captures — it expects PI_RAW at /opt/datasets/PI_RAW on a Pi. This script:

  1. Clones denoise-hw into third_party/ (reference + tunings)
  2. Links or builds datasets/PI_RAW (sample paired PNGs, or symlink to Pi data)
  3. Prints how to run NSA on the same test folders denoise-hw uses

Examples
--------
  python setup_denoise_hw_data.py
  python setup_denoise_hw_data.py --link /opt/datasets/PI_RAW
  python setup_denoise_hw_data.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.denoise_hw_data import (DEFAULT_FILTER, DEFAULT_TEST_REL, apply_auto_dataset,
                                 clone_denoise_hw, dataset_summary, ensure_project_dataset,
                                 normalize_dataset_root, resolve_pi_raw)
from nsa.config import load_config
from nsa.theme import banner, console, kv_table, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--link", metavar="PI_RAW",
                   help="symlink datasets/PI_RAW to this folder (e.g. /opt/datasets/PI_RAW)")
    p.add_argument("--no-clone", action="store_true",
                   help="skip cloning denoise-hw into third_party/")
    p.add_argument("--rebuild-samples", action="store_true",
                   help="regenerate bundled sample PNG pairs")
    p.add_argument("--list", action="store_true", help="list paired folders and exit")
    p.add_argument("--write-config", action="store_true",
                   help="patch config.yaml to use real PI_RAW captures")
    args = p.parse_args()

    banner("denoise-hw dataset setup")

    if not args.no_clone:
        dest = clone_denoise_hw(ROOT / "third_party" / "denoise-hw")
        log(f"denoise-hw reference -> {dest}", "ok")

    local = ROOT / "datasets" / "PI_RAW"
    if args.link:
        link_src = Path(args.link).expanduser().resolve()
        if not link_src.is_dir():
            log(f"Not a directory: {link_src}", "err")
            return 1
        if local.exists() or local.is_symlink():
            if local.is_symlink():
                local.unlink()
            elif local.is_dir() and not any(local.iterdir()):
                local.rmdir()
            else:
                log(f"Remove or rename {local} first", "err")
                return 1
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(link_src, target_is_directory=True)
        log(f"Linked {local} -> {link_src}", "ok")
    elif args.rebuild_samples:
        from nsa.denoise_hw_data import build_sample_pi_raw
        build_sample_pi_raw(local)
        log(f"Rebuilt sample pairs under {local}", "ok")
    else:
        path = ensure_project_dataset(ROOT)
        log(f"Dataset root: {path}", "ok")

    root = resolve_pi_raw(local) or normalize_dataset_root(local)
    info = dataset_summary(root)
    console.print()
    console.print(kv_table([
        ("root", info["root"] or "(none)"),
        ("paired folders", str(info["paired_folders"])),
        ("sample names", ", ".join(info["samples"][:5]) or "—"),
    ], title="PI_RAW"))

    if args.list or info["paired_folders"]:
        from nsa.raw_io import find_paired_folders
        for folder in find_paired_folders(str(root))[:12]:
            rel = Path(folder).relative_to(root) if root else folder
            console.print(f"  [dim]·[/] {rel}")
        if info["paired_folders"] > 12:
            console.print(f"  [dim]… and {info['paired_folders'] - 12} more[/]")

    test = (root / DEFAULT_TEST_REL) if root else None
    console.print()
    if test and test.is_dir():
        log(f"Canonical denoise-hw test folder: {test.relative_to(root)}", "info")
        log(f"Filter tokens: {' '.join(DEFAULT_FILTER)}", "info")
    else:
        log("No cabinet_D50_100/imx219_ag12_test folder — using all paired scenes", "warn")

    console.print()
    console.print("[bold]Run NSA on denoise-hw images:[/]")
    console.print("  python run_demo.py --real --dataset datasets/PI_RAW "
                  f"--filter {' '.join(DEFAULT_FILTER)} --sensor imx219")
    console.print("  python cache.py --dataset datasets/PI_RAW --filter imx219")
    console.print("  python search.py --real --dataset datasets/PI_RAW --hardware hailo8")

    if args.write_config:
        _patch_config(root)
    return 0


def _patch_config(root: Path | None) -> None:
    import re
    cfg_path = ROOT / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    ds = str(root or ROOT / "datasets" / "PI_RAW").replace("\\", "/")
    text = re.sub(r"real_capture:\s*\w+", "real_capture: true", text)
    text = re.sub(r"dataset_path:\s*\S+", f"dataset_path: {ds}", text)
    if "filter:" in text and "imx219" not in text:
        text = re.sub(r"filter:\s*\[\]", "filter: [imx219, ag12]", text)
    cfg_path.write_text(text, encoding="utf-8")
    log(f"Updated {cfg_path} for real PI_RAW captures", "ok")


if __name__ == "__main__":
    raise SystemExit(main())
