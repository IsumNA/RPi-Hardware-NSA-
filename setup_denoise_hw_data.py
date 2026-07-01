#!/usr/bin/env python3
"""Prepare denoise-hw test images for NSA.

The denoise-hw repo (https://github.com/davidplowman/denoise-hw) does not ship
RAW captures — it expects PI_RAW at /opt/datasets/PI_RAW on a Pi. This script:

  1. Clones denoise-hw into third_party/ (reference + tunings)
  2. Links, fetches, or builds datasets/PI_RAW (real DNGs or sample PNGs)
  3. Prints how to run NSA on the same test folders denoise-hw uses

Examples
--------
  # On the AI machine (data already at /opt/datasets/PI_RAW):
  python setup_denoise_hw_data.py --use /opt/datasets/PI_RAW --write-config

  # On Windows — copy the denoise-hw test scene to Desktop, point config there:
  python setup_denoise_hw_data.py --fetch you@ai-machine:/opt/datasets/PI_RAW --desktop --write-config

  # Symlink project datasets/PI_RAW to a local folder:
  python setup_denoise_hw_data.py --link /opt/datasets/PI_RAW --write-config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.denoise_hw_data import (DEFAULT_FILTER, DEFAULT_REMOTE_PI_RAW, DEFAULT_TEST_REL,
                                 clone_denoise_hw, dataset_summary, desktop_pi_raw_path,
                                 ensure_project_dataset, fetch_pi_raw, link_or_point_dataset,
                                 normalize_dataset_root, patch_config_dataset, resolve_pi_raw)
from nsa.theme import banner, console, kv_table, log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--link", metavar="PI_RAW",
                   help="symlink datasets/PI_RAW to this folder (e.g. /opt/datasets/PI_RAW)")
    p.add_argument("--use", metavar="PI_RAW",
                   help="use this PI_RAW root directly (no copy); best on the AI machine")
    p.add_argument("--fetch", metavar="REMOTE",
                   help=f"scp from REMOTE (e.g. user@host:{DEFAULT_REMOTE_PI_RAW})")
    p.add_argument("--desktop", action="store_true",
                   help="with --fetch: copy to ~/Desktop/PI_RAW instead of datasets/PI_RAW")
    p.add_argument("--full", action="store_true",
                   help="with --fetch: copy the entire remote tree (default: test scene only)")
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
    root: Path | None = None

    if args.fetch:
        dest = desktop_pi_raw_path() if args.desktop else local
        try:
            root = fetch_pi_raw(args.fetch, dest, full=args.full)
            label = "Desktop" if args.desktop else "project"
            log(f"Fetched PI_RAW -> {root} ({label})", "ok")
        except Exception as exc:
            log(f"Fetch failed: {exc}", "err")
            return 1
    elif args.use:
        use_src = Path(args.use).expanduser().resolve()
        try:
            root = link_or_point_dataset(local, use_src, prefer_symlink=False)
            log(f"Using PI_RAW at {root}", "ok")
        except Exception as exc:
            log(f"Cannot use {use_src}: {exc}", "err")
            return 1
    elif args.link:
        link_src = Path(args.link).expanduser().resolve()
        try:
            root = link_or_point_dataset(local, link_src)
            log(f"Linked {local} -> {link_src}", "ok")
        except Exception as exc:
            log(f"Link failed: {exc}", "err")
            return 1
    elif args.rebuild_samples:
        from nsa.denoise_hw_data import build_sample_pi_raw
        build_sample_pi_raw(local)
        root = local.resolve()
        log(f"Rebuilt sample pairs under {root}", "ok")
    else:
        root = ensure_project_dataset(ROOT)
        log(f"Dataset root: {root}", "ok")

    root = resolve_pi_raw(root) or normalize_dataset_root(root or local)
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
    ds = info["root"] or "datasets/PI_RAW"
    console.print(f"  python run_demo.py --real --dataset {ds} "
                  f"--filter {' '.join(DEFAULT_FILTER)} --sensor imx219")
    console.print(f"  python cache.py --dataset {ds} --filter imx219")
    console.print(f"  python search.py --real --dataset {ds} --hardware hailo8")

    if args.write_config and root:
        patch_config_dataset(ROOT / "config.yaml", root)
        log(f"Updated {ROOT / 'config.yaml'} -> dataset_path: {root}", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
