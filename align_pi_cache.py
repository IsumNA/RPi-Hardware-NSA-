#!/usr/bin/env python3
"""Align synced Pi CTT burst cache into HCG PI_RAW training pairs.

Reads ``project.json`` from the Pi sync tree (``Pi_Unique_Cache``), sorts DNGs
into per-gain burst folders, optionally splits ``cabinet_H_2`` LCG/HCG mix-up,
and writes ``imx662h_ag*_test/noisy.dng`` + ``gt.tif`` under the project PI_RAW
tree.

Examples
--------
  # Full pipeline (default paths):
  python align_pi_cache.py --all

  # Only build the sort manifest from project.json:
  python align_pi_cache.py --write-manifest

  # Check sync progress without copying anything:
  python align_pi_cache.py --write-manifest --readiness

  # Sort + pairs using an existing manifest:
  python align_pi_cache.py --sort --pairs --manifest datasets/imx662_project/hcg_sort_manifest.json

  # Rebuild pairs from bursts already on disk (no Pi cache needed):
  python align_pi_cache.py --pairs --no-sort --fix-h2

Environment
-----------
  PI_UNIQUE_CACHE   Pi sync destination (default /opt/datasets/PI_RAW/Pi_Unique_Cache)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.dataset_align import (  # noqa: E402
    align_pi_cache,
    build_hcg_sort_manifest,
    cache_readiness,
    default_pi_unique_cache,
    default_project_root,
    project_json_in_cache,
    read_manifest,
    write_manifest,
)
from nsa.theme import banner, console, kv_table, log


def _add_path_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cache", "-c",
        default=str(default_pi_unique_cache()),
        help="Pi_Unique_Cache root (synced imx662/ tree)",
    )
    p.add_argument(
        "--project-root", "-r",
        default=str(default_project_root()),
        help="imx662_project root (bursts/ + PI_RAW/Data output)",
    )
    p.add_argument(
        "--project-json",
        default="",
        help="CTT project.json (default: <cache>/project.json)",
    )
    p.add_argument(
        "--manifest", "-m",
        default="",
        help="hcg_sort_manifest.json (written or read)",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_path_args(p)
    p.add_argument("--write-manifest", action="store_true",
                   help="build hcg_sort_manifest.json from project.json")
    p.add_argument("--readiness", action="store_true",
                   help="report cache coverage for the manifest (no writes)")
    p.add_argument("--sort", action="store_true",
                   help="copy DNGs from cache into bursts/<scene>/<ag>/")
    p.add_argument("--pairs", action="store_true",
                   help="build noisy.dng + gt.tif PI_RAW pairs from bursts")
    p.add_argument("--fix-h2", action="store_true",
                   help="split imx662_5000k_* out of cabinet_H_2 LCG bursts")
    p.add_argument("--no-fix-h2", action="store_true",
                   help="skip cabinet_H_2 LCG/HCG split")
    p.add_argument("--all", action="store_true",
                   help="write-manifest + sort + fix-h2 + pairs")
    p.add_argument("--no-sort", action="store_true",
                   help="skip cache → bursts copy (bursts already populated)")
    args = p.parse_args()

    cache = Path(args.cache).expanduser()
    project_root = Path(args.project_root).expanduser()
    manifest_path = (
        Path(args.manifest).expanduser()
        if args.manifest
        else project_root / "hcg_sort_manifest.json"
    )
    project_json = (
        Path(args.project_json).expanduser()
        if args.project_json
        else project_json_in_cache(cache)
    )

    do_all = args.all
    write_manifest_flag = args.write_manifest or do_all
    sort_flag = (args.sort or do_all) and not args.no_sort
    pairs_flag = args.pairs or do_all
    fix_h2 = (args.fix_h2 or do_all) and not args.no_fix_h2

    if not any((write_manifest_flag, args.readiness, sort_flag, pairs_flag, fix_h2)):
        p.error("pick at least one of --write-manifest, --readiness, --sort, "
                "--pairs, --fix-h2, or --all")

    banner("Pi cache alignment  ·  HCG burst → PI_RAW pairs")

    manifest = None
    if manifest_path.is_file() and not write_manifest_flag:
        manifest = read_manifest(manifest_path)
        log(f"loaded manifest: {manifest_path}", "ok")
    elif write_manifest_flag:
        if not project_json.is_file():
            log(f"project.json not found: {project_json}", "err")
            log("Wait for Pi cache sync or pass --project-json", "warn")
            return 1
        manifest = build_hcg_sort_manifest(project_json)
        write_manifest(manifest, manifest_path)
        log(f"wrote manifest: {manifest_path}", "ok")
        scenes = ", ".join(
            f"{scene} ({sum(len(v) for v in tags.values())} frames)"
            for scene, tags in manifest.items()
        )
        console.print(f"  scenes: {scenes}")

    if args.readiness:
        if manifest is None:
            if not manifest_path.is_file():
                log("need --write-manifest or --manifest for --readiness", "err")
                return 1
            manifest = read_manifest(manifest_path)
        report = cache_readiness(cache, manifest)
        console.print()
        console.print(kv_table([
            ("cache", report["cache_root"]),
            ("wanted", str(report["wanted_files"])),
            ("present", str(report["present_files"])),
            ("coverage", f"{100 * report['fraction']:.1f}%"),
        ], title="Pi cache readiness"))
        if report["missing_sample"]:
            console.print("  missing (sample): " + ", ".join(report["missing_sample"][:6]))
        if report["fraction"] < 1.0:
            log("Sync still in progress — sort/pairs will skip absent files", "warn")

    if sort_flag or pairs_flag or fix_h2:
        manifest_for_align = manifest
        if manifest_for_align is None and manifest_path.is_file():
            manifest_for_align = read_manifest(manifest_path)
        result = align_pi_cache(
            cache_root=cache,
            project_root=project_root,
            project_json=project_json if project_json.is_file() else None,
            manifest_path=manifest_path,
            manifest=manifest_for_align,
            sort=sort_flag,
            build_pairs=pairs_flag,
            fix_h2_contamination=fix_h2,
        )
        if result.sorted_bursts:
            total = sum(sum(tags.values()) for tags in result.sorted_bursts.values())
            log(f"sorted {total} frames into bursts/", "ok")
        if result.h2_split:
            moved = sum(result.h2_split.values())
            if moved:
                log(f"H2 split: moved {moved} HCG frames → cabinet_H_2_hcg/", "ok")
        if result.pairs_built:
            log(f"built {len(result.pairs_built)} imx662h test folders", "ok")
            for item in result.pairs_built[:8]:
                console.print(
                    f"  {item['scene']}/{Path(item['dest']).name}: "
                    f"{item['frames']} frames"
                )
            if len(result.pairs_built) > 8:
                console.print(f"  … and {len(result.pairs_built) - 8} more")
        if result.skipped:
            for msg in result.skipped:
                log(msg, "warn")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
