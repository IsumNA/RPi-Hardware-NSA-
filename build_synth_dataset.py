#!/usr/bin/env python3
"""Build the clean-image cache for synthetic IMX662 pair generation.

Sources handled:

* **bursts** — every ``<bursts_root>/<scene>/<tag>/`` with DNGs gets averaged
  into a single packed-Bayer clean frame under the cache.  Defaults to the
  ag1 bursts (lowest-noise).
* **srgb**   — every image under ``<srgb_root>`` is unprocessed to a packed
  RGGB clean frame.  Point this at DIV2K / Flickr2K / your own JPEGs.

Both write ``.npy`` files (float16 by default) and are combined into a single
``clean_manifest.json`` under the cache root.  A training loop then wires that
manifest + the per-gain noise JSONs into ``SynthPairDataset`` and emits fresh
pairs on the fly.

Examples
--------
    # burst-only cache (fast, uses what's on disk today)
    .venv/bin/python build_synth_dataset.py --skip-srgb

    # add DIV2K after unzipping to datasets/DIV2K_train_HR
    .venv/bin/python build_synth_dataset.py \\
        --srgb-root datasets/DIV2K_train_HR --srgb-tile 1024

    # quick smoke test on 8 sRGBs
    .venv/bin/python build_synth_dataset.py \\
        --srgb-root datasets/DIV2K_train_HR --max-srgb 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.synth.sources import (
    build_burst_clean_cache,
    build_srgb_clean_cache,
    write_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bursts-root", type=Path,
                   default=ROOT / "datasets/imx662_project/bursts")
    p.add_argument("--srgb-root", type=Path,
                   default=ROOT / "datasets/DIV2K_train_HR")
    p.add_argument("--cache-root", type=Path,
                   default=ROOT / "datasets/synth")
    p.add_argument("--burst-tags", nargs="+", default=["ag1"],
                   help="which per-scene bursts to average (default: ag1)")
    p.add_argument("--burst-limit", type=int, default=48,
                   help="max frames averaged per burst (default: 48)")
    p.add_argument("--burst-mode", choices=("mean", "alpha_trim"),
                   default="alpha_trim")
    p.add_argument("--skip-bursts", action="store_true")
    p.add_argument("--skip-srgb", action="store_true")
    p.add_argument("--srgb-tile", type=int, default=1024,
                   help="crop sRGB to this size before unprocessing (0=full)")
    p.add_argument("--max-srgb", type=int, default=None,
                   help="cap number of sRGBs processed (useful for smoke tests)")
    p.add_argument("--dark-scale", type=float, default=0.35,
                   help="scale factor applied after inverse-WB (0.2–0.5 typical)")
    p.add_argument("--seed", type=int, default=662)
    args = p.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    if not args.skip_bursts:
        if not args.bursts_root.is_dir():
            print(f"! bursts_root missing: {args.bursts_root} — skipping",
                  flush=True)
        else:
            burst_cache = args.cache_root / "bursts"
            print(f"Bursts → {burst_cache}", flush=True)
            entries.extend(build_burst_clean_cache(
                args.bursts_root, burst_cache,
                limit=args.burst_limit, mode=args.burst_mode,
                tags=tuple(args.burst_tags),
            ))

    if not args.skip_srgb:
        if not args.srgb_root.is_dir():
            print(f"! srgb_root missing: {args.srgb_root} — skipping "
                  f"(pass --srgb-root or drop images there)", flush=True)
        else:
            srgb_cache = args.cache_root / "srgb"
            tile = args.srgb_tile or None
            print(f"sRGB   → {srgb_cache}  (tile={tile})", flush=True)
            entries.extend(build_srgb_clean_cache(
                args.srgb_root, srgb_cache,
                seed=args.seed, dark_scale=args.dark_scale,
                tile=tile, max_images=args.max_srgb,
            ))

    manifest = args.cache_root / "clean_manifest.json"
    write_manifest(manifest, entries)
    print(f"\n  wrote {manifest}  ({len(entries)} clean frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
