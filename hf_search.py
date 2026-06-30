#!/usr/bin/env python3
"""NSA Hugging Face Model Sourcing
=================================
Search the Hugging Face Hub for a model the legal/hardware way, following a
four-step methodology:

  1. Filter by License  — only Apache-2.0 / MIT are ever returned.
  2. Benchmark Small    — start in the small tier (1-8B) for a speed/cost baseline.
  3. Test the Gap       — bump to mid / large only if accuracy demands it.
  4. Freeze the Weights — lock the exact commit SHA into outputs/hf_lock.json so
                          production never silently drifts (optionally download a
                          pinned snapshot into models/frozen/).

Usage examples
--------------
  # Step 1-2: small, permissively-licensed text-generation models:
  python hf_search.py --query qwen --size small

  # Step 3: step up a tier to see if the accuracy jump is worth the compute:
  python hf_search.py --query qwen --size mid

  # MIT-only vision/image models:
  python hf_search.py --task image-to-image --license mit

  # Step 4: freeze a chosen model's exact revision into the lock file:
  python hf_search.py --freeze Qwen/Qwen3-4B

  # ...and also download the pinned snapshot to local secure storage:
  python hf_search.py --freeze Qwen/Qwen3-4B --download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nsa.hub import (
    ALLOWED_LICENSES, LOCK_PATH, SIZE_LABEL, SIZE_ORDER, HubError,
    freeze_model, human_params, load_lock, next_tier, search_models,
)
from nsa.theme import RPI_GREEN, RPI_RASPBERRY, banner

console = Console()

_GREEN = RPI_GREEN
_RED = RPI_RASPBERRY
_AMBER = "#E8A33D"
_MUTED = "#6B7A9A"
_BRIGHT = "#DDE3F0"

_TIER_COLOR = {"tiny": _MUTED, "small": _GREEN, "mid": _AMBER,
               "large": _RED, "xl": _RED}


def _human_count(n: int) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _results_table(rows: list[dict], title: str) -> Table:
    tbl = Table(title=title, title_style=f"bold {_BRIGHT}",
                border_style=_MUTED, header_style=f"bold {_MUTED}",
                show_lines=False, pad_edge=True)
    tbl.add_column("#", style=_MUTED, width=3, justify="right")
    tbl.add_column("Model", style=_BRIGHT, width=34, no_wrap=True)
    tbl.add_column("Params", width=7, justify="right")
    tbl.add_column("Tier", width=6)
    tbl.add_column("License", style=_GREEN, width=11)
    tbl.add_column("Down", style=_MUTED, width=6, justify="right")
    tbl.add_column("Likes", style=_MUTED, width=6, justify="right")
    for i, r in enumerate(rows, 1):
        tier = r.get("tier") or "—"
        tcol = _TIER_COLOR.get(tier, _MUTED)
        tbl.add_row(
            str(i),
            r.get("id", "?"),
            Text(human_params(r.get("params")), style="bold"),
            Text(tier, style=tcol),
            r.get("license", "?"),
            _human_count(r.get("downloads", 0)),
            _human_count(r.get("likes", 0)),
        )
    return tbl


def _do_freeze(args) -> int:
    console.print()
    console.print(f"  [{_MUTED}]Step 4 · Freezing the weights — resolving exact "
                  f"commit SHA for[/] [bold {_BRIGHT}]{args.freeze}[/]"
                  f"[{_MUTED}] @ {args.revision}…[/]")
    try:
        entry = freeze_model(args.freeze, revision=args.revision,
                             lock_path=Path(args.lock_file), download=args.download)
    except HubError as exc:
        console.print(f"\n[bold {_RED}]Freeze failed:[/] {exc}")
        return 1

    body = (
        f"  [bold {_GREEN}]{entry['id']}[/]\n\n"
        f"  Revision   [bold]{entry['revision']}[/]\n"
        f"  Commit SHA [bold {_BRIGHT}]{entry['sha']}[/]\n"
        f"  License    [bold {_GREEN}]{entry['license']}[/]\n"
        f"  Size       [bold]{entry['params_human']}[/]  "
        f"({entry['tier']} · {SIZE_LABEL.get(entry['tier'], '')})\n"
        f"  Files      {entry['n_files']} tracked in the repo\n"
        f"  Source     [{_MUTED}]{entry['source']}[/]\n"
    )
    if entry.get("local_path"):
        body += f"  Snapshot   [bold {_GREEN}]{entry['local_path']}[/]\n"
    else:
        body += (f"  Snapshot   [{_MUTED}]not downloaded — pass --download to pull "
                 f"a pinned copy[/]\n")
    body += (f"\n  [{_MUTED}]Locked into {args.lock_file}. Re-pull always uses this "
             f"SHA, so an upstream update can never break your pipeline.[/]")

    console.print()
    console.print(Panel(body, title=f"[bold {_GREEN}]WEIGHTS FROZEN[/]",
                        border_style=_GREEN, padding=(0, 2)))

    locked = load_lock(Path(args.lock_file))
    console.print(f"\n  [{_MUTED}]Lock file now tracks {len(locked)} frozen "
                  f"model(s).[/]")
    return 0


def _next_step_hint(args, rows: list[dict]) -> Panel:
    nt = next_tier(args.size) if args.size != "any" else None
    lines = [
        f"  [bold {_BRIGHT}]Methodology[/]  "
        f"[{_GREEN}]1 license[/] → [{_GREEN}]2 small[/] → "
        f"[{_AMBER}]3 test the gap[/] → [{_RED}]4 freeze[/]",
        "",
    ]
    if rows:
        top = rows[0]["id"]
        lines.append(f"  [{_MUTED}]Baseline pick (step 2):[/] [bold]{top}[/] "
                     f"[{_MUTED}]({human_params(rows[0].get('params'))})[/]")
        lines.append(f"  [{_MUTED}]Freeze it (step 4):[/] "
                     f"python hf_search.py --freeze {top}")
        if nt:
            lines.append(f"  [{_MUTED}]Test the gap (step 3):[/] "
                         f"python hf_search.py --query \"{args.query}\" --size {nt}")
    else:
        lines.append(f"  [{_AMBER}]No models matched. Loosen --size or --query, or "
                     f"try --license both.[/]")
    return Panel("\n".join(lines), border_style=_MUTED, padding=(0, 2),
                 title=f"[{_MUTED}]next step[/]", title_align="left")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hf_search.py",
        description="NSA Hugging Face model sourcing (license-safe, size-tiered, freezeable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--query", default="", help="free-text search (e.g. 'qwen', 'llama')")
    p.add_argument("--task", default="text-generation",
                   help="pipeline tag filter (e.g. text-generation, image-to-image, "
                        "image-classification). Use '' for any task.")
    p.add_argument("--license", choices=["apache-2.0", "mit", "both"], default="both",
                   help="Step 1 — allowed license(s) (default: both apache-2.0 + mit)")
    p.add_argument("--size", choices=SIZE_ORDER + ["any"], default="small",
                   help="Step 2/3 — parameter tier (default: small = 1-8B)")
    p.add_argument("--sort", choices=["downloads", "likes", "trendingScore"],
                   default="downloads", help="ranking (default: downloads)")
    p.add_argument("--limit", type=int, default=10, help="rows to show (default: 10)")
    p.add_argument("--no-enrich", dest="enrich", action="store_false",
                   help="skip per-model detail lookups (faster, no params/size tiers)")
    # Step 4
    p.add_argument("--freeze", metavar="MODEL_ID", default=None,
                   help="Step 4 — lock this model's exact commit SHA into the lock file")
    p.add_argument("--revision", default="main",
                   help="branch/tag/commit to freeze (default: main)")
    p.add_argument("--download", action="store_true",
                   help="also download a pinned snapshot (needs huggingface_hub)")
    p.add_argument("--lock-file", dest="lock_file", default=str(LOCK_PATH),
                   help=f"path to the freeze manifest (default: {LOCK_PATH})")
    p.add_argument("--list-locked", action="store_true",
                   help="print the models already frozen in the lock file and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    banner("Hugging Face Model Sourcing")

    if args.list_locked:
        locked = load_lock(Path(args.lock_file))
        if not locked:
            console.print(f"\n  [{_MUTED}]No frozen models yet in {args.lock_file}.[/]")
            return 0
        tbl = Table(title="Frozen models", title_style=f"bold {_BRIGHT}",
                    border_style=_MUTED, header_style=f"bold {_MUTED}")
        tbl.add_column("Model", style=_BRIGHT)
        tbl.add_column("SHA", style=_GREEN)
        tbl.add_column("License", style=_GREEN)
        tbl.add_column("Size")
        tbl.add_column("Frozen at", style=_MUTED)
        for e in locked:
            tbl.add_row(e.get("id", "?"), (e.get("sha") or "")[:12],
                        e.get("license", "?"), e.get("params_human", "—"),
                        e.get("frozen_at", "—"))
        console.print()
        console.print(tbl)
        return 0

    if args.freeze:
        return _do_freeze(args)

    licenses = (["apache-2.0", "mit"] if args.license == "both" else [args.license])

    console.print()
    console.print(f"  [bold {_BRIGHT}]Step 1 · License   :[/] "
                  f"{' + '.join(ALLOWED_LICENSES[l] for l in licenses)} only")
    console.print(f"  [bold {_BRIGHT}]Step 2/3 · Size    :[/] {args.size}"
                  + (f"  ({SIZE_LABEL.get(args.size, '')})" if args.size != "any" else ""))
    console.print(f"  [bold {_BRIGHT}]Task / query       :[/] "
                  f"{args.task or 'any'}  ·  '{args.query or '*'}'")
    console.print(f"  [{_MUTED}]Searching the Hugging Face Hub…[/]\n")

    try:
        rows = search_models(query=args.query, licenses=licenses, task=args.task,
                             sort=args.sort, limit=args.limit, size=args.size,
                             enrich=args.enrich)
    except HubError as exc:
        console.print(f"[bold {_RED}]Search failed:[/] {exc}")
        return 1

    console.print(_results_table(
        rows, f"{len(rows)} model(s) · license-safe · "
              f"{args.size if args.size != 'any' else 'all sizes'}"))
    console.print()
    console.print(_next_step_hint(args, rows))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print(f"\n[bold {_RED}]Aborted.[/]")
        sys.exit(130)
