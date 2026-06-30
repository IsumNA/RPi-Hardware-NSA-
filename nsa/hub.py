"""Hugging Face Hub model sourcing for the NSA stack.

Implements the model-selection methodology around four steps:

  1. Filter by License  — only Apache-2.0 / MIT are returned (eliminate legal risk).
  2. Benchmark Small    — default to the small tier (1-8B) to establish a baseline.
  3. Test the Gap       — step up to mid / large tiers when accuracy demands it.
  4. Freeze the Weights — lock the exact commit SHA into a local manifest (and,
                          optionally, download a pinned snapshot) so production
                          never silently drifts to a new revision.

Only the Python standard library is required for search + freeze (it talks to the
public Hugging Face REST API). Downloading a pinned snapshot additionally needs
the optional ``huggingface_hub`` package.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

HF_API = "https://huggingface.co/api"
HF_WEB = "https://huggingface.co"
USER_AGENT = "nsa-optimization-stack/1.0"

LOCK_PATH = Path("outputs/hf_lock.json")
STORAGE_DIR = Path("models/frozen")

# Step 1 — only these licenses are ever allowed through.
ALLOWED_LICENSES = {"apache-2.0": "Apache-2.0", "mit": "MIT"}

# Parameter-count tiers (absolute counts). Drives "benchmark small" / "test the gap".
SIZE_TIERS = {
    "tiny":  (0,               1_000_000_000),
    "small": (1_000_000_000,   8_500_000_000),
    "mid":   (8_500_000_000,   20_000_000_000),
    "large": (20_000_000_000,  80_000_000_000),
    "xl":    (80_000_000_000,  float("inf")),
}
SIZE_LABEL = {"tiny": "<1B", "small": "1-8B", "mid": "8-20B",
              "large": "20-80B", "xl": ">80B"}
SIZE_ORDER = ["tiny", "small", "mid", "large", "xl"]


class HubError(RuntimeError):
    """Raised for any Hub network / parsing / resolution failure."""


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------
def _get(url: str, timeout: int = 25):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise HubError(f"Hugging Face returned HTTP {exc.code} "
                       f"({exc.reason}).") from exc
    except urllib.error.URLError as exc:
        raise HubError(f"could not reach Hugging Face ({exc.reason}). "
                       f"Check your internet connection.") from exc
    except Exception as exc:  # noqa: BLE001
        raise HubError(f"unexpected error talking to Hugging Face: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def param_tier(params: Optional[int]) -> Optional[str]:
    if not params:
        return None
    for name, (lo, hi) in SIZE_TIERS.items():
        if lo <= params < hi:
            return name
    return "xl"


def human_params(params: Optional[int]) -> str:
    if not params:
        return "—"
    if params >= 1_000_000_000:
        return f"{params / 1e9:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1e6:.0f}M"
    return str(params)


def _licenses_of(tags: Iterable[str]) -> list[str]:
    return [t.split(":", 1)[1] for t in tags if t.startswith("license:")]


# ---------------------------------------------------------------------------
# Step 4 plumbing — model details / commit resolution
# ---------------------------------------------------------------------------
def model_details(model_id: str, revision: str = "main") -> dict:
    """Resolve a model's commit SHA, parameter count, license and file list."""
    safe_id = urllib.parse.quote(model_id, safe="/")
    safe_rev = urllib.parse.quote(revision, safe="")
    try:
        d = _get(f"{HF_API}/models/{safe_id}/revision/{safe_rev}")
    except HubError:
        # Fall back to the default-branch endpoint.
        d = _get(f"{HF_API}/models/{safe_id}")

    card = d.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not lic:
        tag_lics = _licenses_of(d.get("tags", []) or [])
        lic = tag_lics[0] if tag_lics else "?"

    safet = d.get("safetensors") or {}
    params = safet.get("total")

    return {
        "id": d.get("id", model_id),
        "sha": d.get("sha"),
        "license": lic,
        "params": params,
        "tier": param_tier(params),
        "downloads": d.get("downloads", 0) or 0,
        "likes": d.get("likes", 0) or 0,
        "pipeline_tag": d.get("pipeline_tag"),
        "files": [s.get("rfilename") for s in (d.get("siblings") or [])],
    }


# ---------------------------------------------------------------------------
# Steps 1-3 — license-filtered search with size tiers
# ---------------------------------------------------------------------------
def search_models(
    query: str = "",
    licenses: Iterable[str] = ("apache-2.0", "mit"),
    task: str = "text-generation",
    sort: str = "downloads",
    limit: int = 10,
    size: str = "any",
    enrich: bool = True,
) -> list[dict]:
    """Search the Hub, enforcing the allowed-license whitelist (Step 1).

    When ``enrich`` is set, each candidate is resolved to its parameter count +
    commit SHA so size tiers (Steps 2-3) and freezing (Step 4) work directly.
    """
    licenses = [l for l in licenses if l in ALLOWED_LICENSES] or list(ALLOWED_LICENSES)
    want = {f"license:{l}" for l in licenses}

    params = {
        "sort": sort,
        "direction": "-1",
        "limit": str(min(max(limit * 5, limit), 100)),
        "full": "true",
    }
    if query:
        params["search"] = query
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{HF_API}/models?{qs}"
    if task:
        url += f"&filter={urllib.parse.quote(task)}"
    # Server-side license filter is reliable only for a single license (filters AND).
    if len(licenses) == 1:
        url += f"&filter={urllib.parse.quote('license:' + licenses[0])}"

    data = _get(url)
    out: list[dict] = []
    for m in data:
        tags = m.get("tags", []) or []
        if want and not (set(tags) & want):
            continue
        mlic = _licenses_of(tags)
        out.append({
            "id": m.get("id"),
            "downloads": m.get("downloads", 0) or 0,
            "likes": m.get("likes", 0) or 0,
            "license": mlic[0] if mlic else "?",
            "pipeline_tag": m.get("pipeline_tag"),
            "params": None,
            "sha": None,
            "tier": None,
        })

    if enrich:
        enriched: list[dict] = []
        for r in out:
            try:
                d = model_details(r["id"])
                r["params"] = d["params"]
                r["sha"] = d["sha"]
                r["tier"] = d["tier"]
                if d["license"] != "?":
                    r["license"] = d["license"]
            except HubError:
                pass
            if size != "any" and r["tier"] != size:
                continue
            enriched.append(r)
            if len(enriched) >= limit:
                break
        return enriched

    return out[:limit]


def next_tier(size: str) -> Optional[str]:
    """The tier to 'test the gap' against (Step 3)."""
    if size not in SIZE_ORDER:
        return None
    i = SIZE_ORDER.index(size)
    return SIZE_ORDER[i + 1] if i + 1 < len(SIZE_ORDER) else None


# ---------------------------------------------------------------------------
# Step 4 — freeze the weights
# ---------------------------------------------------------------------------
def load_lock(lock_path: Path = LOCK_PATH) -> list[dict]:
    try:
        return json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _write_lock(entries: list[dict], lock_path: Path = LOCK_PATH) -> None:
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _download_snapshot(model_id: str, sha: str, storage_dir: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise HubError("downloading a pinned snapshot needs the optional "
                       "'huggingface_hub' package (pip install huggingface_hub).") from exc
    dest = Path(storage_dir) / model_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=model_id, revision=sha, local_dir=str(dest))
    return str(path)


def freeze_model(
    model_id: str,
    revision: str = "main",
    lock_path: Path = LOCK_PATH,
    download: bool = False,
    storage_dir: Path = STORAGE_DIR,
) -> dict:
    """Resolve + lock a model's exact commit SHA into local secure storage.

    Refuses to freeze a model whose license isn't on the allowed whitelist.
    """
    d = model_details(model_id, revision)
    if d["license"] not in ALLOWED_LICENSES:
        raise HubError(f"refusing to freeze '{model_id}': license '{d['license']}' "
                       f"is not in the allowed set ({', '.join(ALLOWED_LICENSES)}).")
    if not d["sha"]:
        raise HubError(f"could not resolve a commit SHA for '{model_id}@{revision}'.")

    entry = {
        "id": d["id"],
        "revision": revision,
        "sha": d["sha"],
        "license": d["license"],
        "params": d["params"],
        "params_human": human_params(d["params"]),
        "tier": d["tier"],
        "pipeline_tag": d["pipeline_tag"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": f"{HF_WEB}/{d['id']}/tree/{d['sha']}",
        "n_files": len(d["files"]),
        "local_path": None,
    }
    if download:
        entry["local_path"] = _download_snapshot(d["id"], d["sha"], storage_dir)

    entries = [e for e in load_lock(lock_path) if e.get("id") != d["id"]]
    entries.append(entry)
    _write_lock(entries, lock_path)
    return entry
