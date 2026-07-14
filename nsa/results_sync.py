"""Push result images to a laptop / workstation over SSH.

After a compile or a TEST IMAGE run, the validation panels + denoised outputs
can be auto-copied to a destination you can actually look at — e.g.
``you@laptop:~/nsa_results``. rsync when available (incremental), else scp.

Requires the destination to be SSH-reachable FROM this machine (sshd running,
routable address / VPN). Key-based auth is used first; a password (from the
``NSA_RESULTS_PASS`` env var or passed explicitly) is supported when ``sshpass``
is installed.

Configure once in config.yaml::

    output:
      results_dest: you@laptop:~/nsa_results

or ``--results-dest you@laptop:~/nsa_results`` on the CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# The result artifacts worth shipping (only those that exist are sent).
DEFAULT_RESULT_NAMES = (
    "validation_panel.png",
    "image_test.png",
    "live_preview.png",
    "resolution_tops_scaling.png",
    "summary.json",
    "exported_model.onnx",
)

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]


def _looks_remote(dest: str) -> bool:
    """True for ``user@host:path`` or ``host:path`` (not a bare local path)."""
    head = dest.split(":", 1)[0]
    return ":" in dest and "@" in head or (":" in dest and "/" not in head and head != "")


def collect_results(out_dir: Path, names=DEFAULT_RESULT_NAMES) -> list[Path]:
    out_dir = Path(out_dir)
    return [out_dir / n for n in names if (out_dir / n).is_file()]


def push_results(dest: str, out_dir="outputs", *, password: str | None = None,
                 names=DEFAULT_RESULT_NAMES) -> tuple[bool, str]:
    """Copy result files to ``dest``. Returns (ok, message).

    ``dest`` may be a local directory (simple copy) or an SSH target
    ``user@host:path``. Empty ``dest`` is a no-op success (feature disabled).
    """
    dest = (dest or "").strip()
    if not dest:
        return True, ""
    files = collect_results(Path(out_dir), names)
    if not files:
        return False, "no result files to send yet (run a compile / TEST IMAGE first)"

    password = password or os.environ.get("NSA_RESULTS_PASS") or None

    # Local destination — just copy (handy for a mounted share).
    if not _looks_remote(dest):
        try:
            d = Path(dest).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, d / f.name)
            return True, f"copied {len(files)} file(s) to {d}"
        except OSError as exc:
            return False, f"local copy failed: {exc}"

    prefix = ["sshpass", "-e"] if (password and shutil.which("sshpass")) else []
    env = os.environ.copy()
    if prefix:
        env["SSHPASS"] = password
    if password and not prefix:
        return False, ("a password was given but 'sshpass' is not installed — "
                       "install sshpass, or set up key-based SSH to the laptop")

    host, _, remote_path = dest.partition(":")
    remote_path = remote_path or "~/nsa_results"

    if shutil.which("rsync"):
        ssh_cmd = "ssh " + " ".join(_SSH_OPTS)
        cmd = prefix + ["rsync", "-az", "-e", ssh_cmd,
                        *[str(f) for f in files], f"{host}:{remote_path}/"]
    else:
        # scp fallback: ensure the remote dir exists, then copy.
        subprocess.run(prefix + ["ssh", *_SSH_OPTS, host, f"mkdir -p {remote_path}"],
                       env=env, capture_output=True, text=True, timeout=30, check=False)
        cmd = prefix + ["scp", *_SSH_OPTS, *[str(f) for f in files],
                        f"{host}:{remote_path}/"]

    # rsync doesn't create the remote dir itself; make it first.
    if shutil.which("rsync"):
        subprocess.run(prefix + ["ssh", *_SSH_OPTS, host, f"mkdir -p {remote_path}"],
                       env=env, capture_output=True, text=True, timeout=30, check=False)
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"transfer to {host} failed: {exc}"
    if r.returncode == 0:
        return True, f"sent {len(files)} result file(s) to {dest}"
    err = (r.stderr or r.stdout or "").strip().splitlines()
    low = " ".join(err).lower()
    if "permission denied" in low or "publickey" in low:
        hint = " — set up key auth to the laptop, or set NSA_RESULTS_PASS (+ install sshpass)"
    elif r.returncode == 255 or any(s in low for s in
                                    ("connection refused", "could not resolve",
                                     "no route", "timed out", "connect to host")):
        # 255 is always an SSH-layer failure (host unreachable / auth / no sshd).
        hint = (" — the laptop must be SSH-reachable FROM this machine: sshd "
                "running, routable address/VPN, and key auth or NSA_RESULTS_PASS")
    else:
        hint = ""
    return False, f"transfer failed (exit {r.returncode}): {err[-1] if err else '?'}{hint}"
