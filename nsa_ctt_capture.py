#!/usr/bin/env python3
"""Guided CTT → NSA capture wizard.

Drives a Raspberry Pi **Camera Tuning Tool** (CTT) server over its HTTP API to
capture every RAW frame the NSA noise-synthesis pipeline expects, pulls the
DNGs back to this machine, and files them exactly where ``calibrate_noise.py``,
``capture_gt.py`` and ``simulate_dataset.py`` look for them.

It runs as a **station-by-station wizard**: for each station it prints how to
set up the rig, applies the camera controls over the API, shows a live readback
so you can confirm framing/exposure, waits for you to press Enter, fires the
burst, retrieves the DNGs, and drops them into the NSA layout. Change the rig,
press Enter, repeat — no clicking in the browser.

Topology
--------
    This laptop (NSA + wizard)  ──HTTP──►  Pi @ host:5000 (CTT + IMX662)
The DNGs are captured on the Pi; the wizard pulls them here via rsync/ssh
(incremental, recommended) or the CTT project ZIP archive (zero setup).

Example
-------
    # rsync transfer (incremental — recommended):
    python nsa_ctt_capture.py --host 10.3.195.212 \\
        --pi-ssh pi@10.3.195.212 --pi-workspace '~/ctt-server-workspace' \\
        --root datasets/imx662_project --run-post

    # zero-setup transfer via the CTT archive ZIP:
    python nsa_ctt_capture.py --host 10.3.195.212 --transfer archive \\
        --root datasets/imx662_project

    # just show the capture plan and exit:
    python nsa_ctt_capture.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.dataset_layout import (
    IMX662_TARGET_AG_TAGS,
    MANAGER_SCENES,
    resolve_layout,
    scaffold_imx662_project,
)
from nsa.denoise_hw_data import (SYSTEM_PI_RAW, _home_pi_raw, default_publish_dest,
                                  is_remote_publish_dest, normalize_publish_dest)
from nsa.theme import banner, console, kv_table, level_rule, log

# Capture is capped server-side at 16 frames per POST (see ctt-server capture()).
CTT_MAX_BURST = 16

# Per-attempt timeout for /api/health while polling a (re)starting server. Kept
# short so a hung-but-listening server (e.g. still holding a stale camera handle
# after a light-box power-cycle) fails fast and we get many retries inside the
# wait window — instead of one 30s read-timeout eating the whole budget.
HEALTH_POLL_TIMEOUT = 5.0

# Default light-box intensity (%) used when auto-lighting a scene. The light box
# is driven purely by intensity percentage now (no target-lux metering); tweak
# per-station with the GUI's % field / SET button or --lightbox-percent.
DEFAULT_LIGHTBOX_PERCENT = 100.0

# Real-pair capture sweeps the analogue gain across this series per scene: one
# noisy/gt pair per gain, filed into imx662_ag<gain>_test. The IMX662 analogue
# gain register spans 72 dB (up to ~3980×), so these are all reachable in
# hardware — but the libcamera tuning/mode may clamp a requested value, in which
# case the achieved gain is recorded alongside the pair.
DEFAULT_GAIN_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Temporal GT averaging: read noise in the average scales ~ gain/√N, so high gains
# need more frames for the same residual.  Override cap with NSA_GT_BURST_MAX.
GT_BURST_MAX_FRAMES = int(os.environ.get("NSA_GT_BURST_MAX", "512"))


def burst_frames_for_gain(gain: int, base: int, *,
                          max_frames: int | None = None) -> int:
    """Frames to capture for temporal GT at *gain* (base count is at 1×).

    Uses N ≈ base × gain (capped) so residual read noise stays similar across
    the sweep.  At 1× with base=48 → 48 frames; at 64× → 512 (cap); etc.
    """
    g = max(1, int(gain))
    base = max(4, int(base))
    cap = max_frames if max_frames is not None else GT_BURST_MAX_FRAMES
    scaled = int(round(base * g))
    return min(max(base, scaled), max(cap, base))


def parse_gain_sweep(spec: str | list | tuple | None) -> list[int]:
    """Parse a gain-sweep spec ("1,2,4,...,512" or a list) into sorted ints."""
    if spec is None or spec == "":
        return list(DEFAULT_GAIN_SWEEP)
    if isinstance(spec, (list, tuple)):
        raw = spec
    else:
        raw = str(spec).replace("×", "").replace("x", "").split(",")
    gains: list[int] = []
    for tok in raw:
        tok = str(tok).strip()
        if not tok:
            continue
        try:
            g = int(round(float(tok)))
        except ValueError as exc:
            raise ValueError(f"invalid gain in sweep: {tok!r}") from exc
        if g >= 1:
            gains.append(g)
    return sorted(dict.fromkeys(gains)) or list(DEFAULT_GAIN_SWEEP)

# lightSTUDIO-S illuminant names (from probe) — direct 1:1 matches.
LIGHTBOX_ILLUMINANTS = {
    "F12": "F12",  # F12 fluorescent
    "F11": "F11",  # F11 fluorescent
    "D50": "D50",  # D50 daylight
    "D65": "D65",  # D65 daylight
}


def _resolve_illuminant(illum_code: str, brightness: int | None) -> str:
    """Map a scene's illuminant code to a lightSTUDIO-S channel name.

    'H' (horizon/warm) has no single matching channel — the box instead
    exposes three separate halogen channels (Halogen10/100/400).  The
    brightness field from the scene name (e.g. ``cabinet_H_10`` → 10) picks
    the closest halogen channel.
    """
    if illum_code in LIGHTBOX_ILLUMINANTS:
        return LIGHTBOX_ILLUMINANTS[illum_code]
    if illum_code == "H":
        b = brightness if brightness is not None else 100
        if b <= 15:
            return "Halogen10"
        if b <= 200:
            return "Halogen100"
        return "Halogen400"
    return illum_code


def parse_scene_light(scene: str) -> tuple[str | None, int | None]:
    """Parse ``<label>_<illuminant>_<percent>`` scene names.

    Examples::

        cabinet_D50_100  → ('D50', 100)       # D50 at 100 %
        cabinet_F11_25   → ('F11', 25)        # F11 at 25 %
        cabinet_H_10     → ('Halogen10', 10)  # halogen channel + 10 %

    The third field is the light-box **intensity percent** (0–100), not a
    target-lux setpoint.  Returns ``(None, None)`` when the name does not match.
    """
    parts = scene.rsplit("_", 2)  # split from right to grab the last two parts
    if len(parts) != 3:
        return None, None
    illum_str, pct_str = parts[1], parts[2]
    try:
        pct = int(pct_str)
    except ValueError:
        return None, None
    pct = max(0, min(100, pct))
    return _resolve_illuminant(illum_str, pct), pct


def scene_lightbox_percent(scene: str, *, override: float | None = None) -> float:
    """Intensity % for ``set_lightbox`` — scene name, else CLI override, else default."""
    if override is not None:
        return max(0.0, min(100.0, float(override)))
    _, pct = parse_scene_light(scene)
    if pct is not None:
        return float(pct)
    return DEFAULT_LIGHTBOX_PERCENT


# --------------------------------------------------------------------------- #
#  CTT HTTP client
# --------------------------------------------------------------------------- #
class CTTError(RuntimeError):
    pass


class CTTClient:
    """Thin wrapper over the (unauthenticated) CTT server HTTP API."""

    def __init__(self, host: str, port: int = 5000, *, scheme: str = "https",
                 timeout: float = 30.0, verify: bool = False):
        self.base = f"{scheme}://{host}:{port}"
        self.timeout = timeout
        self.s = requests.Session()
        self.s.verify = verify  # the Pi's CTT server uses a self-signed cert
        # For the direct (no-zip) per-file transfer: any URL the capture record
        # hands back for a file, and the route template that last worked.
        self._file_urls: dict[str, str] = {}
        self._file_url_template: str | None = None
        if not verify:
            # Silence the per-request InsecureRequestWarning for the local Pi cert.
            try:
                from urllib3.exceptions import InsecureRequestWarning
                requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    # --- low-level -------------------------------------------------------- #
    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _get(self, path: str, *, timeout: float | None = None, **kw) -> requests.Response:
        try:
            return self.s.get(self._url(path), timeout=timeout or self.timeout, **kw)
        except requests.RequestException as exc:
            raise CTTError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, *, timeout: float | None = None, **kw) -> requests.Response:
        try:
            return self.s.post(self._url(path), timeout=timeout or self.timeout, **kw)
        except requests.RequestException as exc:
            raise CTTError(f"POST {path} failed: {exc}") from exc

    # --- camera ----------------------------------------------------------- #
    def health(self, *, timeout: float | None = None) -> dict:
        r = self._get("/api/health", timeout=timeout)
        try:
            return r.json()
        except ValueError:
            raise CTTError(f"Unexpected /api/health response ({r.status_code})")

    def get_controls(self) -> dict:
        r = self._get("/api/controls")
        if r.status_code != 200:
            raise CTTError(f"/api/controls returned {r.status_code}")
        return r.json()

    def set_controls(self, controls: dict) -> dict:
        r = self._post("/api/controls", json=controls)
        if r.status_code != 200:
            raise CTTError(f"set_controls returned {r.status_code}: {r.text[:200]}")
        return r.json()

    def histogram(self) -> dict:
        r = self._get("/api/histogram")
        return r.json() if r.status_code == 200 else {}

    def macbeth(self) -> dict:
        r = self._get("/api/macbeth")
        return r.json() if r.status_code == 200 else {}

    # --- lightbox --------------------------------------------------------- #
    def lightbox_status(self) -> dict | None:
        """Get lightbox state: {present, channel, illuminant, intensity}.
        Returns None if the lightbox API is unavailable."""
        try:
            r = self._get("/api/lightbox", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        return None

    def set_lightbox(self, illuminant: str, percent: float) -> dict | None:
        """Set lightbox to the named illuminant at the given intensity (0–100%).
        Returns the new state, or None if unavailable."""
        try:
            r = self._post("/api/lightbox",
                          json={"illuminant": illuminant, "percent": float(percent)},
                          timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        return None

    def _settled_lux(self, *, settle: float = 0.6, retries: int = 4) -> float:
        """Read measured lux, retrying briefly if it comes back 0 — a transient
        right after a light/exposure change while auto-exposure re-converges.
        A stray 0 would be logged as bogus capture metadata."""
        lux = 0.0
        for _ in range(retries):
            time.sleep(settle)
            lux = self.get_controls().get("lux", 0)
            if lux > 0:
                return lux
        return lux

    # --- projects --------------------------------------------------------- #
    def project_exists(self, name: str) -> bool:
        # The `project` route 404s when the project is missing, 200 otherwise.
        return self._get(f"/projects/{name}").status_code == 200

    def ensure_project(self, name: str) -> None:
        if self.project_exists(name):
            return
        r = self._post("/projects", data={"name": name})
        # A successful create redirects to the project page (followed → 200).
        if not self.project_exists(name):
            raise CTTError(
                f"Could not create project '{name}' (status {r.status_code}). "
                "Check the name is valid and unique."
            )

    def capture(self, name: str, *, image_type: str, frames: int,
                colour_temp: int | None = None, lux: int | None = None,
                label: str | None = None) -> list[dict]:
        """Capture a burst (looping over the 16-frame server cap). Returns the
        list of ``added`` capture records (each has a ``filename``)."""
        added: list[dict] = []
        remaining = max(1, frames)
        while remaining > 0:
            n = min(remaining, CTT_MAX_BURST)
            body: dict[str, Any] = {"image_type": image_type, "frames": n}
            if image_type != "dark":
                body["colour_temp"] = int(colour_temp if colour_temp is not None else 5000)
            if lux is not None:
                body["lux"] = int(lux)
            if label:
                body["label"] = label
            r = self._post(f"/projects/{name}/capture", json=body, timeout=max(self.timeout, 180))
            if r.status_code != 200:
                raise CTTError(f"capture failed ({r.status_code}): {r.text[:200]}")
            payload = r.json()
            new = payload.get("added", [])
            added.extend(new)
            # Remember any direct URL/path the server returns per file, so the
            # HTTP transfer can pull it straight to disk without guessing routes.
            for a in new:
                fn = a.get("filename")
                if not fn:
                    continue
                for key in ("url", "href", "download_url", "path", "rel_path"):
                    if a.get(key):
                        self._file_urls[fn] = str(a[key])
                        break
            remaining -= n
        return added

    # Likely per-file routes on the CTT server, tried in order (the first that
    # returns non-HTML 200 is cached for the rest of the session).
    _FILE_URL_TEMPLATES = (
        "/projects/{project}/{name}",
        "/projects/{project}/captures/{name}",
        "/projects/{project}/images/{name}",
        "/projects/{project}/raw/{name}",
        "/projects/{project}/files/{name}",
        "/projects/{project}/download/{name}",
    )

    def download_file(self, project: str, filename: str, dest_path: Path) -> Path:
        """Download ONE captured file directly over HTTP into ``dest_path`` — no
        ZIP, no SSH. Prefers a URL from the capture record, then a route that
        already worked, then probes the usual routes."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        attempts: list[tuple[bool, str]] = []
        if filename in self._file_urls:
            attempts.append((False, self._file_urls[filename]))
        ordered = ([self._file_url_template] if self._file_url_template else []) + [
            t for t in self._FILE_URL_TEMPLATES if t != self._file_url_template]
        attempts += [(True, t) for t in ordered]

        last = "no routes tried"
        for is_template, value in attempts:
            path = value.format(project=project, name=filename) if is_template else value
            url = path if path.startswith("http") else self._url(
                path if path.startswith("/") else "/" + path)
            try:
                r = self.s.get(url, stream=True, timeout=max(self.timeout, 120))
            except requests.RequestException as exc:
                last = f"{url}: {exc}"
                continue
            ctype = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and "html" not in ctype:
                tmp = dest_path.with_suffix(dest_path.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                if tmp.stat().st_size > 0:
                    tmp.replace(dest_path)
                    if is_template:
                        self._file_url_template = value
                    return dest_path
                tmp.unlink(missing_ok=True)
            last = f"{url} → {r.status_code} {ctype or '?'}"
        raise CTTError(
            f"could not download '{filename}' directly ({last}). "
            "The CTT server may use a non-standard file route — use "
            "--transfer archive or --transfer rsync instead.")

    def download_archive(self, name: str, dest_zip: Path) -> Path:
        r = self._get(f"/projects/{name}/archive", stream=True, timeout=max(self.timeout, 600))
        if r.status_code != 200:
            raise CTTError(f"archive download failed ({r.status_code})")
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with dest_zip.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        return dest_zip


# --------------------------------------------------------------------------- #
#  SSH — reach the Pi and auto-start the CTT server if it isn't running
# --------------------------------------------------------------------------- #
# accept-new trusts a new host key once (so a fresh Pi doesn't hard-fail);
# BatchMode fails fast instead of hanging on a password prompt (use key auth).
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

# Publish/rsync to the AI server — interactive (password OK). ControlMaster keeps
# one SSH session so you are only prompted once per host.
SSH_CONTROL_PATH = os.path.join(tempfile.gettempdir(), "nsa-ssh-%r@%h:%p")
SSH_OPTS_PUBLISH = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=30",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={SSH_CONTROL_PATH}",
    "-o", "ControlPersist=300",
]


def _ssh_cmd_publish(ssh: str, inner: str) -> list[str]:
    return ["ssh", *SSH_OPTS_PUBLISH, ssh, f"bash -c {shlex.quote(inner)}"]


def _ssh_publish_prefix(password: str | None) -> list[str]:
    """Prefix for ssh/rsync when a password is required (``sshpass`` if installed)."""
    if password and shutil.which("sshpass"):
        return ["sshpass", "-e"]
    return []


def _rsync_ssh_shell(password: str | None) -> str:
    inner = "ssh " + " ".join(SSH_OPTS_PUBLISH)
    prefix = _ssh_publish_prefix(password)
    return (" ".join(prefix) + " " + inner) if prefix else inner


@contextmanager
def ssh_publish_auth(password: str | None):
    """Environment + argv prefix for password-based publish SSH/rsync."""
    prefix = _ssh_publish_prefix(password)
    if not password:
        yield os.environ.copy(), prefix
        return
    env = os.environ.copy()
    if prefix:
        env["SSHPASS"] = password
    else:
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        try:
            safe = password.replace("'", "'\"'\"'")
            script.write(f"#!/bin/sh\nprintf '%s' '{safe}'\n")
            script.close()
            os.chmod(script.name, stat.S_IRWXU)
            env["SSH_ASKPASS"] = script.name
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env.setdefault("DISPLAY", ":0")
            yield env, prefix
        finally:
            try:
                os.unlink(script.name)
            except OSError:
                pass
        return
    yield env, prefix


@contextmanager
def ssh_askpass_env(password: str | None):
    """Back-compat wrapper — prefer :func:`ssh_publish_auth`."""
    with ssh_publish_auth(password) as (env, _prefix):
        yield env


def probe_ssh_key_auth(ssh_target: str) -> bool:
    """True when ``ssh_target`` accepts key-based login without a password."""
    if shutil.which("ssh") is None:
        return False
    try:
        res = subprocess.run(
            ["ssh", *SSH_OPTS, ssh_target, "true"],
            capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return res.returncode == 0


def _ssh_cmd(ssh: str, inner: str) -> list[str]:
    """An ssh argv that runs ``inner`` in a NON-login shell on the Pi.

    A login shell (``bash -lc``) would source the Pi's profile — which on
    Raspberry Pi OS prints a password-warning banner to stdout that pollutes
    command output. We use ``bash -c`` and rely on explicit paths / discovery
    instead of the login PATH.
    """
    return ["ssh", *SSH_OPTS, ssh, f"bash -c {shlex.quote(inner)}"]


def _ssh_hint(stderr: str, *, for_publish: bool = False) -> str | None:
    """Turn a raw ssh stderr line into an actionable hint, or None."""
    low = stderr.lower()
    if "no route to host" in low:
        target = "the AI server" if for_publish else "the Pi"
        return (f"Cannot reach {target} over the network (no route to host). "
                "Check it is powered on and on the same network.")
    if "connection timed out" in low or "operation timed out" in low:
        target = "AI server" if for_publish else "Pi"
        return (f"SSH to the {target} timed out. Check the hostname/IP and "
                "that nothing is blocking port 22.")
    if "connection refused" in low:
        if for_publish:
            return "SSH on the AI server refused the connection (port 22)."
        return ("SSH on the Pi refused the connection (port 22). "
                "Enable SSH on the Pi (raspi-config) or check sshd is running.")
    if "permission denied" in low:
        if for_publish:
            return ("SSH login to the AI server failed. Key login did not work — "
                    "enter your password when the wizard prompts you.")
        return ("SSH login failed (permission denied). Set up key-based login "
                "to the Pi, or check the username in the SSH target.")
    if "name or service not known" in low or "could not resolve hostname" in low:
        return "SSH hostname could not be resolved — check the SSH target."
    return None


def _probe_ssh(ssh: str) -> tuple[bool, str]:
    """Quick reachability check before blaming ctt-server paths."""
    res = subprocess.run(_ssh_cmd(ssh, "echo OK"), capture_output=True, text=True,
                         timeout=15)
    if res.returncode == 0 and "OK" in res.stdout:
        return True, ""
    err = (res.stderr or res.stdout or "unknown ssh error").strip()
    hint = _ssh_hint(err)
    return False, hint or f"SSH to {ssh} failed: {err}"


def _discover_ctt_server(ssh: str) -> str | None:
    """Locate the ctt-server executable on the Pi (usually a venv console script).

    Checks the common venv/user-bin spots, then falls back to a shallow find —
    so the user doesn't have to know it lives in e.g. ~/ctt-venv/bin.
    """
    probe = (
        "for p in ~/ctt-venv/bin/ctt-server ~/.local/bin/ctt-server "
        "~/venv/bin/ctt-server ~/env/bin/ctt-server ~/.venv/bin/ctt-server "
        "/usr/local/bin/ctt-server; do [ -x \"$p\" ] && { echo \"$p\"; exit 0; }; done; "
        "find ~ -maxdepth 4 -name ctt-server -type f 2>/dev/null | head -1"
    )
    res = subprocess.run(_ssh_cmd(ssh, probe), capture_output=True, text=True)
    # Guard against stray shell output: only accept an absolute path to ctt-server.
    for ln in res.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("/") and ln.endswith("ctt-server"):
            return ln
    return None


def ensure_server(client: "CTTClient", *, ssh: str | None, ctt_cmd: str = "ctt-server",
                  port: int = 5000, workspace: str | None = None,
                  autostart: bool = True, wait_s: float = 45.0,
                  status: Callable[[str], None] = lambda _m: None) -> str:
    """Make sure the CTT server is answering; SSH in and start it if not.

    Returns "running" (already up), "started" (we launched it), and raises
    CTTError with an actionable message otherwise.
    """
    try:
        client.health(timeout=HEALTH_POLL_TIMEOUT)
        return "running"
    except CTTError as exc:
        ctt_err = str(exc)  # not reachable yet — may try SSH auto-start below

    if not autostart:
        raise CTTError(
            f"CTT server at {client.base} is not responding (auto-start is off).\n"
            f"Last error: {ctt_err}\n"
            "Start ctt-server on the Pi manually, or re-enable auto-start.")

    if not ssh:
        raise CTTError(
            f"CTT server at {client.base} is not responding.\n"
            f"Last error: {ctt_err}\n"
            "Set the Pi SSH target (e.g. pi@10.3.195.212) so the wizard can "
            "auto-start ctt-server, or start it on the Pi yourself.")
    if shutil.which("ssh") is None:
        raise CTTError("ssh was not found on PATH on this machine.")

    ok, ssh_err = _probe_ssh(ssh)
    if not ok:
        raise CTTError(
            f"CTT server at {client.base} is not responding, and SSH could not "
            f"reach the Pi to start it.\n\n{ssh_err}")

    # If the given command isn't directly runnable (common when it's a venv
    # console script off the login PATH), auto-discover its real location.
    # Skipped for compound commands like 'source …/activate && ctt-server'.
    if all(tok not in ctt_cmd for tok in ("&&", ";", "|", " ")):
        # A path form (~/ctt-venv/bin/ctt-server) is checked for existence; a
        # bare name (ctt-server) is looked up on PATH. Keep a leading ~/ unquoted
        # so the Pi's shell still expands it (quoting the whole path would not).
        if "/" in ctt_cmd:
            target = ("~/" + shlex.quote(ctt_cmd[2:])) if ctt_cmd.startswith("~/") \
                else shlex.quote(ctt_cmd)
            probe = f"test -x {target}"
        else:
            probe = f"command -v {shlex.quote(ctt_cmd)}"
        chk = subprocess.run(_ssh_cmd(ssh, probe), capture_output=True, text=True)
        if chk.returncode != 0:
            discovered = _discover_ctt_server(ssh)
            if discovered:
                status(f"Located ctt-server at {discovered}")
                ctt_cmd = discovered
            else:
                raise CTTError(
                    f"'{ctt_cmd}' is not on the Pi and could not be found in the "
                    "usual venv locations. Give the full path (e.g. "
                    "~/ctt-venv/bin/ctt-server) or a command like "
                    "'source ~/ctt-venv/bin/activate && ctt-server'.")

    status(f"Stopping any stale ctt-server on {ssh} …")
    subprocess.run(_ssh_cmd(ssh, "pkill -f ctt-server || true"),
                   capture_output=True, text=True, timeout=15)
    time.sleep(2.0)

    status(f"Starting {ctt_cmd} on {ssh} …")
    full = f"{ctt_cmd} --host 0.0.0.0 --port {int(port)}"
    if workspace:
        full += f" --workspace {shlex.quote(workspace)}"
    # setsid+nohup+</dev/null fully detaches so the ssh call returns immediately
    # while the server keeps running; output goes to a log on the Pi.
    launch = f"setsid nohup {full} > ~/ctt-server.log 2>&1 < /dev/null & echo LAUNCHED"
    res = subprocess.run(_ssh_cmd(ssh, launch), capture_output=True, text=True, timeout=25)
    if res.returncode != 0 or "LAUNCHED" not in res.stdout:
        err = (res.stderr or res.stdout or "unknown error").strip()
        hint = _ssh_hint(err)
        raise CTTError("Could not start ctt-server over SSH"
                       + (f":\n{hint}" if hint else f": {err}"))

    deadline = time.time() + wait_s
    last = ""
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            client.health(timeout=HEALTH_POLL_TIMEOUT)
            status("ctt-server is up.")
            return "started"
        except CTTError as exc:
            last = str(exc)
            status(f"Waiting for ctt-server to come up… "
                   f"({max(0, int(deadline - time.time()))}s left)")
    raise CTTError(f"ctt-server did not respond within {int(wait_s)}s — check "
                   f"~/ctt-server.log on the Pi. Last error: {last}")


# --------------------------------------------------------------------------- #
#  Transfer backends — get the raw DNGs from the Pi onto this machine
# --------------------------------------------------------------------------- #
class Transfer:
    """Interface: make captured DNGs available locally, keyed by CTT filename."""

    # True  → fetch() pulls just this station's files (rsync); the wizard can save
    #         each cabinet as it's captured.
    # False → files are only available in bulk at finalize() (archive ZIP); the
    #         wizard defers filing to the end so it doesn't re-pull per cabinet.
    incremental: bool = False

    def fetch(self, filenames: list[str]) -> dict[str, Path]:
        """Return {ctt_filename: local_path} for the given DNG filenames.

        May return an empty/partial mapping if the backend defers to ``finalize``.
        """
        raise NotImplementedError

    def finalize(self) -> dict[str, Path]:
        """Final sync; return {ctt_filename: local_path} for everything pulled."""
        return {}


class RsyncTransfer(Transfer):
    """Incremental pull of the CTT project directory over SSH with rsync."""

    incremental = True

    def __init__(self, ssh: str, remote_workspace: str, project: str, mirror: Path):
        self.ssh = ssh
        self.remote = f"{remote_workspace.rstrip('/')}/{project}/"
        self.mirror = mirror
        self.mirror.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync") is None:
            raise CTTError("rsync not found on PATH — use --transfer archive instead.")

    def _sync(self) -> None:
        cmd = [
            "rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
            "--include=*/", "--include=*.dng", "--exclude=*",
            f"{self.ssh}:{self.remote}", f"{self.mirror}/",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise CTTError(f"rsync failed: {res.stderr.strip() or res.stdout.strip()}")

    def _index(self) -> dict[str, Path]:
        return {p.name: p for p in self.mirror.rglob("*.dng")}

    def fetch(self, filenames: list[str]) -> dict[str, Path]:
        self._sync()
        idx = self._index()
        return {f: idx[f] for f in filenames if f in idx}

    def finalize(self) -> dict[str, Path]:
        self._sync()
        return self._index()


def _extract_archive_dngs(client: "CTTClient", project: str, mirror: Path) -> dict[str, Path]:
    """Pull the project archive and extract loose DNGs into ``mirror``.

    The ZIP is a transient transport only — it's downloaded to a temp dir,
    the DNGs are unpacked as individual files, and the ZIP is discarded. No
    archive file is ever kept in the dataset.
    """
    mirror.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / f"{project}.zip"
        log("Pulling captured DNGs from CTT (project archive, extracted to "
            "loose files — no zip kept) …", "info")
        client.download_archive(project, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".dng"):
                    out = mirror / Path(member).name
                    with zf.open(member) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    return {p.name: p for p in mirror.rglob("*.dng")}


class HttpFileTransfer(Transfer):
    """Pull captured DNGs over the CTT HTTP API — no SSH, files land loose.

    The stock rpi-ctt server exposes raw DNGs only via the whole-project
    archive (there is no per-file raw route — ``/captures/<f>/jpeg`` serves a
    developed JPEG, not the DNG). So this backend tries a direct per-file
    download first (for custom servers that support it) and otherwise falls
    back to extracting the archive into loose files. Either way the result is
    individual DNGs in the folder — never a saved zip.
    """

    def __init__(self, client: "CTTClient", project: str, mirror: Path):
        self.client = client
        self.project = project
        self.mirror = mirror
        self.mirror.mkdir(parents=True, exist_ok=True)
        # None = not probed yet; True = per-file route works; False = fell back
        # to the archive (then behaves like ArchiveTransfer: bulk at finalize).
        self._direct: bool | None = None

    @property
    def incremental(self) -> bool:  # type: ignore[override]
        return self._direct is not False

    def fetch(self, filenames: list[str]) -> dict[str, Path]:
        if self._direct is False:
            return {}  # no per-file route — defer to a single archive pull
        out: dict[str, Path] = {}
        for fn in filenames:
            dst = self.mirror / fn
            if dst.exists() and dst.stat().st_size > 0:
                out[fn] = dst
                continue
            try:
                self.client.download_file(self.project, fn, dst)
                out[fn] = dst
            except CTTError:
                # First failure: this server has no per-file DNG route. Switch to
                # the archive once and stop hammering 404s for every frame.
                self._direct = False
                log("CTT server has no per-file download route — switching to the "
                    "project archive (still extracted to loose DNGs, no zip kept).",
                    "warn")
                return {}
        self._direct = True
        return out

    def finalize(self) -> dict[str, Path]:
        if self._direct:
            return {p.name: p for p in self.mirror.rglob("*.dng")}
        return _extract_archive_dngs(self.client, self.project, self.mirror)


class ArchiveTransfer(Transfer):
    """Zero-setup fallback: pull the whole-project ZIP and extract loose DNGs.

    Downloading the archive re-zips the entire project each call, so this backend
    defers to a single pull in ``finalize`` rather than fetching per station.
    """

    incremental = False

    def __init__(self, client: CTTClient, project: str, mirror: Path):
        self.client = client
        self.project = project
        self.mirror = mirror
        self.mirror.mkdir(parents=True, exist_ok=True)

    def fetch(self, filenames: list[str]) -> dict[str, Path]:
        return {}  # deferred — see finalize()

    def finalize(self) -> dict[str, Path]:
        return _extract_archive_dngs(self.client, self.project, self.mirror)


# --------------------------------------------------------------------------- #
#  Capture plan
# --------------------------------------------------------------------------- #
@dataclass
class Station:
    station_id: str
    title: str
    setup: str                      # rig instructions shown to the user
    image_type: str                 # dark | alsc | macbeth
    frames: int
    dest: Path                      # local NSA destination folder
    naming: Callable[[int], str]    # index -> local filename (DNG only)
    controls: dict | None = None    # controls to apply; None = auto-meter+lock
    colour_temp: int | None = None
    lux: int | None = None
    check_chart: bool = False       # show Macbeth confidence during verify
    meta: dict = field(default_factory=dict)


@dataclass
class Recorded:
    station: Station
    ctt_filenames: list[str]


def build_plan(project_root: Path, args: argparse.Namespace,
               controls_range: dict) -> list[Station]:
    """Assemble the ordered station plan from the dataset layout + CLI options.

    Two modes (``args.mode``):
      * ``real``  — capture genuine noisy/gt scene pairs (real sensor noise) into
                    PI_RAW/Data/<scene>/imx662_<ag>_test/. No calibration.
      * ``calib`` — bias/dark/flat noise-model calibration + clean scene bursts
                    for the synthesis pipeline.
    """
    mode = getattr(args, "mode", "real")
    ag_tag = getattr(args, "ag_tag", "ag24")
    gain = args.gain
    cal = project_root / f"calibration/imx662_gain{gain}"
    # The advertised exposure_max is the sensor's absolute ceiling (hundreds of
    # seconds); the *usable* ceiling is frame-duration limited (~33 ms at 30 fps).
    # So flats ramp over explicit millisecond bounds, only clamped to the sensor.
    exp_min = int(controls_range.get("exposure_min", 13))
    exp_max = int(controls_range.get("exposure_max", 33_000))

    # CRITICAL: the noise model is fitted straight from these frames and applied
    # at synthesis WITHOUT any gain scaling — so bias/dark/flat must ALL be shot
    # at the real operating (low-light) gain, or the model under-represents the
    # noise. `gain` is the target sensor gain (e.g. 256×/512×); the analogue-gain
    # overrides default to it so calibration matches the regime we synthesise for.
    cal_gain = float(args.analogue_gain) if getattr(args, "analogue_gain", 0) else float(gain)
    flat_gain = float(args.flat_gain) if getattr(args, "flat_gain", 0) else cal_gain

    stations: list[Station] = []

    # bias/dark/flat are built here but only kept for mode == "calib" (filtered
    # out at the end for "real" mode, which captures scene pairs only).
    # 1) bias — read noise at the operating gain: lens cap, minimum exposure.
    stations.append(Station(
        station_id="bias",
        title=f"BIAS  ·  read noise + ADC offset at {cal_gain:g}× gain",
        setup=(
            "• LENS CAP ON — fully opaque, no light leaks (tape foil over a thin cap).\n"
            "• Room lighting doesn't matter with the cap on.\n"
            f"• Shot at {cal_gain:g}× gain. Measures read noise + ADC offset;\n"
            "  the preview is BLACK on purpose."
        ),
        image_type="dark",
        frames=args.bias_frames,
        dest=cal / "bias",
        naming=lambda i: f"bias_{i:02d}.dng",
        controls={"auto_exposure": False, "exposure": exp_min, "gain": cal_gain},
    ))

    # 2) dark — row/pattern noise at the operating gain: lens cap, normal exposure.
    dark_exp = int(args.dark_exposure_ms * 1000)
    stations.append(Station(
        station_id="dark",
        title=f"DARK  ·  fixed-pattern noise at {cal_gain:g}× gain",
        setup=(
            "• LENS CAP ON (opaque, no leaks — same as bias).\n"
            f"• Shot at {cal_gain:g}× gain. Measures dark current / fixed-pattern\n"
            "  (row) noise. Preview stays black on purpose."
        ),
        image_type="dark",
        frames=args.dark_frames,
        dest=cal / "dark",
        naming=lambda i: f"dark_{i:02d}.dng",
        controls={"auto_exposure": False, "exposure": dark_exp, "gain": cal_gain},
    ))

    # 3) flat/level_XX — photon-transfer curve at the operating gain. One manual
    #    setup (uniform grey card + constant light); the wizard ramps EXPOSURE
    #    across levels. High gain amplifies signal, so the exposure ceiling is
    #    scaled down ∝ 1/gain to keep a sensible sweep instead of instant clipping.
    gain_scale = min(1.0, 16.0 / max(flat_gain, 1.0))
    lo = min(max(exp_min, int(args.flat_min_ms * 1000 * gain_scale)), exp_max)
    hi = min(max(lo + 1, int(args.flat_max_ms * 1000 * gain_scale)), exp_max)
    n = max(2, args.flat_levels)
    for k in range(1, n + 1):
        # Geometric exposure ramp lo → hi across the levels.
        t = (k - 1) / (n - 1)
        exp = int(round(lo * (hi / lo) ** t))
        lvl = f"{k:02d}"
        setup = (
            "• Lens cap OFF, room lit. Fill the frame with a UNIFORM grey card /\n"
            "  diffuser (no texture, no hotspots).\n"
            f"• Shot at {flat_gain:g}× gain. If CLIPPING flags the brightest level,\n"
            "  dim the light, then keep light + gain fixed.\n"
            f"• Level {k}/{n}: exposure ≈ {exp/1000:.2f} ms (auto-set)."
        ) if k == 1 else None
        stations.append(Station(
            station_id=f"flat_{lvl}",
            title=f"FLAT level {k}/{n}  ·  exposure ≈ {exp/1000:.2f} ms",
            setup=setup or f"Same grey card / light. Exposure auto-set to ≈ {exp/1000:.2f} ms.",
            image_type="alsc",
            frames=2,  # a + b at each level
            dest=cal / "flat" / f"level_{lvl}",
            naming=lambda i: ("a.dng", "b.dng")[i] if i < 2 else f"c{i}.dng",
            controls={"auto_exposure": False, "exposure": exp, "gain": flat_gain},
            colour_temp=args.colour_temp,
            meta={"first_flat": k == 1},
        ))

    # 4) scene bursts. The burst frames land in bursts/<scene>/take01/ either way;
    #    the mode changes only what we derive from them afterwards.
    pi_raw = project_root / "PI_RAW"
    gain_sweep = parse_gain_sweep(getattr(args, "gain_sweep", None) or DEFAULT_GAIN_SWEEP)
    for scene in args.scenes:
        if mode == "real":
            gains_txt = ", ".join(f"{g}×" for g in gain_sweep)
            title = f"REAL PAIR SWEEP  ·  {scene}"
            setup = (
                f"• Rigid tripod, lens cap OFF. Nothing may move during the sweep.\n"
                "• Light it well (~100%, back off only if CLIPPING flags highlights).\n"
                f"• CAPTURE averages more frames at higher gain for cleaner GT "
                f"(base {args.burst_frames} @ 1×, up to {GT_BURST_MAX_FRAMES}; "
                f"gains {gains_txt}), dropping exposure as gain rises.\n"
                f"• → PI_RAW/Data/{scene}/imx662_ag<GAIN>_test/"
            )
            meta = {
                "scene": scene, "is_real_pair": True, "gain_sweep": list(gain_sweep),
                "burst_root": str(project_root / "bursts" / scene),
                "pair_root": str(pi_raw / "Data" / scene),
                "exp_min": exp_min, "exp_max": exp_max,
            }
        else:
            title = f"SCENE BURST  ·  {scene}"
            setup = (
                f"• Rigid tripod, lens cap OFF. Nothing may move during the burst.\n"
                "• Light it well — this is the CLEAN reference (low-light is\n"
                "  synthesised later from the noise model).\n"
                f"• CAPTURE meters once, locks exposure/gain, then shoots "
                f"{args.burst_frames} frames to average."
            )
            meta = {"scene": scene}
        # CTT rejects "macbeth"-type captures with no (or non-positive) lux.
        # The scene-name suffix is intensity %, not lux — use CLI/default here;
        # the measured lux is filled in after the lightbox is set at capture time.
        station_lux = args.lux or 500
        stations.append(Station(
            station_id=f"burst_{scene}",
            title=title,
            setup=setup,
            image_type="macbeth",
            frames=args.burst_frames,
            dest=project_root / "bursts" / scene / "take01",
            naming=lambda i: f"burst_{i:03d}.dng",
            controls=None,  # auto-meter then lock
            colour_temp=args.colour_temp,
            lux=station_lux,
            check_chart=False,
            meta=meta,
        ))

    # Real mode captures scene pairs only — drop the calibration stations.
    if mode == "real":
        stations = [s for s in stations if s.station_id.startswith("burst_")]
    return stations


def derive_real_pair(burst_dir: Path | str, test_dir: Path | str, *,
                     min_frames: int = 8, max_side: int = 0) -> dict:
    """Turn a captured burst into a real noisy/gt pair (denoise-hw layout).

    Both files are written as PNG — matching every other <sensor>_ag<N>_test
    folder already in PI_RAW (imx219_*, and the existing imx662_ag12_test
    sample), not a DNG-for-noisy format unique to this tool. ``noisy.png`` is
    one real frame decoded straight off the sensor (genuine sensor noise, not
    synthesised or averaged); ``gt.png`` is the temporal average of the burst.
    Written into ``test_dir`` (e.g. PI_RAW/Data/<scene>/imx662_ag24_test/).
    """
    from nsa.gt_capture import burst_folder_to_gt, list_burst_frames, write_gt_png
    from nsa.raw_io import _load_any

    burst_dir = Path(burst_dir)
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    frames = list_burst_frames(burst_dir)  # sorted; raises if empty

    noisy_rgb = _load_any(frames[0])  # decode one real frame — real sensor noise
    write_gt_png(test_dir / "noisy.png", noisy_rgb)

    manifest = burst_folder_to_gt(
        str(burst_dir), str(test_dir / "gt.png"),
        min_frames=min(max(2, min_frames), len(frames)), max_side=max_side)
    return {"scene_dir": str(test_dir), "noisy": "noisy.png", "gt": "gt.png",
            "frames_used": manifest["frames_used"]}


# --------------------------------------------------------------------------- #
#  Gain sweep — one rig setup, many gains
# --------------------------------------------------------------------------- #
def capture_gain_sweep(client: CTTClient, project: str, station: Station, *,
                       burst_frames: int,
                       status: Callable[..., None] = lambda *_a, **_k: None,
                       stop: Callable[[], bool] = lambda: False) -> list[Recorded]:
    """Sweep the analogue gain for a real-pair scene after a single rig setup.

    Meters the scene ONCE (auto-exposure) to fix the target brightness = exposure
    × gain, then for each gain in ``station.meta['gain_sweep']`` sets a
    constant-brightness exposure (∝ 1/gain), captures a burst, and returns one
    ``Recorded`` per gain — each filed to ``bursts/<scene>/ag<g>`` and paired into
    ``imx662_ag<g>_test``. Folder tags use the REQUESTED gain; the actual gain the
    sensor reports (it can clamp at the tuning/mode limit) is kept in the meta so
    a sidecar can record the ground truth.
    """
    meta = station.meta
    scene = meta["scene"]
    gains = list(meta["gain_sweep"])
    burst_root = Path(meta["burst_root"])
    pair_root = Path(meta["pair_root"])
    exp_min = int(meta.get("exp_min", 13))
    exp_max = int(meta.get("exp_max", 33_000))

    # Meter once (auto-exposure) to establish the target brightness. Using the
    # exposure×gain product means we can rebuild a matching exposure at any gain.
    client.set_controls({"auto_exposure": True})
    time.sleep(1.5)
    cur = client.get_controls()
    product = max(float(exp_min),
                  float(cur.get("exposure", 10_000)) * float(cur.get("gain", 1.0)))
    status(f"Metered {scene}: brightness target ≈ {product/1000:.1f} ms·× — "
           f"sweeping {len(gains)} gain(s).", "ok")

    recs: list[Recorded] = []
    for g in gains:
        if stop():
            break
        n_frames = burst_frames_for_gain(g, burst_frames)
        exposure = int(min(max(round(product / g), exp_min), exp_max))
        client.set_controls({"auto_exposure": False, "gain": float(g),
                             "exposure": exposure})
        time.sleep(0.4)  # gain/exposure latch a frame or two later
        actual = float(client.get_controls().get("gain", g))
        clamped = abs(actual - g) / g > 0.10
        if clamped:
            status(f"ag{g}: sensor clamped gain to {actual:g}× (tuning/mode limit); "
                   f"folder still tagged ag{g}, actual gain recorded.", "warn")
        added = client.capture(project, image_type=station.image_type,
                               frames=n_frames, colour_temp=station.colour_temp,
                               lux=station.lux)
        fnames = [a["filename"] for a in added
                  if a.get("filename", "").lower().endswith(".dng")]
        gstation = Station(
            station_id=f"burst_{scene}_ag{g}",
            title=f"{scene} · ag{g}",
            setup="",
            image_type=station.image_type,
            frames=n_frames,
            dest=burst_root / f"ag{g}",
            naming=lambda i: f"burst_{i:03d}.dng",
            colour_temp=station.colour_temp,
            lux=station.lux,
            meta={"scene": scene, "is_real_pair": True,
                  "pair_dest": str(pair_root / f"imx662_ag{g}_test"),
                  "requested_gain": g, "actual_gain": actual,
                  "clamped": clamped, "exposure_us": exposure,
                  "burst_frames": n_frames},
        )
        recs.append(Recorded(station=gstation, ctt_filenames=fnames))
        status(f"ag{g}: {len(fnames)}/{n_frames} DNG(s) at {actual:g}× "
               f"(exp {exposure/1000:.2f} ms, GT avg).", "ok")
    return recs


def write_gain_sidecar(pair_dir: Path | str, rec_meta: dict) -> None:
    """Record the TRUE captured gain next to a derived pair. Folder tags use the
    requested gain; this ``gain.json`` is the ground truth of what the sensor did
    (and flags when the requested gain was clamped)."""
    import json
    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "gain.json").write_text(json.dumps({
        "requested_gain": rec_meta.get("requested_gain"),
        "actual_gain": rec_meta.get("actual_gain"),
        "clamped": bool(rec_meta.get("clamped", False)),
        "exposure_us": rec_meta.get("exposure_us"),
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Wizard
# --------------------------------------------------------------------------- #
def _apply_controls(client: CTTClient, station: Station) -> dict:
    """Apply the station's controls (auto-meter+lock when controls is None)."""
    if station.controls is not None:
        return client.set_controls(station.controls)
    # Auto-meter then lock: let AE settle, read the metered values, fix them.
    client.set_controls({"auto_exposure": True})
    time.sleep(1.5)
    cur = client.get_controls()
    return client.set_controls({
        "auto_exposure": False,
        "exposure": int(cur.get("exposure", 10_000)),
        "gain": float(cur.get("gain", 1.0)),
    })


def _verify_panel(client: CTTClient, station: Station) -> None:
    c = client.get_controls()
    hist = client.histogram()
    rows = [
        ("exposure", f"{c.get('exposure', 0)/1000:.2f} ms"),
        ("gain", f"{c.get('gain', 0):g}×"),
        ("colour temp", f"{c.get('colour_temp', 0)} K"),
        ("lux", f"{c.get('lux', 0)}"),
        ("focus FoM", str(c.get("focus_fom", 0))),
        ("auto exposure", "on" if c.get("auto_exposure") else "off (locked)"),
    ]
    clip = hist.get("clipping") or {}
    if clip:
        rows.append(("clipping", "  ".join(f"{k} {v}%" for k, v in clip.items())))
    console.print(kv_table(rows, title=f"Live · {station.title}"))
    if clip and any(float(v) > 1.0 for v in clip.values()):
        log("Highlights are clipping — lower exposure/light for a clean flat.", "warn")
    if station.check_chart:
        mb = client.macbeth()
        if mb.get("found"):
            log(f"Macbeth chart detected (confidence {mb.get('confidence', 0):.2f}).", "ok")
        else:
            log("Macbeth chart NOT detected — reframe the chart.", "warn")


def _place_files(mirror_map: dict[str, Path], rec: Recorded) -> int:
    """Copy the station's DNGs from the local mirror into the NSA layout."""
    rec.station.dest.mkdir(parents=True, exist_ok=True)
    placed = 0
    for i, fname in enumerate(rec.ctt_filenames):
        src = mirror_map.get(fname)
        if src is None or not src.exists():
            continue
        dst = rec.station.dest / rec.station.naming(i)
        shutil.copy2(src, dst)
        placed += 1
    return placed


def _prompt(msg: str) -> str:
    try:
        return console.input(f"[bold cyan]{msg}[/] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def run_wizard(client: CTTClient, transfer: Transfer, stations: list[Station],
               *, dry_run: bool, lightbox_percent: float | None = None) -> list[Recorded]:
    recorded: list[Recorded] = []

    for n, st in enumerate(stations, 1):
        level_rule(n, st.title)
        console.print(st.setup)
        console.print()

        if dry_run:
            extra = ""
            if st.meta.get("is_real_pair"):
                scene = st.meta.get("scene", "")
                illum, _ = parse_scene_light(scene)
                if illum:
                    pct = scene_lightbox_percent(scene, override=lightbox_percent)
                    extra += f" · lightbox {illum} at {pct:.0f}%"
                pair_root = st.meta.get("pair_root")
                if pair_root:
                    extra += f" → pairs under {pair_root}/imx662_ag<gain>_test/"
            log(f"[dry-run] would apply controls={st.controls} · {st.frames} frame(s) "
                f"→ {st.dest}{extra}", "info")
            continue

        # For scene bursts, set the lightbox illuminant + intensity BEFORE the
        # camera auto-meters — otherwise the locked exposure is metered against
        # the wrong (pre-light-change) brightness. The box is driven by intensity
        # % only; we then read the measured lux back purely as capture metadata.
        capture_lux = st.lux
        if st.meta.get("is_real_pair"):
            scene = st.meta.get("scene")
            illum, _ = parse_scene_light(scene)
            if illum:
                try:
                    pct = scene_lightbox_percent(scene, override=lightbox_percent)
                    client.set_lightbox(illum, pct)
                    meas = client._settled_lux()
                    log(f"Lightbox set to {illum} at {pct:.0f}% "
                        f"(measured {meas:.0f} lux)", "ok")
                    if meas > 0:
                        capture_lux = int(round(meas))  # metadata only
                except Exception as exc:  # noqa: BLE001
                    log(f"Lightbox setup failed: {exc} (proceeding anyway)", "warn")

        # Apply controls up-front so the live view reflects the shot settings.
        try:
            _apply_controls(client, st)
        except CTTError as exc:
            log(f"Could not apply controls: {exc}", "err")
            if _prompt("[s]kip this station or [q]uit?") == "q":
                break
            continue

        # Verify / capture loop.
        action = "capture"
        while True:
            try:
                _verify_panel(client, st)
            except CTTError as exc:
                log(f"Live read failed: {exc}", "warn")
            choice = _prompt(
                f"Set up the rig, then [Enter] to capture {st.frames} frame(s)  ·  "
                "[r]echeck  ·  [s]kip  ·  [q]uit:"
            )
            if choice in ("", "c", "go"):
                action = "capture"
                break
            if choice == "r":
                continue
            if choice == "s":
                action = "skip"
                break
            if choice == "q":
                action = "quit"
                break
        if action == "skip":
            log(f"Skipped {st.station_id}.", "warn")
            continue
        if action == "quit":
            log("Quitting the wizard.", "warn")
            return recorded

        # Real-pair scenes sweep the gain series in one go; everything else fires
        # a single burst.
        if st.meta.get("gain_sweep"):
            try:
                recs = capture_gain_sweep(
                    client, PROJECT_NAME, st, burst_frames=st.frames,
                    status=lambda m, k="info": log(m, k))
            except CTTError as exc:
                log(f"Gain sweep failed: {exc}", "err")
                if _prompt("[q]uit · [Enter] continue:") == "q":
                    break
                continue
            recorded.extend(recs)
            if transfer.incremental:
                for rec in recs:
                    try:
                        mirror = transfer.fetch(rec.ctt_filenames)
                        placed = _place_files(mirror, rec)
                        log(f"Filed {placed} DNG(s) → {rec.station.dest}", "ok")
                    except CTTError as exc:
                        log(f"Transfer failed (will retry at finalize): {exc}", "warn")
            continue

        # Fire the burst.
        try:
            added = client.capture(
                PROJECT_NAME, image_type=st.image_type, frames=st.frames,
                colour_temp=st.colour_temp, lux=capture_lux,
            )
        except CTTError as exc:
            log(f"Capture failed: {exc}", "err")
            if _prompt("[r]etry later by re-running · [q]uit · [Enter] continue:") == "q":
                break
            continue

        fnames = [a["filename"] for a in added if a.get("filename", "").lower().endswith(".dng")]
        rec = Recorded(station=st, ctt_filenames=fnames)
        recorded.append(rec)
        log(f"Captured {len(fnames)} DNG(s): {', '.join(fnames[:4])}"
            + (" …" if len(fnames) > 4 else ""), "ok")

        # Pull + place immediately when the backend supports per-station fetch;
        # the archive path defers to finalize().
        if transfer.incremental:
            try:
                mirror = transfer.fetch(fnames)
                placed = _place_files(mirror, rec)
                log(f"Pulled + filed {placed} DNG(s) → {rec.station.dest}", "ok")
            except CTTError as exc:
                log(f"Transfer failed (will retry at finalize): {exc}", "warn")

    return recorded


def finalize_placement(transfer: Transfer, recorded: list[Recorded]) -> None:
    """Ensure every recorded station's files are pulled and filed locally."""
    log("Finalising transfer — pulling any remaining DNGs …", "info")
    mirror = transfer.finalize()
    if not mirror:
        log("Nothing to finalise (files already placed).", "info")
        return
    total = 0
    for rec in recorded:
        # Only re-place stations whose destination is empty/partial.
        want = len(rec.ctt_filenames)
        have = len(list(rec.station.dest.glob("*.dng"))) if rec.station.dest.is_dir() else 0
        if have >= want and have > 0:
            continue
        total += _place_files(mirror, rec)
    log(f"Finalised placement ({total} DNG(s) filed).", "ok")


# --------------------------------------------------------------------------- #
#  Publish to the AI-server dataset root (with read-back verification)
# --------------------------------------------------------------------------- #
def is_remote_publish_dest(dest: str) -> bool:
    """True for ``user@host:/path`` rsync destinations."""
    from nsa.denoise_hw_data import is_remote_publish_dest as _is_remote
    return _is_remote(dest)


def check_publish_dest(dest_root: Path | str) -> str | None:
    """Return a human-readable problem with *dest_root*, or None if it looks OK."""
    dest_s = normalize_publish_dest(str(dest_root).strip())
    if is_remote_publish_dest(dest_s):
        if shutil.which("rsync") is None:
            return "rsync is not on PATH — install rsync for remote publish."
        return None
    dest = Path(dest_s).expanduser()
    if not dest.exists():
        return (
            f"{dest} does not exist on THIS computer.\n\n"
            "If the AI training server is a different machine, set the dataset "
            "root to  user@ai-host:/opt/datasets/PI_RAW  (rsync over SSH)."
        )
    # Publish only creates Data/<scene>/imx662_ag<GAIN>_test/ — check that path.
    data = dest / "Data"
    if data.is_dir():
        if os.access(data, os.W_OK):
            return None
    elif os.access(dest, os.W_OK):
        return None
    owner = "another user"
    try:
        import pwd
        st = (data if data.is_dir() else dest).stat()
        owner = pwd.getpwuid(st.st_uid).pw_name
    except (ImportError, KeyError, OSError):
        pass
    data_path = data if data.is_dir() else dest / "Data"
    return (
        f"Your account cannot write new imx662 pair folders under {dest}.\n\n"
        f"Owner is «{owner}» (see ls -la). The shared PI_RAW dataset is already "
        "there — do NOT chown or recreate the whole tree.\n\n"
        "Ask the owner or admin to grant write under Data/ only, e.g.:\n"
        f"  sudo usermod -aG {owner} $USER   # then log out/in\n"
        f"  sudo chmod g+w {data_path}\n"
        f"  sudo chmod g+w {data_path}/*/    # scene folders (for new imx662_* dirs)\n\n"
        "Or ACLs (adds access without changing existing file ownership):\n"
        f"  sudo setfacl -R -m u:$USER:rwx {data_path}\n"
        f"  sudo setfacl -R -d -m u:$USER:rwx {data_path}\n\n"
        "Publish only adds Data/<scene>/imx662_ag<GAIN>_test/ — existing "
        "imx219/sensor data is not deleted or overwritten elsewhere.\n\n"
        "Without contacting the owner you can:\n"
        "  • Uncheck Publish — captures stay in your project PI_RAW folder.\n"
        f"  • Set publish path to {_home_pi_raw()} (your home folder).\n"
        "  • Point Dataset Studio / training at that PI_RAW root instead of /opt.")


def _rsync_ssh_arg_publish() -> str:
    return "ssh " + " ".join(SSH_OPTS_PUBLISH)


# Only these PI_RAW leaf folders may be published into the shared training dataset.
# Existing imx219_* / other sensor trees are never touched or deleted.
_IMX662_PAIR_DIR = re.compile(r"^imx662_ag\d+_test$")


def imx662_pair_dirs(pi_raw: Path | str,
                     hints: Sequence[Path | str] | None = None) -> list[Path]:
    """Pair folders safe to publish (``Data/<scene>/imx662_ag<GAIN>_test`` only)."""
    root = Path(pi_raw).expanduser().resolve()
    if hints:
        out: list[Path] = []
        for item in hints:
            d = Path(item).expanduser().resolve()
            if not d.is_dir():
                continue
            try:
                rel = d.relative_to(root)
            except ValueError:
                continue
            if (len(rel.parts) >= 2 and rel.parts[0] == "Data"
                    and _IMX662_PAIR_DIR.match(rel.name)):
                out.append(d)
        return out
    return sorted(
        d for d in root.glob("Data/*/*")
        if d.is_dir() and _IMX662_PAIR_DIR.match(d.name))


def _collect_publish_files(pi_raw: Path,
                           hints: Sequence[Path | str] | None = None) -> tuple[list[Path], list[Path]]:
    """Return (pair_dirs, files) scoped to new IMX662 pair folders only."""
    src = pi_raw.expanduser().resolve()
    pair_dirs = imx662_pair_dirs(src, hints)
    files: list[Path] = []
    seen: set[str] = set()
    for d in pair_dirs:
        for f in d.rglob("*"):
            if f.is_file():
                key = str(f)
                if key not in seen:
                    seen.add(key)
                    files.append(f)
    return pair_dirs, files


def _apply_publish_copy(files: list[Path], src: Path, dest: Path,
                        summary: dict[str, Any]) -> None:
    """Copy *files* into *dest* — additive only; never deletes existing data."""
    for f in files:
        rel = f.relative_to(src)
        target = dest / rel
        try:
            if target.exists() and target.stat().st_size == f.stat().st_size:
                summary["skipped"] += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                summary["copied"] += 1
            if target.exists() and target.stat().st_size == f.stat().st_size:
                summary["verified"] += 1
            else:
                summary["failures"].append(str(rel))
        except OSError as exc:
            summary["failures"].append(f"{rel}: {exc}")


def _publish_auth_error_extra(err: str, host: str, *, had_password: bool) -> str:
    """Extra guidance for SSH auth failures during publish."""
    low = err.lower()
    if "permission denied" not in low and "publickey" not in low:
        return ""
    lines = [
        "",
        "Publish copies from this computer to the AI training dataset. "
        "The Pi's SSH keys are not used for this step — only login from "
        f"here to {host}.",
    ]
    if had_password:
        if not shutil.which("sshpass"):
            lines.append(
                "A password was entered but SSH could not use it. Install "
                "sshpass:  sudo apt install sshpass")
        else:
            lines.append(
                "SSH rejected the password — check it, or set up key login: "
                f"ssh-copy-id {host}")
    else:
        lines.append("Enter your password when prompted, or run ssh-copy-id.")
    return "\n".join(lines)


def _publish_pi_raw_rsync(src: Path, dest: str, *,
                          ssh_password: str | None = None,
                          only_under: Sequence[Path | str] | None = None) -> dict:
    """``rsync`` new IMX662 pair folders into ``user@host:/path`` (additive only)."""
    summary: dict[str, Any] = {"dest": dest, "copied": 0, "verified": 0,
                               "skipped": 0, "failures": [], "error": None,
                               "remote": True, "src": str(src),
                               "needs_password": False, "auth_failed": False,
                               "published_dirs": []}
    try:
        pair_dirs, files = _collect_publish_files(src, only_under)
        summary["published_dirs"] = [str(d.relative_to(src)) for d in pair_dirs]
        if not pair_dirs:
            summary["error"] = (
                f"nothing to publish — no imx662_ag<GAIN>_test folders under {src}. "
                "Existing imx219/other sensor data is never copied or changed.")
            return summary
        if not files:
            summary["error"] = f"nothing to publish — pair folders have no files yet."
            return summary
        if shutil.which("rsync") is None:
            summary["error"] = "rsync not found on PATH — install rsync for remote publish."
            return summary

        host, _, remote_path = dest.partition(":")
        remote_path = remote_path.rstrip("/") or "/"

        if ssh_password and not shutil.which("sshpass"):
            summary["auth_failed"] = True
            summary["error"] = (
                "Password-based SSH publish needs sshpass on this computer.\n"
                "Install it:  sudo apt install sshpass\n\n"
                f"Or set up key login:  ssh-copy-id {host}"
                + _publish_auth_error_extra("permission denied", host,
                                            had_password=True))
            return summary

        with ssh_publish_auth(ssh_password) as (env, prefix):
            rsync_ssh = _rsync_ssh_shell(ssh_password)
            for pair_dir in pair_dirs:
                rel = pair_dir.relative_to(src).as_posix()
                remote_sub = f"{host}:{remote_path}/{rel}/"
                mkdir = subprocess.run(
                    prefix + _ssh_cmd_publish(
                        host, f"mkdir -p {shlex.quote(f'{remote_path}/{rel}')}"),
                    capture_output=True, text=True, env=env, timeout=120)
                if mkdir.returncode != 0:
                    err = (mkdir.stderr or mkdir.stdout or "").strip()
                    had_pw = bool(ssh_password)
                    if "permission denied" in err.lower() or "publickey" in err.lower():
                        summary["auth_failed"] = True
                        if not had_pw:
                            summary["needs_password"] = True
                    summary["error"] = (
                        f"cannot create remote folder {remote_sub}: {err}"
                        + (f"\n\n{_ssh_hint(err, for_publish=True)}" if err else "")
                        + _publish_auth_error_extra(err, host, had_password=had_pw))
                    return summary

                dry = subprocess.run(
                    ["rsync", "-az", "--dry-run", "--itemize-changes",
                     "-e", rsync_ssh, f"{pair_dir}/", remote_sub],
                    capture_output=True, text=True, env=env, timeout=600)
                if dry.returncode != 0:
                    err = (dry.stderr or dry.stdout or "").strip()
                    had_pw = bool(ssh_password)
                    if "permission denied" in err.lower() or "publickey" in err.lower():
                        summary["auth_failed"] = True
                        if not had_pw:
                            summary["needs_password"] = True
                    summary["error"] = (
                        f"rsync dry-run failed for {rel}: {err}"
                        + (f"\n\n{_ssh_hint(err, for_publish=True)}" if err else "")
                        + _publish_auth_error_extra(err, host, had_password=had_pw))
                    return summary
                pending = [ln for ln in dry.stdout.splitlines()
                           if ln and ln[0] in "<>ch"]
                dir_files = [f for f in files
                             if str(f).startswith(str(pair_dir) + os.sep)]
                summary["skipped"] += len(dir_files) - len(pending)

                res = subprocess.run(
                    ["rsync", "-az", "-e", rsync_ssh, f"{pair_dir}/", remote_sub],
                    capture_output=True, text=True, env=env, timeout=3600)
                if res.returncode != 0:
                    err = (res.stderr or res.stdout or "").strip()
                    had_pw = bool(ssh_password)
                    if "permission denied" in err.lower() or "publickey" in err.lower():
                        summary["auth_failed"] = True
                        if not had_pw:
                            summary["needs_password"] = True
                    summary["error"] = (
                        f"rsync failed for {rel}: {err}"
                        + _publish_auth_error_extra(err, host, had_password=had_pw))
                    return summary
                summary["copied"] += len(pending)

            for f in files:
                rel = f.relative_to(src).as_posix()
                remote_file = f"{remote_path}/{rel}"
                stat_res = subprocess.run(
                    prefix + _ssh_cmd_publish(
                        host,
                        f"test -f {shlex.quote(remote_file)} && "
                        f"stat -c %s {shlex.quote(remote_file)}"),
                    capture_output=True, text=True, env=env, timeout=60)
                if stat_res.returncode != 0:
                    summary["failures"].append(f"{rel}: missing on remote")
                    continue
                try:
                    remote_size = int(stat_res.stdout.strip())
                except ValueError:
                    summary["failures"].append(f"{rel}: bad remote stat")
                    continue
                if remote_size == f.stat().st_size:
                    summary["verified"] += 1
                else:
                    summary["failures"].append(
                        f"{rel}: size mismatch "
                        f"(local {f.stat().st_size} vs remote {remote_size})")
    except subprocess.TimeoutExpired:
        summary["error"] = "publish timed out — check the network and retry."
    except OSError as exc:
        summary["error"] = f"publish failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"unexpected publish error: {exc}"

    return summary


def publish_pi_raw(project_pi_raw: Path | str,
                   dest_root: Path | str | None = None,
                   *, ssh_password: str | None = None,
                   only_under: Sequence[Path | str] | None = None) -> dict:
    """Copy new IMX662 pair folders into the shared training dataset (additive).

    Only ``Data/<scene>/imx662_ag<GAIN>_test/`` trees are ever published.
    Existing imx219_* and other sensor folders under ``dest_root`` are left
    untouched — nothing is deleted, moved, or rsync'd with ``--delete``.

    ``project_pi_raw`` is the wizard's ``<project>/PI_RAW`` folder.
    Pass ``only_under`` to limit to pair folders from the current cabinet/session.

    Returns a dict:
        {"dest": str, "src": str, "copied": int, "verified": int, "skipped": int,
         "failures": [str, ...], "error": str | None, "remote": bool,
         "published_dirs": [str, ...]}
  """
    src = Path(project_pi_raw).expanduser().resolve()
    dest_s = normalize_publish_dest(
        str(dest_root if dest_root is not None else default_publish_dest()).strip())
    if is_remote_publish_dest(dest_s):
        return _publish_pi_raw_rsync(src, dest_s, ssh_password=ssh_password,
                                     only_under=only_under)

    dest = Path(dest_s).expanduser()
    summary: dict[str, Any] = {"dest": str(dest), "src": str(src), "copied": 0,
                               "verified": 0, "skipped": 0, "failures": [],
                               "error": None, "remote": False, "published_dirs": []}

    pair_dirs, files = _collect_publish_files(src, only_under)
    summary["published_dirs"] = [str(d.relative_to(src)) for d in pair_dirs]
    if not pair_dirs:
        summary["error"] = (
            f"nothing to publish — no imx662_ag<GAIN>_test folders under {src}. "
            "Existing sensor data at the destination is not modified.")
        return summary
    if not files:
        summary["error"] = "nothing to publish — pair folders have no files yet."
        return summary

    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        summary["error"] = (
            f"cannot write under {dest}: {exc}.\n\n"
            "The shared dataset may already exist — do not chown the whole "
            "PI_RAW tree. Ask your admin for write access under Data/ only.")
        return summary

    _apply_publish_copy(files, src, dest, summary)

    return summary


def _log_publish(summary: dict) -> None:
    """Log a human-readable confirmation of a publish_pi_raw() result."""
    dest = summary["dest"]
    if summary["error"]:
        log(f"NOT published to AI server ({dest}): {summary['error']}", "err")
        return
    verified = summary["verified"]
    total = verified + len(summary["failures"])
    if summary["failures"]:
        log(f"Published to AI server {dest}: verified {verified}/{total} file(s); "
            f"FAILED: {', '.join(summary['failures'][:5])}"
            + (" …" if len(summary["failures"]) > 5 else ""), "warn")
    else:
        log(f"Confirmed on AI server {dest}: {verified} file(s) present "
            f"(copied {summary['copied']}, already-there {summary['skipped']}).", "ok")


# --------------------------------------------------------------------------- #
#  Post-processing — chain the NSA pipeline
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> int:
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def post_process(project_root: Path, args: argparse.Namespace,
                 scenes: list[str]) -> None:
    py = sys.executable
    cal_dir = project_root / f"calibration/imx662_gain{args.gain}"
    noise_json = project_root / f"models/noise/imx662_gain{args.gain}.json"
    clean_root = project_root / "clean_scenes"
    pi_raw = project_root / "PI_RAW"

    cmds: list[tuple[str, list[str]]] = [
        ("Calibrate the 5-phase noise model",
         [py, "calibrate_noise.py", "-i", str(cal_dir), "-o", str(noise_json),
          "--gain", str(args.gain)]),
    ]
    for scene in scenes:
        burst = project_root / "bursts" / scene / "take01"
        gt = clean_root / scene / "gt_01.png"
        cmds.append((f"Build clean GT for '{scene}' (temporal average)",
                     [py, "capture_gt.py", "-b", str(burst), "-o", str(gt),
                      "--min-frames", str(min(8, args.burst_frames))]))
    cmds.append((
        "Synthesize the IMX662 noisy/GT dataset",
        [py, "simulate_dataset.py", "-i", str(clean_root), "-o", str(pi_raw),
         "--calibration", str(noise_json), "--gain", str(args.gain)],
    ))

    console.print()
    level_rule(99, "Post-processing — NSA pipeline")
    if not _have_rawpy():
        log("rawpy is NOT installed in this environment — the steps below read raw "
            "DNGs and will fail without it. Install it first:", "err")
        console.print(f"[dim]  {py} -m pip install rawpy[/]")
    for title, cmd in cmds:
        console.print(f"\n[bold]{title}[/]")
        if args.run_post:
            rc = _run(cmd)
            log("ok" if rc == 0 else f"exit {rc}", "ok" if rc == 0 else "err")
        else:
            console.print(f"[dim]$ {' '.join(cmd)}[/]")
    if not args.run_post:
        log("Re-run with --run-post to execute these automatically.", "info")


def _rawpy_status() -> tuple[bool, str]:
    """(installed?, reason). Catches ANY import failure — rawpy can fail beyond
    a plain ImportError (missing libraw .so, NumPy ABI mismatch, etc.) — and
    reports the interpreter so 'install rawpy' points at the RIGHT Python."""
    try:
        import rawpy  # noqa: F401
        return True, getattr(rawpy, "__version__", "installed")
    except BaseException as exc:  # noqa: BLE001 - report, don't crash the wizard
        return False, (f"{type(exc).__name__}: {exc}\n"
                       f"Install it into THIS Python:\n"
                       f"    {sys.executable} -m pip install rawpy")


def _have_rawpy() -> bool:
    return _rawpy_status()[0]


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
PROJECT_NAME = "imx662"  # set from args in main()


def main() -> int:
    global PROJECT_NAME
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Connection
    p.add_argument("--host", default="10.3.195.212", help="CTT server host/IP")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--scheme", choices=("https", "http"), default="https",
                   help="CTT serves HTTPS with a self-signed cert by default")
    p.add_argument("--verify-tls", action="store_true",
                   help="verify the TLS cert (default: off, for the Pi's self-signed cert)")
    p.add_argument("--project", default="imx662", help="CTT project name")
    # NSA layout
    p.add_argument("--root", "-o", default="datasets/imx662_project",
                   help="NSA project root (holds PI_RAW/, calibration/, bursts/)")
    p.add_argument("--mode", choices=("real", "calib"), default="real",
                   help="'real' = capture genuine noisy/gt scene pairs (real noise); "
                        "'calib' = bias/dark/flat noise-model calibration + synthesis")
    p.add_argument("--ag-tag", default="ag24",
                   help="(legacy) single analogue-gain folder tag; ignored in real "
                        "mode, which now sweeps --gain-sweep")
    p.add_argument("--gain-sweep", type=parse_gain_sweep, default=list(DEFAULT_GAIN_SWEEP),
                   help="comma list of analogue gains to sweep per scene in real "
                        "mode (default: 1,2,4,8,16,32,64,128,256,512). Each lands in "
                        "PI_RAW/Data/<scene>/imx662_ag<gain>_test.")
    p.add_argument("--gain", type=int, default=256,
                   help="operating/calibration gain (folder imx662_gain<N>)")
    p.add_argument("--scenes", nargs="+", default=list(MANAGER_SCENES))
    p.add_argument("--colour-temp", type=int, default=5000)
    p.add_argument("--lux", type=int, default=None,
                   help="fallback lux metadata for CTT (measured lux used when available)")
    p.add_argument("--lightbox-percent", type=float, default=None,
                   help="override scene-name intensity %% for all stations (default: "
                        "parse from scene, e.g. cabinet_F11_25 → 25%%)")
    # Capture parameters
    p.add_argument("--analogue-gain", type=float, default=0.0,
                   help="analogue gain for bias/dark calibration; 0 = use --gain "
                        "(calibrate at the real operating gain)")
    p.add_argument("--bias-frames", type=int, default=8)
    p.add_argument("--dark-frames", type=int, default=5)
    p.add_argument("--dark-exposure-ms", type=float, default=20.0)
    p.add_argument("--flat-levels", type=int, default=12)
    p.add_argument("--flat-gain", type=float, default=0.0,
                   help="analogue gain for flat frames; 0 = use the calibration gain")
    p.add_argument("--flat-min-ms", type=float, default=1.0,
                   help="lowest flat exposure in ms (clamped to the sensor range)")
    p.add_argument("--flat-max-ms", type=float, default=30.0,
                   help="highest flat exposure in ms — keep below clipping")
    p.add_argument("--burst-frames", type=int, default=48,
                   help="frames per scene burst (looped over the 16-frame cap)")
    # Transfer
    p.add_argument("--transfer", choices=("rsync", "http", "archive"), default=None,
                   help="how to pull DNGs into the folder: rsync = incremental, "
                        "direct file copy over ssh (no zip; recommended); "
                        "http = over the CTT API, extracted to loose files (the "
                        "stock server has no per-file raw route, so this pulls the "
                        "project archive and unpacks it — no zip kept); archive = "
                        "same as http. Default: rsync if --pi-ssh given, else http.")
    p.add_argument("--pi-ssh", default=None,
                   help="ssh target for rsync + auto-start, e.g. pi@10.3.195.212")
    p.add_argument("--pi-workspace", default="~/ctt-server-workspace",
                   help="CTT workspace root on the Pi (CTT_CAPTURE_WORKSPACE)")
    p.add_argument("--pi-ctt-cmd", default="~/ctt-venv/bin/ctt-server",
                   help="command to launch the CTT server on the Pi "
                        "(auto-discovered in common venvs if not found)")
    p.add_argument("--no-autostart", action="store_true",
                   help="do not SSH in to start ctt-server if it isn't already running")
    p.add_argument("--autostart-wait", type=float, default=45.0,
                   help="seconds to wait for ctt-server to come up after launching")
    # Behaviour
    p.add_argument("--run-post", action="store_true",
                   help="also run calibrate_noise / capture_gt / simulate_dataset")
    p.add_argument("--publish-to", default=None,
                   help="dataset root to copy finished PI_RAW pairs into and verify "
                        f"(default: {default_publish_dest()} — USER@ai:{SYSTEM_PI_RAW} "
                        "from a laptop, or local /opt when on the AI server).")
    p.add_argument("--no-publish", action="store_true",
                   help="do not copy the finished pairs to the AI-server dataset root")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    PROJECT_NAME = args.project
    banner("CTT → NSA guided capture")

    # Resolve + scaffold the NSA layout so downstream tools & Dataset Studio see it.
    project_root, _ = resolve_layout(args.root)
    scaffold_imx662_project(project_root, gain=args.gain,
                            imx662_ag_tags=IMX662_TARGET_AG_TAGS,
                            scenes=tuple(args.scenes),
                            flat_levels=max(2, args.flat_levels))
    log(f"NSA project root: {project_root}", "ok")

    client = CTTClient(args.host, args.port, scheme=args.scheme, verify=args.verify_tls)

    controls_range: dict = {}
    if not args.dry_run:
        # Make sure the server is up — SSH in and start it if needed.
        try:
            state = ensure_server(
                client, ssh=args.pi_ssh, ctt_cmd=args.pi_ctt_cmd, port=args.port,
                workspace=args.pi_workspace, autostart=not args.no_autostart,
                wait_s=args.autostart_wait, status=lambda m: log(m, "info"))
            if state == "started":
                log("Auto-started ctt-server on the Pi.", "ok")
        except CTTError as exc:
            log(str(exc), "err")
            return 2
        try:
            h = client.health()
        except CTTError as exc:
            log(f"Cannot reach CTT at {client.base}: {exc}", "err")
            return 2
        if not h.get("camera"):
            log(f"CTT is up but reports no camera: {h.get('error', 'unknown')}", "err")
            return 2
        log(f"Connected to CTT at {client.base} — camera ready.", "ok")
        client.ensure_project(args.project)
        controls_range = client.get_controls()
    else:
        # Plausible IMX662 ranges for dry-run planning.
        controls_range = {"exposure_min": 100, "exposure_max": 33_000}

    stations = build_plan(project_root, args, controls_range)
    log(f"Plan: {len(stations)} stations · gain label {args.gain} · "
        f"{len(args.scenes)} scene burst(s).", "info")

    if args.dry_run:
        run_wizard(client, Transfer(), stations, dry_run=True,
                   lightbox_percent=args.lightbox_percent)
        return 0

    # Pick the transfer backend.
    mirror = project_root / ".ctt_mirror"
    mode = args.transfer or ("rsync" if args.pi_ssh else "http")
    try:
        if mode == "rsync":
            if not args.pi_ssh:
                log("--transfer rsync needs --pi-ssh (e.g. pi@10.3.195.212).", "err")
                return 2
            transfer: Transfer = RsyncTransfer(args.pi_ssh, args.pi_workspace,
                                               args.project, mirror)
            log(f"Transfer: rsync from {args.pi_ssh}:{args.pi_workspace}/{args.project}", "info")
        elif mode == "archive":
            transfer = ArchiveTransfer(client, args.project, mirror)
            log("Transfer: CTT project archive, extracted to loose DNGs "
                "(no zip kept), pulled at the end.", "info")
        else:
            transfer = HttpFileTransfer(client, args.project, mirror)
            log("Transfer: over the CTT API → loose DNGs in the folder "
                "(per-file if supported, else archive-extracted; no zip kept).",
                "info")
    except CTTError as exc:
        log(str(exc), "err")
        return 2

    recorded = run_wizard(client, transfer, stations, dry_run=False,
                          lightbox_percent=args.lightbox_percent)
    if not recorded:
        log("No captures recorded — nothing to place.", "warn")
        return 0

    finalize_placement(transfer, recorded)

    if args.mode == "real":
        console.print()
        level_rule(99, "Deriving real noisy/gt pairs")
        if not _have_rawpy():
            log("rawpy is required to average the burst into gt.png — install it: "
                f"{sys.executable} -m pip install rawpy", "err")
        for rec in recorded:
            meta = rec.station.meta
            if not meta.get("is_real_pair"):
                continue
            try:
                res = derive_real_pair(rec.station.dest, meta["pair_dest"],
                                       min_frames=min(8, args.burst_frames))
                write_gain_sidecar(meta["pair_dest"], meta)
                tag = ""
                if meta.get("requested_gain") is not None:
                    tag = f" [ag{meta['requested_gain']} → {meta.get('actual_gain', '?')}×]"
                log(f"{meta['scene']}{tag}: {res['noisy']} + {res['gt']} "
                    f"(gt from {res['frames_used']} frames) → {res['scene_dir']}", "ok")
            except Exception as exc:  # noqa: BLE001
                log(f"{meta['scene']}: could not derive pair: {exc}", "err")
    else:
        post_process(project_root, args, list(args.scenes))

    # Publish the finished PI_RAW pairs to the AI-server dataset root so training
    # picks them up, and read them back to confirm they actually landed.
    if not args.no_publish:
        console.print()
        level_rule(99, "Publishing to the AI-server dataset")
        summary = publish_pi_raw(
            project_root / "PI_RAW",
            args.publish_to if args.publish_to is not None else default_publish_dest())
        _log_publish(summary)

    console.print()
    log("Done. Point Dataset Studio's PI_RAW root at "
        f"{project_root} to inspect.", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
