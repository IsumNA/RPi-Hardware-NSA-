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
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.dataset_layout import (
    IMX662_TARGET_AG_TAGS,
    MANAGER_SCENES,
    resolve_layout,
    scaffold_imx662_project,
)
from nsa.theme import banner, console, kv_table, level_rule, log

# Capture is capped server-side at 16 frames per POST (see ctt-server capture()).
CTT_MAX_BURST = 16


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
    def health(self) -> dict:
        r = self._get("/api/health")
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
            added.extend(payload.get("added", []))
            remaining -= n
        return added

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


def _ssh_cmd(ssh: str, inner: str) -> list[str]:
    """An ssh argv that runs ``inner`` in a NON-login shell on the Pi.

    A login shell (``bash -lc``) would source the Pi's profile — which on
    Raspberry Pi OS prints a password-warning banner to stdout that pollutes
    command output. We use ``bash -c`` and rely on explicit paths / discovery
    instead of the login PATH.
    """
    return ["ssh", *SSH_OPTS, ssh, f"bash -c {shlex.quote(inner)}"]


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
        client.health()
        return "running"
    except CTTError:
        pass  # not reachable yet

    if not autostart:
        raise CTTError(f"CTT server at {client.base} is not responding "
                       "(auto-start is off).")
    if not ssh:
        raise CTTError("CTT server is not reachable and no SSH target is set to "
                       "start it. Set the Pi SSH target (e.g. pi@10.3.195.212).")
    if shutil.which("ssh") is None:
        raise CTTError("ssh was not found on PATH on this machine.")

    # If the given command isn't directly runnable (common when it's a venv
    # console script off the login PATH), auto-discover its real location.
    # Skipped for compound commands like 'source …/activate && ctt-server'.
    if all(tok not in ctt_cmd for tok in ("&&", ";", "|", " ")):
        chk = subprocess.run(_ssh_cmd(ssh, f"command -v {shlex.quote(ctt_cmd)}"),
                             capture_output=True, text=True)
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
                    "'source ~/ctt-venv/bin/activate && ctt-server'."
                    + (f"\n(ssh error: {chk.stderr.strip()})" if chk.stderr.strip() else ""))

    status(f"Starting {ctt_cmd} on {ssh} …")
    full = f"{ctt_cmd} --host 0.0.0.0 --port {int(port)}"
    if workspace:
        full += f" --workspace {shlex.quote(workspace)}"
    # setsid+nohup+</dev/null fully detaches so the ssh call returns immediately
    # while the server keeps running; output goes to a log on the Pi.
    launch = f"setsid nohup {full} > ~/ctt-server.log 2>&1 < /dev/null & echo LAUNCHED"
    res = subprocess.run(_ssh_cmd(ssh, launch), capture_output=True, text=True, timeout=25)
    if res.returncode != 0 or "LAUNCHED" not in res.stdout:
        raise CTTError("Could not start ctt-server over SSH: "
                       + (res.stderr.strip() or res.stdout.strip() or "unknown error"))

    deadline = time.time() + wait_s
    last = ""
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            client.health()
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


class ArchiveTransfer(Transfer):
    """Zero-setup fallback: pull the whole-project ZIP and extract DNGs.

    Downloading the archive re-zips the entire project each call, so this backend
    defers to a single pull in ``finalize`` rather than fetching per station.
    """

    def __init__(self, client: CTTClient, project: str, mirror: Path):
        self.client = client
        self.project = project
        self.mirror = mirror
        self.mirror.mkdir(parents=True, exist_ok=True)

    def fetch(self, filenames: list[str]) -> dict[str, Path]:
        return {}  # deferred — see finalize()

    def finalize(self) -> dict[str, Path]:
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / f"{self.project}.zip"
            log(f"Downloading project archive from CTT …", "info")
            self.client.download_archive(self.project, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".dng"):
                        out = self.mirror / Path(member).name
                        with zf.open(member) as src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
        return {p.name: p for p in self.mirror.glob("*.dng")}


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
    """Assemble the ordered station plan from the dataset layout + CLI options."""
    gain = args.gain
    cal = project_root / f"calibration/imx662_gain{gain}"
    # The advertised exposure_max is the sensor's absolute ceiling (hundreds of
    # seconds); the *usable* ceiling is frame-duration limited (~33 ms at 30 fps).
    # So flats ramp over explicit millisecond bounds, only clamped to the sensor.
    exp_min = int(controls_range.get("exposure_min", 13))
    exp_max = int(controls_range.get("exposure_max", 33_000))

    stations: list[Station] = []

    # 1) bias — read noise: lens cap, minimum exposure, unity gain.
    stations.append(Station(
        station_id="bias",
        title="BIAS  ·  read noise + ADC offset",
        setup=(
            "• Put the LENS CAP on (or fully dark enclosure).\n"
            "• Shortest possible exposure, analogue gain 1×.\n"
            "• No light must reach the sensor."
        ),
        image_type="dark",
        frames=args.bias_frames,
        dest=cal / "bias",
        naming=lambda i: f"bias_{i:02d}.dng",
        controls={"auto_exposure": False, "exposure": exp_min, "gain": 1.0},
    ))

    # 2) dark — row/pattern noise at the night gain: lens cap, normal exposure.
    dark_exp = int(args.dark_exposure_ms * 1000)
    stations.append(Station(
        station_id="dark",
        title=f"DARK  ·  fixed-pattern noise at {args.analogue_gain:g}× gain",
        setup=(
            "• Keep the LENS CAP on.\n"
            f"• Analogue gain {args.analogue_gain:g}×, exposure {args.dark_exposure_ms:g} ms.\n"
            "• Still fully dark — this measures dark current / row noise."
        ),
        image_type="dark",
        frames=args.dark_frames,
        dest=cal / "dark",
        naming=lambda i: f"dark_{i:02d}.dng",
        controls={"auto_exposure": False, "exposure": dark_exp, "gain": float(args.analogue_gain)},
    ))

    # 3) flat/level_XX — photon-transfer curve. One manual setup (uniform grey
    #    card + constant light); the wizard ramps EXPOSURE across levels, so no
    #    per-level light changes are needed.
    lo = min(max(exp_min, int(args.flat_min_ms * 1000)), exp_max)
    hi = min(max(lo + 1, int(args.flat_max_ms * 1000)), exp_max)
    n = max(2, args.flat_levels)
    for k in range(1, n + 1):
        # Geometric exposure ramp lo → hi across the levels.
        t = (k - 1) / (n - 1)
        exp = int(round(lo * (hi / lo) ** t))
        lvl = f"{k:02d}"
        setup = (
            "• Fill the frame with a UNIFORM, evenly-lit grey card / integrating sphere.\n"
            "• Keep the LIGHT and GAIN constant for every level — the wizard ramps\n"
            "  exposure automatically to walk up the signal curve.\n"
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
            controls={"auto_exposure": False, "exposure": exp, "gain": float(args.flat_gain)},
            colour_temp=args.colour_temp,
            meta={"first_flat": k == 1},
        ))

    # 4) scene bursts — clean GT via temporal averaging.
    for scene in args.scenes:
        stations.append(Station(
            station_id=f"burst_{scene}",
            title=f"SCENE BURST  ·  {scene}",
            setup=(
                f"• Frame the scene '{scene}' on a rigid tripod — it must be perfectly STATIC.\n"
                "• Normal lighting. The wizard meters once, locks exposure/gain, then\n"
                f"  shoots {args.burst_frames} identical frames for temporal averaging.\n"
                "• Nothing in the frame may move during the burst."
            ),
            image_type="macbeth",
            frames=args.burst_frames,
            dest=project_root / "bursts" / scene / "take01",
            naming=lambda i: f"burst_{i:03d}.dng",
            controls=None,  # auto-meter then lock
            colour_temp=args.colour_temp,
            lux=args.lux,
            check_chart=False,
            meta={"scene": scene},
        ))

    return stations


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
               *, dry_run: bool) -> list[Recorded]:
    recorded: list[Recorded] = []
    incremental = isinstance(transfer, RsyncTransfer)

    for n, st in enumerate(stations, 1):
        level_rule(n, st.title)
        console.print(st.setup)
        console.print()

        if dry_run:
            log(f"[dry-run] would apply controls={st.controls} · {st.frames} frame(s) "
                f"→ {st.dest}", "info")
            continue

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

        # Fire the burst.
        try:
            added = client.capture(
                PROJECT_NAME, image_type=st.image_type, frames=st.frames,
                colour_temp=st.colour_temp, lux=st.lux,
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

        # Pull + place immediately for rsync; archive defers to finalize().
        if incremental:
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


def _have_rawpy() -> bool:
    try:
        import rawpy  # noqa: F401
        return True
    except ImportError:
        return False


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
    p.add_argument("--gain", type=int, default=256,
                   help="calibration gain label (folder imx662_gain<N>)")
    p.add_argument("--scenes", nargs="+", default=list(MANAGER_SCENES))
    p.add_argument("--colour-temp", type=int, default=5000)
    p.add_argument("--lux", type=int, default=None)
    # Capture parameters
    p.add_argument("--analogue-gain", type=float, default=16.0,
                   help="AnalogueGain for the dark station (night gain)")
    p.add_argument("--bias-frames", type=int, default=8)
    p.add_argument("--dark-frames", type=int, default=5)
    p.add_argument("--dark-exposure-ms", type=float, default=20.0)
    p.add_argument("--flat-levels", type=int, default=12)
    p.add_argument("--flat-gain", type=float, default=1.0)
    p.add_argument("--flat-min-ms", type=float, default=1.0,
                   help="lowest flat exposure in ms (clamped to the sensor range)")
    p.add_argument("--flat-max-ms", type=float, default=30.0,
                   help="highest flat exposure in ms — keep below clipping")
    p.add_argument("--burst-frames", type=int, default=48,
                   help="frames per scene burst (looped over the 16-frame cap)")
    # Transfer
    p.add_argument("--transfer", choices=("rsync", "archive"), default=None,
                   help="how to pull DNGs (default: rsync if --pi-ssh given, else archive)")
    p.add_argument("--pi-ssh", default=None,
                   help="ssh target for rsync + auto-start, e.g. pi@10.3.195.212")
    p.add_argument("--pi-workspace", default="~/ctt-server-workspace",
                   help="CTT workspace root on the Pi (CTT_CAPTURE_WORKSPACE)")
    p.add_argument("--pi-ctt-cmd", default="ctt-server",
                   help="command to launch the CTT server on the Pi "
                        "(auto-discovered in common venvs if not on PATH)")
    p.add_argument("--no-autostart", action="store_true",
                   help="do not SSH in to start ctt-server if it isn't already running")
    p.add_argument("--autostart-wait", type=float, default=45.0,
                   help="seconds to wait for ctt-server to come up after launching")
    # Behaviour
    p.add_argument("--run-post", action="store_true",
                   help="also run calibrate_noise / capture_gt / simulate_dataset")
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
        run_wizard(client, Transfer(), stations, dry_run=True)
        return 0

    # Pick the transfer backend.
    mirror = project_root / ".ctt_mirror"
    mode = args.transfer or ("rsync" if args.pi_ssh else "archive")
    try:
        if mode == "rsync":
            if not args.pi_ssh:
                log("--transfer rsync needs --pi-ssh (e.g. pi@10.3.195.212).", "err")
                return 2
            transfer: Transfer = RsyncTransfer(args.pi_ssh, args.pi_workspace,
                                               args.project, mirror)
            log(f"Transfer: rsync from {args.pi_ssh}:{args.pi_workspace}/{args.project}", "info")
        else:
            transfer = ArchiveTransfer(client, args.project, mirror)
            log("Transfer: CTT project ZIP archive (pulled at the end).", "info")
    except CTTError as exc:
        log(str(exc), "err")
        return 2

    recorded = run_wizard(client, transfer, stations, dry_run=False)
    if not recorded:
        log("No captures recorded — nothing to place.", "warn")
        return 0

    finalize_placement(transfer, recorded)
    post_process(project_root, args, list(args.scenes))

    console.print()
    log("Done. Point Dataset Studio's PI_RAW root at "
        f"{project_root} to inspect.", "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
