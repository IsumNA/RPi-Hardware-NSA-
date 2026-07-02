"""Run live camera testing on a Raspberry Pi over SSH.

When you compile on the AI server and click LIVE TEST in the GUI, this syncs
``outputs/model.pt`` to the Pi and opens an SSH terminal that runs
``live.py --source picamera`` on Pi hardware (where the CSI camera works).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nsa.pi_camera import on_raspberry_pi


def load_pi_live_settings(project_root: Path | None = None) -> dict:
    """Settings from config.yaml ``pi_live:`` + env overrides."""
    root = project_root or Path(__file__).resolve().parents[1]
    settings = {
        "enabled": True,
        "ssh_host": os.environ.get("RPI_SSH_HOST", "rpi"),
        "repo": os.environ.get("RPI_REPO", "~/RPi-Hardware-NSA-"),
        "source": os.environ.get("RPI_LIVE_SOURCE", "picamera"),
        # Which display the OpenCV window renders on. ":0" = the Pi's own monitor
        # (its local desktop session). Empty = headless (save a preview PNG only).
        "display": os.environ.get("RPI_LIVE_DISPLAY", ":0"),
    }
    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            block = data.get("pi_live") or {}
            for key in settings:
                if key in block and block[key] is not None:
                    settings[key] = block[key]
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("RPI_LIVE_REMOTE", "").lower() in ("0", "false", "no"):
        settings["enabled"] = False
    return settings


def should_use_pi_remote(project_root: Path | None = None) -> bool:
    """True on Linux AI server — SSH to Pi for CSI camera (not Windows / not on-Pi)."""
    if sys.platform.startswith("win"):
        return False
    if os.environ.get("RPI_LIVE_LOCAL", "").lower() in ("1", "true", "yes"):
        return False
    settings = load_pi_live_settings(project_root)
    if not settings.get("enabled", True):
        return False
    if on_raspberry_pi():
        return False
    return True


def ssh_reachable(host: str, timeout: int = 8) -> tuple[bool, str]:
    """Return (ok, detail) from a quick BatchMode SSH probe."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=" + str(timeout),
             "-o", "StrictHostKeyChecking=accept-new", host, "true"],
            capture_output=True, text=True, timeout=timeout + 5, check=False)
        if r.returncode == 0:
            return True, ""
        detail = (r.stderr or r.stdout or "").strip()
        return False, detail or f"exit {r.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def ssh_setup_help(host: str, detail: str) -> str:
    return (
        f"Cannot SSH to '{host}'.\n\n"
        f"Detail: {detail}\n\n"
        "Fix on the AI server (one time):\n"
        "  1. Run:  ./setup_ssh_pi.sh YOUR_USER PI_IP\n"
        "     (needs Pi SSH enabled by lab admin + password once)\n"
        "  2. Or edit config.yaml:\n"
        "       pi_live:\n"
        "         ssh_host: YOUR_USER@PI_IP\n"
        "  3. Test:  ssh YOUR_USER@PI_IP\n"
        "     or:    python -m nsa.pi_remote --check"
    )


def remote_display_status(host: str, display: str = ":0") -> tuple[bool, str]:
    """Probe whether ``display`` on the Pi can accept an X window.

    Returns (ok, detail). Used by --check so the user learns *before* launching
    that the plain-SSH session has no reachable desktop for the OpenCV window.
    """
    if not display:
        return False, "headless (no display configured)"
    probe = (f"export DISPLAY={display}; "
             f"export XAUTHORITY=${{XAUTHORITY:-$HOME/.Xauthority}}; "
             f"if command -v xset >/dev/null 2>&1; then "
             f"xset q >/dev/null 2>&1 && echo OK || echo NODISP; "
             f"else echo NOXSET; fi")
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, probe],
            capture_output=True, text=True, timeout=20, check=False)
        out = (r.stdout or "").strip()
        if "OK" in out:
            return True, f"display {display} reachable"
        if "NOXSET" in out:
            return True, (f"cannot verify {display} (xset not installed) — "
                          f"the window will still be attempted")
        return False, (f"no desktop on {display}. Plug a monitor into the Pi with "
                       f"its desktop running, or set pi_live.display: '' for a "
                       f"headless preview PNG.")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def remote_live_shell_command(repo: str, source: str = "picamera",
                              display: str = ":0") -> str:
    """Bash one-liner run on the Pi after SSH.

    A plain SSH session has no ``$DISPLAY``, so the OpenCV preview window has
    nowhere to draw. Exporting ``DISPLAY`` (the Pi's own desktop, usually ``:0``)
    makes the RAW|DENOISED window appear on the monitor attached to the Pi. When
    ``display`` is empty we stay headless and live.py saves a preview PNG.
    """
    repo = repo.rstrip("/")
    disp = ""
    if display:
        # Point at the Pi's local X session; XAUTHORITY lets us attach to a
        # desktop started by the same login user without needing `xhost`.
        disp = (f"export DISPLAY={display}; "
                f"export XAUTHORITY=${{XAUTHORITY:-$HOME/.Xauthority}}; ")
    return (
        f"cd {repo} && mkdir -p outputs && "
        f"if [ -d .venv/bin ]; then . .venv/bin/activate; fi && "
        f"{disp}"
        f"(python3 live.py --source {source} || python live.py --source {source})"
    )


def sync_model_to_pi(project_root: Path, host: str, repo: str) -> str | None:
    """Copy outputs/model.pt to the Pi. Returns an error message or None."""
    ckpt = project_root / "outputs" / "model.pt"
    if not ckpt.is_file():
        return "No outputs/model.pt — run COMPILE first."
    remote_dir = repo.rstrip("/") + "/outputs"
    try:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", host,
             f"mkdir -p {remote_dir}"],
            check=True, timeout=20)
        subprocess.run(
            ["scp", "-o", "ConnectTimeout=8", str(ckpt),
             f"{host}:{remote_dir}/model.pt"],
            check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        return f"Could not copy model.pt to Pi (exit {exc.returncode})."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Could not copy model.pt to Pi: {exc}"
    summary = project_root / "outputs" / "summary.json"
    if summary.is_file():
        try:
            subprocess.run(
                ["scp", "-o", "ConnectTimeout=8", str(summary),
                 f"{host}:{remote_dir}/summary.json"],
                check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def launch_pi_terminal(host: str, remote_cmd: str,
                       project_root: Path | None = None) -> str | None:
    """Start the remote live session as a detached background SSH process.

    The camera window renders on the *Pi's* screen (via DISPLAY in remote_cmd),
    so the AI server never needs a local terminal window. We deliberately do NOT
    spawn gnome-terminal/konsole/etc — on a headless server those hang on a
    D-Bus ``org.gnome.Terminal`` timeout. SSH runs in the background and its
    output (live.py logs) is teed to ``outputs/pi_live.log``.
    """
    ssh_args = ["ssh", "-tt", host, remote_cmd]
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    log_path = root / "outputs" / "pi_live.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        subprocess.Popen(ssh_args, stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
        return None
    except OSError as exc:
        return f"Could not start SSH session to {host}: {exc}"


def run_live_on_pi(project_root: Path | None = None) -> str | None:
    """Sync model, open Pi terminal with live.py. Returns error text or None."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    s = load_pi_live_settings(root)
    host = str(s["ssh_host"])
    repo = str(s["repo"])
    source = str(s["source"])
    display = str(s.get("display", ":0"))

    ok, detail = ssh_reachable(host)
    if not ok:
        return ssh_setup_help(host, detail)

    err = sync_model_to_pi(root, host, repo)
    if err:
        return err

    remote_cmd = remote_live_shell_command(repo, source, display)
    return launch_pi_terminal(host, remote_cmd, project_root=root)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        print("\nSettings: config.yaml pi_live:  or  RPI_SSH_HOST / RPI_REPO env vars")
        return 0
    if argv and argv[0] in ("--check", "-c"):
        s = load_pi_live_settings(root)
        host = str(s["ssh_host"])
        display = str(s.get("display", ":0"))
        ok, detail = ssh_reachable(host)
        if not ok:
            print(ssh_setup_help(host, detail), file=sys.stderr)
            return 1
        print(f"SSH OK: {host}")
        disp_ok, disp_detail = remote_display_status(host, display)
        marker = "OK" if disp_ok else "!!"
        print(f"Display {marker}: {disp_detail}")
        if not disp_ok:
            print("\nThe camera window needs a screen. Options:\n"
                  "  - Attach a monitor to the Pi (its desktop must be running), or\n"
                  "  - Set  pi_live.display: ''  in config.yaml for a headless\n"
                  "    preview saved to outputs/live_preview.png on the Pi.",
                  file=sys.stderr)
        return 0 if disp_ok else 2
    err = run_live_on_pi(root)
    if err:
        print(err, file=sys.stderr)
        return 1
    print(f"Opened Pi live testing (ssh {load_pi_live_settings(root)['ssh_host']}).")
    print("Press q or ESC in the Pi window to stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
