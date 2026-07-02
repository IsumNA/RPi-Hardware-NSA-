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
        # Which display the OpenCV window renders on. "auto" = detect the Pi's
        # running desktop (X11/Wayland). ":0" forces that display. Empty =
        # headless (save a preview PNG only).
        "display": os.environ.get("RPI_LIVE_DISPLAY", "auto"),
        # Virtualenv on the Pi. "auto" tries .venv/venv/env; or set an explicit
        # path (e.g. ~/RPi-Hardware-NSA-/.venv) if yours lives elsewhere.
        "venv": os.environ.get("RPI_VENV", "auto"),
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


def _gui_env_snippet(display: str) -> str:
    """Bash that exports DISPLAY/XAUTHORITY/WAYLAND_DISPLAY for the Pi's desktop.

    A plain SSH shell has none of these, so a GUI window has nowhere to draw. We
    resolve them the robust way: copy them straight out of a currently-running
    desktop process's ``/proc/<pid>/environ`` (works for X11, Wayland and the
    XWayland server that Pi OS Bookworm's labwc/wayfire session uses — whose X
    auth cookie is NOT at ~/.Xauthority). Falls back to a forced display and a
    search of the usual XWayland/lightdm auth-file locations.
    """
    forced = "" if display in ("", "auto") else display
    force_line = f'export DISPLAY="${{DISPLAY:-{forced}}}"; ' if forced else ""
    return (
        r'''uid=$(id -u); '''
        r'''for pid in $(pgrep -u "$uid" 2>/dev/null); do '''
        r'''if [ -r "/proc/$pid/environ" ] && grep -qza "^DISPLAY=" "/proc/$pid/environ" 2>/dev/null; then '''
        r'''eval "$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | '''
        r'''grep -E '^(DISPLAY|XAUTHORITY|WAYLAND_DISPLAY|XDG_RUNTIME_DIR)=' | sed 's/^/export /')"; '''
        r'''break; fi; done; '''
        + force_line +
        r'''if [ -z "$XAUTHORITY" ] || [ ! -f "$XAUTHORITY" ]; then '''
        r'''for x in "$HOME/.Xauthority" /run/user/$uid/.mutter-Xwaylandauth.* '''
        r'''/run/user/$uid/xauth_* /var/run/lightdm/root/:0; do '''
        r'''[ -f "$x" ] && export XAUTHORITY="$x" && break; done; fi; '''
    )


def remote_display_status(host: str, display: str = "auto") -> tuple[bool, str]:
    """Probe whether the Pi has a reachable desktop for the OpenCV window.

    Returns (ok, detail). Used by --check so the user learns *before* launching
    whether the window can appear on the Pi (X11 or Wayland/XWayland).
    """
    if not display:
        return False, "headless (no display configured)"
    probe = _gui_env_snippet(display) + (
        'echo "resolved DISPLAY=${DISPLAY:-none} '
        'WAYLAND=${WAYLAND_DISPLAY:-none} XAUTH=${XAUTHORITY:-none}"; '
        'rt="${XDG_RUNTIME_DIR:-/run/user/$uid}"; '
        'if [ -n "$WAYLAND_DISPLAY" ] && [ -S "$rt/$WAYLAND_DISPLAY" ]; then echo WOK; fi; '
        'if command -v xset >/dev/null 2>&1; then '
        'DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" xset q >/dev/null 2>&1 '
        '&& echo XOK || echo XNO; else echo NOXSET; fi')
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, probe],
            capture_output=True, text=True, timeout=20, check=False)
        out = (r.stdout or "").strip()
        resolved = ""
        for line in out.splitlines():
            if line.startswith("resolved "):
                resolved = line[len("resolved "):]
        if "XOK" in out:
            return True, f"X display reachable  ({resolved})"
        if "WOK" in out:
            return True, f"Wayland session reachable via XWayland  ({resolved})"
        if "NOXSET" in out and "DISPLAY=none" not in out:
            return True, f"cannot verify (xset missing) — will try anyway  ({resolved})"
        return False, (f"no reachable desktop  ({resolved or 'nothing found'}). "
                       f"Ensure the Pi is logged into its desktop, or set "
                       f"pi_live.display: '' for a headless preview PNG.")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def _venv_activate_snippet(venv: str = "auto") -> str:
    """Bash that activates the Pi's virtualenv (so cv2/torch are importable).

    "auto" tries the common names (.venv, venv, env); otherwise the given path
    is activated directly. Emits a warning to stderr if nothing is found, since
    running the system python usually means a missing-module error.
    """
    if venv and venv != "auto":
        v = venv.rstrip("/")
        return (f'if [ -f "{v}/bin/activate" ]; then . "{v}/bin/activate"; '
                f'else echo "warning: no venv at {v}" >&2; fi; ')
    return (
        'for v in .venv venv env; do '
        'if [ -f "$v/bin/activate" ]; then . "$v/bin/activate"; break; fi; '
        'done; '
        'if [ -z "$VIRTUAL_ENV" ]; then '
        'echo "warning: no .venv/venv/env found - using system python" >&2; fi; '
    )


def remote_live_shell_command(repo: str, source: str = "picamera",
                              display: str = "auto", venv: str = "auto") -> str:
    """Bash one-liner run on the Pi after SSH.

    Activates the Pi's virtualenv, resolves the desktop DISPLAY/XAUTHORITY so
    the OpenCV window shows on a monitor attached to the Pi (or saves a preview
    PNG when ``display`` is empty).
    """
    repo = repo.rstrip("/")
    act = _venv_activate_snippet(venv)
    disp = "" if not display else _gui_env_snippet(display)
    tail = f"--source {source}".strip()
    return (
        f"cd {repo} && mkdir -p outputs && "
        f"{act}"
        f"{disp}"
        f"(python3 live.py {tail} || python live.py {tail})"
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
    """Sync model, launch live.py on the Pi over SSH. Returns error text or None."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    s = load_pi_live_settings(root)
    host = str(s["ssh_host"])
    repo = str(s["repo"])
    source = str(s["source"])
    display = str(s.get("display", "auto"))
    venv = str(s.get("venv", "auto"))

    ok, detail = ssh_reachable(host)
    if not ok:
        return ssh_setup_help(host, detail)

    err = sync_model_to_pi(root, host, repo)
    if err:
        return err

    remote_cmd = remote_live_shell_command(repo, source, display, venv)
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
        display = str(s.get("display", "auto"))
        ok, detail = ssh_reachable(host)
        if not ok:
            print(ssh_setup_help(host, detail), file=sys.stderr)
            return 1
        print(f"SSH OK: {host}")
        disp_ok, disp_detail = remote_display_status(host, display)
        marker = "OK" if disp_ok else "!!"
        print(f"Display {marker}: {disp_detail}")
        if not disp_ok:
            print("\nThe camera window needs somewhere to show. Options:\n"
                  "  - Attach a monitor to the Pi (its desktop must be running), or\n"
                  "  - Set  pi_live.display: ''  for a headless preview PNG.",
                  file=sys.stderr)
        return 0 if disp_ok else 2
    err = run_live_on_pi(root)
    if err:
        print(err, file=sys.stderr)
        return 1
    print(f"Opened Pi live testing (ssh {load_pi_live_settings(root)['ssh_host']}).")
    print("The RAW | DENOISED window opens on the monitor attached to the Pi.")
    print("Press q or ESC in that window to stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
