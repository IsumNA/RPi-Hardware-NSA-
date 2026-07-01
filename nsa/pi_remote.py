"""Run live camera testing on a Raspberry Pi over SSH.

When you compile on the AI server and click LIVE TEST in the GUI, this syncs
``outputs/model.pt`` to the Pi and opens an SSH terminal that runs
``live.py --source picamera`` on Pi hardware (where the CSI camera works).
"""

from __future__ import annotations

import os
import shutil
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


def remote_live_shell_command(repo: str, source: str = "picamera") -> str:
    """Bash one-liner run on the Pi after SSH."""
    repo = repo.rstrip("/")
    return (
        f"cd {repo} && mkdir -p outputs && "
        f"if [ -d .venv/bin ]; then . .venv/bin/activate; fi && "
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


def launch_pi_terminal(host: str, remote_cmd: str) -> str | None:
    """Open a new terminal running ``ssh host 'remote_cmd'`` (Linux)."""
    ssh_args = ["ssh", "-t", host, remote_cmd]
    for spec in (
        ["gnome-terminal", "--", *ssh_args],
        ["konsole", "-e", *ssh_args],
        ["xfce4-terminal", "-e", " ".join(ssh_args)],
        ["xterm", "-e", *ssh_args],
    ):
        exe = shutil.which(spec[0])
        if exe:
            subprocess.Popen([exe, *spec[1:]])
            return None
    try:
        subprocess.Popen(ssh_args)
        return None
    except OSError as exc:
        return f"Could not launch SSH terminal: {exc}"


def run_live_on_pi(project_root: Path | None = None) -> str | None:
    """Sync model, open Pi terminal with live.py. Returns error text or None."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    s = load_pi_live_settings(root)
    host = str(s["ssh_host"])
    repo = str(s["repo"])
    source = str(s["source"])

    ok, detail = ssh_reachable(host)
    if not ok:
        return ssh_setup_help(host, detail)

    err = sync_model_to_pi(root, host, repo)
    if err:
        return err

    remote_cmd = remote_live_shell_command(repo, source)
    return launch_pi_terminal(host, remote_cmd)


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
        ok, detail = ssh_reachable(host)
        if ok:
            print(f"SSH OK: {host}")
            return 0
        print(ssh_setup_help(host, detail), file=sys.stderr)
        return 1
    err = run_live_on_pi(root)
    if err:
        print(err, file=sys.stderr)
        return 1
    print(f"Opened Pi live testing (ssh {load_pi_live_settings(root)['ssh_host']}).")
    print("Press q or ESC in the Pi window to stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
