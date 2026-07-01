"""Raspberry Pi CSI camera diagnostics and no-sudo setup guidance.

On Raspberry Pi OS, ``python3-picamera2`` is usually *already installed* at the
system level (no apt/sudo needed to add it). The common problem is a virtualenv
that hides system packages — fix with ``--system-site-packages``.

When picamera2 truly isn't importable, ``live.py`` can still use the preinstalled
``rpicam-vid`` / ``libcamera-vid`` CLI (also shipped with Pi OS, no extra apt).
"""

from __future__ import annotations

import shutil
import site
import subprocess
import sys
from pathlib import Path


def on_raspberry_pi() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        model = Path("/proc/device-tree/model")
        if model.exists():
            return b"raspberry" in model.read_bytes().lower()
    except Exception:  # noqa: BLE001
        pass
    return False


def venv_uses_system_site_packages() -> bool:
    """True when this interpreter's venv can see apt-installed packages."""
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if not cfg.exists():
        return True  # not a venv
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("include-system-site-packages"):
                return "true" in line.lower().split("=", 1)[-1]
    except Exception:  # noqa: BLE001
        pass
    return False


def _system_picamera2_dirs() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    for base in ("/usr/lib/python3/dist-packages",
                 "/usr/local/lib/python3/dist-packages"):
        roots.append(Path(base))
    out: list[Path] = []
    for r in roots:
        p = r / "picamera2"
        if p.is_dir():
            out.append(p)
    return out


def find_rpicam_tool(*names: str) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def list_cameras_text() -> str | None:
    for cmd in ("rpicam-hello", "libcamera-hello"):
        exe = shutil.which(cmd)
        if not exe:
            continue
        try:
            r = subprocess.run(
                [exe, "--list-cameras"], capture_output=True, text=True,
                timeout=12, check=False)
            if r.stdout.strip():
                return r.stdout
        except Exception:  # noqa: BLE001
            pass
    return None


def diagnose() -> dict:
    """Return a structured report of Pi camera options available *right now*."""
    d: dict = {
        "on_pi": on_raspberry_pi(),
        "picamera2_importable": False,
        "picamera2_system_dir": None,
        "venv_system_site_packages": venv_uses_system_site_packages(),
        "rpicam_vid": find_rpicam_tool("rpicam-vid", "libcamera-vid"),
        "rpicam_hello": find_rpicam_tool("rpicam-hello", "libcamera-hello"),
        "gstreamer_libcamera": bool(shutil.which("gst-inspect-1.0") and
                                    _gst_has_element("libcamerasrc")),
        "opencv_gstreamer": False,
        "v4l2_devices": [],
        "cameras_listing": None,
        "recommendations": [],
    }
    try:
        import picamera2  # noqa: F401
        d["picamera2_importable"] = True
    except Exception:  # noqa: BLE001
        pass
    sys_dirs = _system_picamera2_dirs()
    if sys_dirs:
        d["picamera2_system_dir"] = str(sys_dirs[0])
    for dev in Path("/dev").glob("video*"):
        if dev.is_char_device() or dev.exists():
            d["v4l2_devices"].append(str(dev))
    d["v4l2_devices"].sort()
    if d["on_pi"]:
        d["cameras_listing"] = list_cameras_text()
    try:
        import cv2
        d["opencv_gstreamer"] = "GStreamer" in (cv2.getBuildInformation() or "")
    except Exception:  # noqa: BLE001
        pass
    d["recommendations"] = _recommendations(d)
    return d


def _gst_has_element(name: str) -> bool:
    gst = shutil.which("gst-inspect-1.0")
    if not gst:
        return False
    try:
        r = subprocess.run([gst, name], capture_output=True, text=True,
                             timeout=8, check=False)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _recommendations(d: dict) -> list[str]:
    if not d["on_pi"]:
        return ["Not on a Raspberry Pi — use a USB webcam (OpenCV) or --source sim."]
    rec: list[str] = []
    if d["picamera2_importable"]:
        rec.append("picamera2 is ready — live testing can use the CSI camera.")
        return rec
    if d["picamera2_system_dir"] and not d["venv_system_site_packages"]:
        rec.append(
            "NO SUDO NEEDED: picamera2 is already on the system image, but your "
            "venv cannot see it. Recreate the venv:\n"
            "    deactivate\n"
            "    rm -rf .venv\n"
            "    python3 -m venv --system-site-packages .venv\n"
            "    source .venv/bin/activate\n"
            "    pip install -r requirements.txt")
        rec.append(
            "Or flip one line in .venv/pyvenv.cfg:\n"
            "    include-system-site-packages = true")
    elif not d["picamera2_system_dir"]:
        rec.append(
            "picamera2 not found on this image. Without sudo apt, try pip-only "
            "(Bookworm Pi):\n"
            "    pip install -r requirements-pi.txt")
    if d["rpicam_vid"]:
        rec.append(
            f"Fallback: {Path(d['rpicam_vid']).name} is installed — live.py will "
            "use it automatically when picamera2 is unavailable (no sudo).")
    if d["gstreamer_libcamera"] and d["opencv_gstreamer"]:
        rec.append("GStreamer libcamerasrc is available as another CSI fallback.")
    if not rec:
        rec.append(
            "No CSI backend detected. Ask an admin to run:\n"
            "    sudo apt install -y python3-picamera2\n"
            "or use a USB webcam (--source opencv).")
    return rec


def format_report(d: dict | None = None) -> str:
    d = d or diagnose()
    lines = ["Raspberry Pi camera diagnostics", "=" * 32]
    lines.append(f"On Pi hardware:        {d['on_pi']}")
    lines.append(f"picamera2 importable:  {d['picamera2_importable']}")
    if d.get("picamera2_system_dir"):
        lines.append(f"picamera2 on system:   {d['picamera2_system_dir']}")
    lines.append(f"venv sees system pkgs: {d['venv_system_site_packages']}")
    lines.append(f"rpicam-vid / libcamera: {d.get('rpicam_vid') or '—'}")
    lines.append(f"GStreamer libcamerasrc: {d.get('gstreamer_libcamera')}")
    lines.append(f"OpenCV + GStreamer:    {d.get('opencv_gstreamer')}")
    if d.get("v4l2_devices"):
        lines.append(f"V4L2 devices:          {', '.join(d['v4l2_devices'])}")
    if d.get("cameras_listing"):
        lines.append("")
        lines.append(d["cameras_listing"].strip())
    lines.append("")
    lines.append("Recommendations:")
    for i, r in enumerate(d.get("recommendations") or [], 1):
        lines.append(f"  {i}. {r}")
    return "\n".join(lines)
