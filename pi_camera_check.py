#!/usr/bin/env python3
"""Check Raspberry Pi CSI camera options without needing sudo.

Run this on the Pi when live testing falls back to the simulated stream:

    python pi_camera_check.py

It reports whether picamera2 is already on the system image (most Pi OS installs),
whether your virtualenv can see it, and whether rpicam-vid / GStreamer fallbacks
are available — plus exact no-sudo fix steps.
"""

from __future__ import annotations

import sys

from nsa.pi_camera import diagnose, format_report
from nsa.theme import banner, console, log


def main() -> int:
    banner("Pi camera check  ·  no sudo required")
    d = diagnose()
    console.print(format_report(d))
    console.print()
    if d.get("picamera2_importable") or d.get("rpicam_vid"):
        log("At least one CSI backend is available for live testing.", "ok")
        return 0
    log("No CSI backend ready yet — follow the recommendations above.", "warn")
    return 1


if __name__ == "__main__":
    sys.exit(main())
