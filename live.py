#!/usr/bin/env python3
"""NSA Live Testing
==================
Run the compiled denoiser on a live camera stream and show the raw sensor feed
next to the denoised output in real time — the on-Pi proof that the optimization
actually cleans up the low-light camera.

Camera backends are tried in order (override with --source):
  1. picamera2  — Python CSI API (usually pre-installed on Pi OS; venv needs
                  --system-site-packages, NO sudo apt required)
  2. rpicam-vid — Pi CSI via the preinstalled libcamera CLI (no picamera2)
  3. GStreamer  — libcamerasrc when OpenCV was built with GStreamer
  4. OpenCV     — USB / V4L2 webcam (--camera-index N)
  5. simulated  — synthetic low-light stream for dev machines

  Run ``python pi_camera_check.py`` on the Pi to see what works without sudo.

It loads the exact model trained by the last compile (outputs/model.pt). If that
checkpoint is missing it rebuilds from flags and does a quick calibration so the
window still shows real denoising.

Examples
--------
  # After a compile, just run it (auto-detects camera, loads outputs/model.pt):
  python live.py

  # Force the Pi camera at high gain (low-light) and 720p:
  python live.py --source picamera --cam-gain 8 --width 1280 --height 720

  # Force a USB webcam:
  python live.py --source opencv --camera-index 0

  # No camera? Demo on the synthetic low-light stream:
  python live.py --source sim --sensor imx662 --gain 512
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from nsa.config import ModelConfig
from nsa.inference import to_image, to_tensor
from nsa.models import build_model, count_params
from nsa.raw_io import _capture, _synthetic_scene
from nsa.sensors import SENSOR_KEYS, get_sensor
from nsa.theme import RPI_GREEN, RPI_RASPBERRY, banner, console, log

OUT = Path("outputs")
CKPT = OUT / "model.pt"

# BGR colours for the OpenCV overlay (OpenCV is BGR-ordered).
_RASP_BGR = (74, 26, 197)      # raspberry red
_GREEN_BGR = (74, 192, 108)
_WHITE = (245, 245, 245)
_DARK = (40, 40, 40)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args) -> tuple[torch.nn.Module, dict]:
    """Load the last-compiled model, or rebuild + quick-calibrate from flags."""
    if CKPT.exists() and not args.fresh:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m = ck.get("model", {})
        if m.get("hf_imgutils") and m.get("hf_variant"):
            from nsa.hf_runner import ImgutilsNafnetDenoiser
            model = ImgutilsNafnetDenoiser(m["hf_variant"])
            model.eval()
            log(f"Loaded Hugging Face NAFNet ({m['hf_variant']}) via imgutils", "ok")
            ck.setdefault("sensor", args.sensor)
            ck.setdefault("gain", args.gain)
            return model, ck
        hf_onnx = m.get("hf_onnx")
        if hf_onnx and Path(hf_onnx).is_file():
            from nsa.hf_runner import OnnxDenoiser
            model = OnnxDenoiser(Path(hf_onnx))
            model.eval()
            log(f"Loaded Hugging Face ONNX: {Path(hf_onnx).name}", "ok")
            ck.setdefault("sensor", args.sensor)
            ck.setdefault("gain", args.gain)
            return model, ck
        if m.get("hf_model"):
            from nsa.hf_runner import load_hf_model
            cfg = ModelConfig(
                model_family=m.get("family", "nafnet"),
                base_channels=m.get("base_channels", 32),
                block_depth=m.get("block_depth", 4),
                conv_type=m.get("conv_type", "depthwise"),
                activation=m.get("activation", "relu"),
                nafnet_enc_blocks=list(m.get("nafnet_enc", []) or []),
                nafnet_middle_blocks=m.get("nafnet_middle", 1),
                nafnet_dec_blocks=list(m.get("nafnet_dec", []) or []),
                hf_model=m.get("hf_model"),
                hf_weight=m.get("hf_weight"),
            )
            model, spec = load_hf_model(
                m["hf_model"], cfg, weight=m.get("hf_weight"), download=False)
            log(f"Loaded Hugging Face model: {spec.model_id} "
                f"({spec.weight_path.name})", "ok")
            ck.setdefault("sensor", args.sensor)
            ck.setdefault("gain", args.gain)
            return model, ck
        cfg = ModelConfig(
            model_family=m.get("family", "nafnet"),
            base_channels=m.get("base_channels", 32),
            block_depth=m.get("block_depth", 4),
            conv_type=m.get("conv_type", "depthwise"),
            activation=m.get("activation", "relu"),
            nafnet_enc_blocks=list(m.get("nafnet_enc", []) or []),
            nafnet_middle_blocks=m.get("nafnet_middle", 1),
            nafnet_dec_blocks=list(m.get("nafnet_dec", []) or []),
        )
        model = build_model(cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        log(f"Loaded compiled model: {cfg.model_family.upper()} "
            f"({count_params(model):,} params) from {CKPT}", "ok")
        ck.setdefault("sensor", args.sensor)
        ck.setdefault("gain", args.gain)
        return model, ck

    # No checkpoint — rebuild from flags and quick-calibrate on synthetic data.
    from nsa.inference import calibrate_multi
    from nsa.raw_io import build_frame
    log("No outputs/model.pt found — rebuilding from flags and quick-calibrating "
        "on synthetic frames…", "warn")
    cfg = ModelConfig(model_family=args.model_family, base_channels=args.base_channels,
                      block_depth=args.block_depth, conv_type=args.conv_type,
                      activation=args.activation)
    model = build_model(cfg)
    sensor = get_sensor(args.sensor)
    frames = [build_frame(None, args.gain, 64, 192, sensor, args.seed + i)
              for i in range(3)]
    pairs = [(f.noisy_rgb, f.clean_rgb) for f in frames]
    calibrate_multi(model, pairs, max(args.calibrate, 40), args.seed)
    model.eval()
    log(f"Quick-calibrated {cfg.model_family.upper()} "
        f"({count_params(model):,} params)", "ok")
    return model, {"model": {"family": cfg.model_family}, "sensor": args.sensor,
                   "gain": args.gain}


# ---------------------------------------------------------------------------
# Camera backends — each exposes .read() -> BGR uint8 frame, and .close()
# ---------------------------------------------------------------------------
def _on_raspberry_pi() -> bool:
    """True only on real Pi hardware (not every Linux box)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        model = Path("/proc/device-tree/model")
        if model.exists():
            return b"raspberry" in model.read_bytes().lower()
    except Exception:  # noqa: BLE001
        pass
    return False


def _picamera2_available() -> bool:
    """picamera2 is usually pre-installed on Pi OS; venv may need system-site-packages."""
    if not _on_raspberry_pi():
        return False
    try:
        import picamera2  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _picamera2_setup_hint() -> str:
    try:
        from nsa.pi_camera import format_report, diagnose
        d = diagnose()
        if d.get("recommendations"):
            return d["recommendations"][0]
        return format_report(d)
    except Exception:  # noqa: BLE001
        return (
            "picamera2 not importable. On Pi OS it is usually already installed — "
            "recreate your venv with: python3 -m venv --system-site-packages .venv"
        )


class PiCam:
    name = "Raspberry Pi CSI camera (picamera2)"

    def __init__(self, args):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        config = self.cam.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"})
        self.cam.configure(config)
        controls = {"AnalogueGain": float(args.cam_gain)}
        if args.exposure > 0:
            controls["ExposureTime"] = int(args.exposure)
            controls["AeEnable"] = False
        try:
            self.cam.set_controls(controls)
        except Exception:  # noqa: BLE001
            pass
        self.cam.start()
        time.sleep(0.6)

    def read(self):
        # picamera2 'RGB888' returns a BGR-ordered ndarray (ready for OpenCV).
        return self.cam.capture_array()

    def close(self):
        try:
            self.cam.stop()
        except Exception:  # noqa: BLE001
            pass


class RpicamVidCam:
    """CSI camera via the preinstalled rpicam-vid / libcamera-vid CLI (no picamera2)."""

    def __init__(self, args, exe: str):
        self.exe = exe
        self.name = f"Raspberry Pi CSI ({Path(exe).name} pipe)"
        cmd = [
            exe, "-t", "0", "--codec", "mjpeg", "--inline", "-o", "-", "-n",
            "--width", str(args.width), "--height", str(args.height),
            "--framerate", "30",
        ]
        if args.cam_gain and args.cam_gain > 1:
            cmd += ["--gain", str(float(args.cam_gain))]
        self._buf = b""
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def read(self):
        if self.proc.poll() is not None:
            return None
        while True:
            chunk = self.proc.stdout.read(8192)
            if not chunk:
                return None
            self._buf += chunk
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass


class GStreamerPiCam:
    """CSI camera via GStreamer libcamerasrc (when OpenCV is built with GStreamer)."""

    name = "Raspberry Pi CSI (GStreamer libcamerasrc)"

    def __init__(self, args):
        pipeline = (
            f"libcamerasrc ! video/x-raw,width={args.width},height={args.height},"
            f"format=RGB ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("GStreamer libcamerasrc pipeline failed to open")
        if _warm_read(self.cap, attempts=25) is None:
            self.cap.release()
            raise RuntimeError("GStreamer opened but delivered no frames")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass


def _try_rpicam_vid(args):
    from nsa.pi_camera import find_rpicam_tool
    exe = find_rpicam_tool("rpicam-vid", "libcamera-vid")
    if not exe:
        return None
    try:
        cam = RpicamVidCam(args, exe)
        # Verify the pipe actually delivers a frame.
        frame = cam.read()
        if frame is None:
            cam.close()
            return None
        cam._buf = b""  # consumed test frame; stream continues
        return cam
    except Exception:  # noqa: BLE001
        return None


def _try_gstreamer_pi(args):
    if not _on_raspberry_pi():
        return None
    try:
        if "GStreamer" not in (cv2.getBuildInformation() or ""):
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        from nsa.pi_camera import _gst_has_element
        if not _gst_has_element("libcamerasrc"):
            return None
        return GStreamerPiCam(args)
    except Exception:  # noqa: BLE001
        return None


class CvCam:
    def __init__(self, cap, args, index=0):
        self.cap = cap
        self.index = index
        self.name = f"USB / webcam (OpenCV, index {index})"
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Some webcams drop an occasional frame; retry briefly before giving up.
            for _ in range(3):
                time.sleep(0.01)
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    return frame
            return None
        return frame

    def close(self):
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass


def _cv_backends() -> list:
    """Platform-appropriate OpenCV capture backends, best first.

    On Windows the default (MSMF) backend frequently opens a device but never
    delivers a frame; DirectShow is far more reliable, so we try it first.
    """
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def _warm_read(cap, attempts: int = 20, delay: float = 0.08):
    """Read until a real frame arrives (webcams often need a warm-up)."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(delay)
    return None


def _open_cv_capture(index: int, args):
    """Open a webcam at ``index``, trying each backend and verifying it streams.

    Returns an opened, frame-delivering ``cv2.VideoCapture`` or ``None``.
    """
    for be in _cv_backends():
        cap = None
        try:
            cap = cv2.VideoCapture(index, be)
        except Exception:  # noqa: BLE001
            continue
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        # Pass 1: honour requested resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if _warm_read(cap) is not None:
            return cap
        # Pass 2: some drivers only stream at the native default resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if _warm_read(cap) is not None:
            return cap
        # Pass 3: don't touch resolution at all.
        cap.release()
        try:
            cap = cv2.VideoCapture(index, be)
        except Exception:  # noqa: BLE001
            continue
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        if _warm_read(cap) is not None:
            return cap
        cap.release()
    return None


def probe_cameras(max_index: int = 9, width: int = 640, height: int = 480) -> list[int]:
    """Return webcam indices that deliver at least one frame (OpenCV only)."""
    args = make_args(width=width, height=height)
    found: list[int] = []
    for idx in range(max_index + 1):
        cap = _open_cv_capture(idx, args)
        if cap is not None:
            found.append(idx)
            cap.release()
    return found


# Human-readable hint when no camera is found (shown in the GUI).
_CAMERA_HELP = (
    "No camera found. On a Pi CSI module without sudo apt: run "
    "python pi_camera_check.py — picamera2 is usually already on Pi OS; "
    "recreate the venv with --system-site-packages, or live.py will try "
    "rpicam-vid automatically. USB webcams use OpenCV (index 0–9)."
)


class SimCam:
    name = "simulated low-light stream (sensor noise model)"

    def __init__(self, args, sensor_key, gain):
        self.sensor = get_sensor(sensor_key)
        self.gain = gain
        self.h, self.w = args.height, args.width
        # A wider clean scene so we can pan across it for apparent motion.
        self.scene = _synthetic_scene(self.h, int(self.w * 1.5), args.seed)
        self.prnu = np.random.default_rng(7).normal(
            1.0, self.sensor.prnu, (self.h, self.w, 3)).astype(np.float32)
        self.rng = np.random.default_rng(args.seed)
        self.x = 0
        self.dx = 2
        self.max_x = self.scene.shape[1] - self.w

    def read(self):
        self.x += self.dx
        if self.x <= 0 or self.x >= self.max_x:
            self.dx = -self.dx
            self.x = max(0, min(self.x, self.max_x))
        crop = self.scene[:, self.x:self.x + self.w, :]
        noisy = _capture(crop, self.gain, self.sensor, self.rng, self.prnu)
        return cv2.cvtColor((noisy * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def close(self):
        pass


def open_camera(args, sensor_key, gain):
    src = args.source
    if src in ("auto", "picamera") and _picamera2_available():
        try:
            cam = PiCam(args)
            log(f"Camera: {cam.name}", "ok")
            return cam
        except Exception as exc:  # noqa: BLE001
            if src == "picamera":
                raise SystemExit(f"picamera2 backend failed: {exc}")
            log(f"picamera2 unavailable ({exc}); trying Pi CSI fallbacks…", "warn")
    elif src == "picamera":
        raise SystemExit(_picamera2_setup_hint())

    # Pi CSI without picamera2: rpicam-vid / libcamera-vid (preinstalled on Pi OS).
    if src in ("auto", "rpicam", "picamera") and _on_raspberry_pi():
        cam = _try_rpicam_vid(args)
        if cam is not None:
            log(f"Camera: {cam.name}", "ok")
            return cam
        if src == "rpicam":
            raise SystemExit(
                "rpicam-vid / libcamera-vid not found or not streaming. "
                "Run: python pi_camera_check.py")

    if src in ("auto", "gstreamer", "picamera") and _on_raspberry_pi():
        try:
            cam = _try_gstreamer_pi(args)
            if cam is not None:
                log(f"Camera: {cam.name}", "ok")
                return cam
        except Exception as exc:  # noqa: BLE001
            if src == "gstreamer":
                raise SystemExit(f"GStreamer CSI camera failed: {exc}") from exc
            log(f"GStreamer CSI unavailable ({exc})", "warn")

    if src in ("auto", "opencv"):
        # Probe the requested index first, then 0..max (many laptops use index 1).
        max_idx = max(9, int(getattr(args, "camera_index", 0)))
        indices = [args.camera_index] + [i for i in range(max_idx + 1)
                                         if i != args.camera_index]
        for idx in indices:
            cap = _open_cv_capture(idx, args)
            if cap is not None:
                cam = CvCam(cap, args, idx)
                log(f"Camera: {cam.name}", "ok")
                return cam
            if src == "opencv":
                break
        if src == "opencv":
            raise SystemExit(
                f"no working OpenCV camera at index {args.camera_index} "
                f"(tried backends: DSHOW/MSMF/V4L2). Is another app using it?")
        log("No physical camera detected — using the simulated low-light stream.",
            "warn")
        log(_CAMERA_HELP, "warn")

    cam = SimCam(args, sensor_key, gain)
    log(f"Camera: {cam.name}  [{sensor_key} @ {gain}×]", "ok")
    return cam


def make_args(**overrides):
    """Build a default argument namespace (for embedding live testing in the GUI)."""
    args = build_parser().parse_args([])
    env_idx = os.environ.get("NSA_CAMERA_INDEX")
    if env_idx and "camera_index" not in overrides:
        try:
            setattr(args, "camera_index", int(env_idx))
        except ValueError:
            pass
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Inference + metrics
# ---------------------------------------------------------------------------
def denoise_bgr(model, bgr: np.ndarray) -> tuple[np.ndarray, float]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = to_tensor(rgb)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    out_rgb = to_image(out)
    out_bgr = cv2.cvtColor((np.clip(out_rgb, 0, 1) * 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
    return out_bgr, dt_ms


def noise_level(bgr: np.ndarray) -> float:
    """High-frequency noise estimate: std of the luminance Laplacian."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).std())


def _label(img: np.ndarray, text: str, color) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), _DARK, -1)
    cv2.putText(img, text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)


def _stat_line(img: np.ndarray, lines: list[str]) -> None:
    h = img.shape[0]
    y0 = h - 12 - 22 * (len(lines) - 1)
    cv2.rectangle(img, (0, y0 - 22), (img.shape[1], h), _DARK, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (12, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _WHITE, 1, cv2.LINE_AA)


def compose(raw_bgr, out_bgr, fps, dt_ms, noise_in, noise_out, model_name) -> np.ndarray:
    left = raw_bgr.copy()
    right = out_bgr.copy()
    _label(left, "RAW SENSOR (noisy)", _WHITE)
    _label(right, "NSA DENOISED", _GREEN_BGR)
    drop = (1.0 - noise_out / noise_in) * 100.0 if noise_in > 1e-6 else 0.0
    _stat_line(left, [f"noise {noise_in:5.1f}"])
    _stat_line(right, [
        f"{model_name}   {dt_ms:4.1f} ms/frame   {fps:4.1f} FPS",
        f"noise {noise_out:5.1f}   (-{max(drop,0):4.1f}% vs raw)",
    ])
    sep = np.full((left.shape[0], 3, 3), _RASP_BGR, np.uint8)
    return np.hstack([left, sep, right])


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live.py",
        description="NSA live testing — run the compiled denoiser on a camera stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--source",
                   choices=["auto", "picamera", "rpicam", "gstreamer", "opencv", "sim"],
                   default="auto", help="camera backend (default: auto-detect)")
    p.add_argument("--camera-index", dest="camera_index", type=int, default=0,
                   help="OpenCV/V4L2 camera index (default: 0)")
    p.add_argument("--width", type=int, default=640, help="capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="capture height (default: 480)")
    p.add_argument("--cam-gain", dest="cam_gain", type=float, default=8.0,
                   help="picamera2 analogue gain for low light (default: 8.0)")
    p.add_argument("--exposure", type=int, default=0,
                   help="picamera2 exposure time in µs (0 = auto)")
    p.add_argument("--sensor", choices=list(SENSOR_KEYS), default="imx662",
                   help="sensor profile for the simulated stream (default: imx662)")
    p.add_argument("--gain", type=int, default=512,
                   help="analog gain for the simulated stream (default: 512)")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="auto-stop after N seconds (0 = run until 'q'/ESC)")
    p.add_argument("--seed", type=int, default=662)
    p.add_argument("--fresh", action="store_true",
                   help="ignore outputs/model.pt and rebuild from the flags below")
    p.add_argument("--calibrate", type=int, default=120,
                   help="quick-calibration steps when no checkpoint exists")
    # Rebuild flags (only used with --fresh or when no checkpoint exists).
    p.add_argument("--model-family", dest="model_family", default="nafnet")
    p.add_argument("--base-channels", dest="base_channels", type=int, default=32)
    p.add_argument("--block-depth", dest="block_depth", type=int, default=4)
    p.add_argument("--conv-type", dest="conv_type", default="depthwise")
    p.add_argument("--activation", default="relu")
    return p


def main() -> int:
    args = build_parser().parse_args()
    banner("Live Testing  ·  raw vs denoised")

    model, ck = load_model(args)
    model_name = str(ck.get("model", {}).get("family", "model")).upper()
    sensor_key = ck.get("sensor", args.sensor)
    gain = int(ck.get("gain", args.gain))

    cam = open_camera(args, sensor_key, gain)

    win = "NSA Live Testing — RAW  |  DENOISED   (press q or ESC to quit)"
    headless = False
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except Exception:  # noqa: BLE001
        headless = True
        log("No display available — will save a sample composite instead.", "warn")

    log("Streaming… press 'q' or ESC in the window to stop.", "step")
    fps = 0.0
    t_prev = time.perf_counter()
    t_start = t_prev
    saved = None
    try:
        while True:
            raw = cam.read()
            if raw is None:
                log("Camera returned no frame — stopping.", "warn")
                break
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            if raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            out, dt_ms = denoise_bgr(model, raw)
            n_in, n_out = noise_level(raw), noise_level(out)

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            t_prev = now

            canvas = compose(raw, out, fps, dt_ms, n_in, n_out, model_name)
            saved = canvas

            if headless:
                if now - t_start > 2.0:      # grab ~2s worth, then stop
                    break
            else:
                cv2.imshow(win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:  # noqa: BLE001
                    pass
            if args.seconds > 0 and now - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    if saved is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        shot = OUT / "live_preview.png"
        cv2.imwrite(str(shot), saved)
        log(f"Saved a side-by-side sample -> {shot}", "ok")
    log("Live testing finished.", "ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print(f"\n[bold {RPI_RASPBERRY}]Live testing aborted.[/]")
        sys.exit(130)
