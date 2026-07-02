#!/usr/bin/env python3
"""NSA Compiler - desktop UI.

A Raspberry Pi Imager-styled front-end for the 6-Level Optimization Stack:
clean white surface, raspberry-red accents, the official Raspberry Pi logo,
a rounded minimal sans typeface, and a live pipeline progress sidebar.

The UI shells out to ``run_demo.py`` so it always runs the exact same pipeline
the CLI does, streaming the live compilation log into the window.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = ImageTk = None


# -- Display scaling (readable text on Windows, Linux and macOS) ---------------
# Windows: derive from the real DPI. Linux/macOS: ctypes.windll doesn't exist, so
# we use a comfortable default (text was tiny before because it fell back to 1.0
# AND "Segoe UI" is absent on Linux). Override anytime with NSA_UI_SCALE=1.5.
USE_TK_SCALING = False  # set True only on the Windows DPI path

# Default text/UI size. "Extra Large" in App Options == 1.9×; we ship that as
# the out-of-the-box default so the interface is comfortably readable everywhere.
# Override anytime with NSA_UI_SCALE, or pick another size in App Options.
BASE_SCALE = 1.9


def _detect_scale() -> float:
    global USE_TK_SCALING
    env = os.environ.get("NSA_UI_SCALE")
    if env:
        try:
            return max(0.8, min(3.0, float(env)))
        except ValueError:
            pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor v2
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            try:
                dpi = ctypes.windll.user32.GetDpiForSystem()
            except Exception:
                dpi = 96
            USE_TK_SCALING = True
            # Never smaller than "Extra Large"; go bigger still on hi-DPI displays.
            return max(BASE_SCALE, dpi / 96.0)
        except Exception:
            return BASE_SCALE
    # Linux / macOS: ship the "Extra Large" default so the UI is readable out of the box.
    return BASE_SCALE


SCALE = _detect_scale()

# Resolved at runtime (after a Tk root exists) to an installed family.
FONT_FAMILY = "Segoe UI"
_FONT_PREFS = ["Segoe UI", "Nunito", "Cantarell", "Ubuntu", "Noto Sans",
               "DejaVu Sans", "Helvetica", "Arial", "TkDefaultFont"]


def _resolve_font_family():
    """Pick the first installed preferred family (needs a Tk root to exist)."""
    global FONT_FAMILY
    try:
        available = set(tkfont.families())
        for fam in _FONT_PREFS:
            if fam in available:
                FONT_FAMILY = fam
                return
    except Exception:
        pass


def S(x: float) -> int:
    """Scale a pixel dimension for the current display."""
    return int(round(x * SCALE))


def FT(size: float) -> int:
    """Scale a font point size for the current display."""
    return int(round(size * SCALE))


def font(size: float, weight: str = "normal", family: str | None = None):
    family = family or FONT_FAMILY
    if weight == "normal":
        return (family, FT(size))
    return (family, FT(size), weight)


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "rpi_logo.png"
PANEL_PATH = ROOT / "outputs" / "validation_panel.png"

# Image-sensor catalogue shown as the primary input. Each card carries a
# transparent product picture so the operator can see exactly which module
# they're targeting (keeps IMX219 / IMX662 / IMX-NG from getting mixed up).
SENSOR_CARDS = [
    {
        "key": "imx219",
        "name": "IMX219",
        "family": "Legacy CMOS",
        "tag": "LEGACY",
        "image": ROOT / "assets" / "sensor_imx219.png",
        "blurb": "Camera Module v2. High read noise, messy chroma splotches — "
                 "needs a deeper denoiser.",
        "specs": "QE 55%  ·  read 4.0e\u207b  ·  10-bit RGGB",
    },
    {
        "key": "imx662",
        "name": "IMX662",
        "family": "Starvis 2",
        "tag": "CURRENT",
        "image": ROOT / "assets" / "sensor_imx662.png",
        "blurb": "Current Starvis 2. Low read noise, mostly photon-shot limited "
                 "— a lean network cleans it well.",
        "specs": "QE 80%  ·  read 2.0e\u207b  ·  12-bit RGGB",
    },
    {
        "key": "imxng",
        "name": "IMX-NG",
        "family": "Starvis 2 · unreleased",
        "tag": "FUTURE",
        "image": ROOT / "assets" / "sensor_imxng.png",
        "blurb": "Unreleased next-gen low-light. Shot-noise dominated and very "
                 "uniform — optimise before silicon ships.",
        "specs": "QE 92%  ·  read 0.8e\u207b  ·  12-bit RGGB",
    },
]

# Curated, project-relevant Hugging Face categories so the browser can show a
# ready-made list of denoising / low-light / restoration models without the user
# having to think up a search query. (label -> search query, pipeline tag).
HF_CATEGORIES = [
    ("NAFNet restoration (recommended)", "deepghs image_restoration", "image-to-image"),
    ("Low-light enhancement", "low-light", "image-to-image"),
    ("Image denoising", "denoise", "image-to-image"),
    ("Image restoration", "restoration", "image-to-image"),
    ("Super-resolution", "super-resolution", "image-to-image"),
    ("Image-to-image (all)", "", "image-to-image"),
    ("Denoisers (any task)", "denoise", ""),
]
HF_CATEGORY_MAP = {label: (q, t) for label, q, t in HF_CATEGORIES}

# -- Raspberry Pi Imager palette ----------------------------------------------
WHITE = "#FFFFFF"
INK = "#2B2B2B"
SUBTLE = "#8C8C8C"
RASPBERRY = "#C51A4A"
RASPBERRY_DK = "#A50F37"
GREEN = "#6CC04A"
LINE = "#E4E4E4"
FIELD = "#F4F4F4"
HOVER = "#F2F2F2"
AMBER = "#C98A1B"

# Fitness rating -> colour (clear words: OPTIMAL > STRONG > FAIR > WEAK).
GRADE_COLORS = {"OPTIMAL": GREEN, "STRONG": "#3F9142", "FAIR": AMBER,
                "WEAK": RASPBERRY}

# Per-chip suitability verdict -> colour + short label.
VERDICT_COLORS = {"SUITABLE": GREEN, "CAVEATS": AMBER, "UNSUITABLE": RASPBERRY}
VERDICT_LABEL = {"SUITABLE": "RUNS WELL", "CAVEATS": "WITH CAVEATS",
                 "UNSUITABLE": "NOT REC."}
VERDICT_RANK = {"SUITABLE": 0, "CAVEATS": 1, "UNSUITABLE": 2}
CHIP_LABEL = {"all": "ALL CHIPS", "rpi5_cpu": "PI 5 CPU",
              "hailo8": "HAILO-8", "deepx": "DEEPX"}
SENSOR_SHORT = {"imx219": "219", "imx662": "662", "imxng": "NG", "all": "all"}

LEVELS = [
    ("1", "Sensor"),
    ("2", "Data / GT"),
    ("3", "Architecture"),
    ("4", "Compiler"),
    ("5", "Calibration"),
    ("6", "Export"),
    ("✓", "Report"),
]


def _round_points(x1, y1, x2, y2, r):
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


class RoundButton(tk.Canvas):
    """Flat rounded button matching the Imager's primary/secondary styles."""

    def __init__(self, parent, text, command, kind="primary", width=170, height=44):
        self._is_hero = kind == "hero"
        # Hero is slightly larger than primary, but must not blow past the content area.
        if self._is_hero:
            w, h = S(min(width, 210)), S(min(height, 48))
        else:
            w, h = S(width), S(height)
        super().__init__(parent, width=w, height=h, bg=parent["bg"],
                         highlightthickness=0, bd=0)
        self.command = command
        self.kind = kind
        self.text = text
        self.w, self.h = w, h
        self._enabled = True
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda e: self._draw(hover=True))
        self.bind("<Leave>", lambda e: self._draw(hover=False))
        self._draw()

    def _palette(self, hover):
        if self.kind == "hero":
            fill = RASPBERRY_DK if hover else RASPBERRY
            if not self._enabled:
                fill = "#E2A9B8"
            return fill, RASPBERRY_DK, "white"
        if self.kind == "primary":
            fill = RASPBERRY_DK if hover else RASPBERRY
            if not self._enabled:
                fill = "#E2A9B8"
            return fill, fill, "white"
        return (HOVER if hover else WHITE), RASPBERRY, RASPBERRY

    def _draw(self, hover=False):
        self.delete("all")
        fill, border, fg = self._palette(hover)
        r = self.h * 2 // 5 if self._is_hero else self.h // 2
        outline = 2.5 if self._is_hero else 1.5
        self.create_polygon(_round_points(2, 2, self.w - 2, self.h - 2, r),
                            smooth=True, fill=fill, outline=border, width=outline)
        fsize = 12 if self._is_hero else 11
        # Shrink label if the canvas is tight so text is not clipped at the edges.
        ft = font(fsize, "bold")
        tw = tkfont.Font(font=ft).measure(self.text)
        if tw > self.w - S(20):
            fsize = max(9, fsize - 1)
            ft = font(fsize, "bold")
        self.create_text(self.w / 2, self.h / 2, text=self.text, fill=fg,
                        font=ft)

    def _click(self, _e):
        if self._enabled and self.command:
            self.command()

    def set_enabled(self, on: bool):
        self._enabled = on
        self._draw()

    def set_text(self, text: str):
        self.text = text
        self._draw()


class Sidebar(tk.Canvas):
    """Branded, live pipeline-progress sidebar (Canvas-drawn for rounded pills)."""

    def __init__(self, parent):
        self.WIDTH = S(220)
        super().__init__(parent, width=self.WIDTH, bg=WHITE, highlightthickness=0, bd=0)
        self.states = ["pending"] * len(LEVELS)
        self._logo_img = None
        self.bind("<Configure>", lambda e: self.redraw())

    def reset(self):
        self.states = ["pending"] * len(LEVELS)
        self.redraw()

    def set_active(self, idx):
        for i in range(len(LEVELS)):
            self.states[i] = "done" if i < idx else ("active" if i == idx else "pending")
        self.redraw()

    def all_done(self):
        self.states = ["done"] * len(LEVELS)
        self.redraw()

    def redraw(self):
        self.delete("all")
        h = self.winfo_height() or S(600)
        self.create_line(self.WIDTH - 1, 0, self.WIDTH - 1, h, fill=LINE)

        if ImageTk and LOGO_PATH.exists():
            if self._logo_img is None:
                im = Image.open(LOGO_PATH).convert("RGBA").resize((S(58), S(58)), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(im)
            self.create_image(S(34), S(46), image=self._logo_img)
            tx = S(70)
        else:
            tx = S(22)
        self.create_text(tx, S(36), text="NAS", anchor="w", fill=RASPBERRY,
                        font=font(20, "bold"))
        self.create_text(tx, S(58), text="compiler", anchor="w", fill=SUBTLE,
                        font=font(10))

        self.create_text(S(22), S(108), text="PIPELINE", anchor="w", fill=RASPBERRY,
                        font=font(9, "bold"))

        y = S(134)
        step = S(46)
        rad = S(11)
        for i, (num, name) in enumerate(LEVELS):
            state = self.states[i]
            cx = S(37)
            if state == "active":
                self.create_polygon(_round_points(S(14), y - S(15), self.WIDTH - S(16), y + S(15), S(14)),
                                    smooth=True, fill=RASPBERRY, outline=RASPBERRY)
                self.create_oval(cx - rad, y - rad, cx + rad, y + rad, fill="white", outline="")
                self.create_text(cx, y, text=num, fill=RASPBERRY, font=font(10, "bold"))
                self.create_text(S(60), y, text=name, anchor="w", fill="white", font=font(11, "bold"))
            elif state == "done":
                self.create_oval(cx - rad, y - rad, cx + rad, y + rad, fill=GREEN, outline="")
                self.create_text(cx, y, text="✓", fill="white", font=font(10, "bold"))
                self.create_text(S(60), y, text=name, anchor="w", fill=INK, font=font(11))
            else:
                self.create_oval(cx - rad, y - rad, cx + rad, y + rad, fill=FIELD, outline=LINE)
                self.create_text(cx, y, text=num, fill=SUBTLE, font=font(10))
                self.create_text(S(60), y, text=name, anchor="w", fill=SUBTLE, font=font(11))
            y += step

        self.create_text(S(22), h - S(24), text="Raspberry Pi 5  ·  Hailo-8  ·  DeepX",
                        anchor="w", fill="#BDBDBD", font=font(8))


class ConfigRow(tk.Frame):
    """One Imager-style list row: bold title + grey description + a control."""

    def __init__(self, parent, title, desc, values, default, command=None):
        super().__init__(parent, bg=WHITE)
        self.columnconfigure(0, weight=1)
        left = tk.Frame(self, bg=WHITE)
        left.grid(row=0, column=0, sticky="w")
        self._title_lbl = tk.Label(left, text=title, bg=WHITE, fg=INK,
                                   font=font(11, "bold"))
        self._title_lbl.pack(anchor="w")
        tk.Label(left, text=desc, bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")

        self.var = tk.StringVar(value=str(default))
        self.combo = ttk.Combobox(self, textvariable=self.var,
                                  values=[str(v) for v in values],
                                  state="readonly", width=14, font=font(10),
                                  style="Rpi.TCombobox")
        self.combo.grid(row=0, column=1, sticky="e", padx=(S(8), 0))
        if command is not None:
            self.combo.bind("<<ComboboxSelected>>", lambda _e: command())
        tk.Frame(self, bg=LINE, height=1).grid(row=1, column=0, columnspan=2,
                                               sticky="ew", pady=(S(12), 0))

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(str(value))

    def set_enabled(self, on: bool):
        try:
            self.combo.config(state="readonly" if on else "disabled")
            self._title_lbl.config(fg=INK if on else "#B6B6B6")
        except tk.TclError:
            pass


class SensorSelector(tk.Frame):
    """Primary input: a row of clickable sensor cards, each with a transparent
    product picture so the operator can see which module they're choosing.

    Exposes the same ``.get()`` / ``.set()`` surface as ``ConfigRow`` so the rest
    of the form treats it as ``self.rows['sensor']``.
    """

    THUMB = 116  # logical px; scaled with S()

    def __init__(self, parent, cards, default, command=None):
        super().__init__(parent, bg=WHITE)
        self.cards = cards
        self.var = tk.StringVar(value=default)
        self.command = command
        self._imgs = {}      # keep PhotoImage refs alive
        self._frames = {}    # key -> card frame (for highlight)
        self._inner = {}     # key -> dict of child widgets to re-tint

        grid = tk.Frame(self, bg=WHITE)
        grid.pack(fill="x")
        for i in range(len(cards)):
            grid.columnconfigure(i, weight=1, uniform="sensor")

        for i, c in enumerate(cards):
            self._build_card(grid, c, i)
        self._refresh()

    def _load_thumb(self, path):
        if not (ImageTk and Path(path).exists()):
            return None
        try:
            im = Image.open(path).convert("RGBA")
            box = S(self.THUMB)
            im.thumbnail((box, box), Image.LANCZOS)
            return ImageTk.PhotoImage(im)
        except Exception:
            return None

    def _build_card(self, parent, c, col):
        key = c["key"]
        card = tk.Frame(parent, bg=WHITE, highlightthickness=2,
                        highlightbackground=LINE, highlightcolor=LINE,
                        cursor="hand2")
        card.grid(row=0, column=col, sticky="nsew", padx=S(5))
        pad = tk.Frame(card, bg=WHITE)
        pad.pack(fill="both", expand=True, padx=S(10), pady=S(10))

        img = self._load_thumb(c["image"])
        self._imgs[key] = img
        thumb_bg = tk.Frame(pad, bg=FIELD, height=S(self.THUMB + 14))
        thumb_bg.pack(fill="x")
        thumb_bg.pack_propagate(False)
        if img is not None:
            lbl = tk.Label(thumb_bg, image=img, bg=FIELD)
        else:
            lbl = tk.Label(thumb_bg, text="sensor", bg=FIELD, fg=SUBTLE,
                           font=font(9))
        lbl.pack(expand=True)

        head = tk.Frame(pad, bg=WHITE); head.pack(fill="x", pady=(S(8), 0))
        name = tk.Label(head, text=c["name"], bg=WHITE, fg=INK,
                        font=font(13, "bold"))
        name.pack(side="left")
        tag = tk.Label(head, text=f" {c['tag']} ", bg=FIELD, fg=SUBTLE,
                       font=font(7, "bold"))
        tag.pack(side="right")
        fam = tk.Label(pad, text=c["family"], bg=WHITE, fg=RASPBERRY,
                       font=font(9, "bold"), wraplength=S(150), justify="left",
                       anchor="w")
        fam.pack(anchor="w", fill="x")
        specs = tk.Label(pad, text=c["specs"], bg=WHITE, fg=INK, font=font(8),
                         wraplength=S(150), justify="left", anchor="w")
        specs.pack(anchor="w", fill="x", pady=(S(2), 0))
        blurb = tk.Label(pad, text=c["blurb"], bg=WHITE, fg=SUBTLE,
                         font=font(8), wraplength=S(150), justify="left",
                         anchor="w")
        blurb.pack(anchor="w", fill="x", pady=(S(4), 0))

        self._frames[key] = card
        self._inner[key] = {"pad": pad, "thumb_bg": thumb_bg, "img": lbl,
                            "head": head, "name": name, "fam": fam,
                            "specs": specs, "blurb": blurb, "tag": tag}

        # Keep wrap width in sync with the actual card width so long spec strings
        # always wrap instead of being clipped (which made text look blank/white
        # on narrow cards at large UI scales).
        def _wrap(_e, k=key):
            w = max(self._frames[k].winfo_width() - S(34), S(80))
            for nm in ("fam", "specs", "blurb"):
                self._inner[k][nm].configure(wraplength=w)
        card.bind("<Configure>", _wrap)

        for w in (card, pad, thumb_bg, lbl, head, name, fam, specs, blurb):
            w.bind("<Button-1>", lambda _e, k=key: self._select(k))

    def _select(self, key):
        if self.var.get() != key:
            self.var.set(key)
            self._refresh()
            if self.command:
                self.command()

    def _refresh(self):
        sel = self.var.get()
        for key, card in self._frames.items():
            on = key == sel
            border = RASPBERRY if on else LINE
            bg = "#FCEEF2" if on else WHITE
            card.configure(highlightbackground=border, highlightcolor=border,
                           bg=bg)
            parts = self._inner[key]
            for name in ("pad", "head", "name", "fam", "specs", "blurb"):
                try:
                    parts[name].configure(bg=bg)
                except Exception:
                    pass

    def get(self):
        return self.var.get()

    def set(self, value):
        if value in self._frames:
            self.var.set(value)
            self._refresh()


class LiveView(tk.Toplevel):
    """In-app live camera testing, styled to match the rest of the UI.

    Runs the compiled denoiser on a real camera frame-by-frame and shows the raw
    sensor feed beside the cleaned output, with live FPS / latency / noise-drop.
    Capture + inference run on a worker thread; the Tk thread only paints the
    latest frames, so the window stays responsive and looks like the rest of the
    app (white surface, raspberry accents, official logo, clean type).
    """

    PANEL_W = 372  # logical px per video panel (scaled by S)

    def __init__(self, master, source="auto", camera_index=0):
        super().__init__(master, bg=WHITE)
        self.title("NAS  ·  Live Testing")
        self.configure(bg=WHITE)
        # On Windows/macOS skip picamera2 and go straight to OpenCV webcams.
        if source == "auto" and not sys.platform.startswith("linux"):
            source = "opencv"
        self.source = source
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._cam_index_var = tk.StringVar(value=str(camera_index))
        self._lock = threading.Lock()
        self._latest = None            # (raw_rgb, out_rgb, stats)
        self._status = "Loading compiled model…"
        self._model_name = "MODEL"
        self._imgs = {}                # keep PhotoImage refs alive
        self._logo_img = None
        self.panels = {}
        self.stat_vals = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())

        self.update_idletasks()
        w, h = S(840), S(700)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(w, sw - 40), min(h, sh - 80)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(S(620), S(540))
        try:
            self.transient(master)
        except Exception:  # noqa: BLE001
            pass

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self.after(80, self._paint)
        self.after(200, self._probe_cameras_bg)

    # -- layout --------------------------------------------------------------
    def _build_ui(self):
        pad = S(24)
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(18), S(2)))
        if ImageTk and LOGO_PATH.exists():
            try:
                im = Image.open(LOGO_PATH).convert("RGBA").resize(
                    (S(40), S(40)), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(im)
                tk.Label(header, image=self._logo_img, bg=WHITE).pack(
                    side="left", padx=(0, S(12)))
            except Exception:  # noqa: BLE001
                pass
        htext = tk.Frame(header, bg=WHITE)
        htext.pack(side="left")
        tk.Label(htext, text="Live testing", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        self.src_lbl = tk.Label(htext, text=self._status, bg=WHITE, fg=SUBTLE,
                                font=font(10))
        self.src_lbl.pack(anchor="w", pady=(S(1), 0))
        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        # -- Webcam picker (USB / built-in — no picamera2/apt needed) --------
        camrow = tk.Frame(self, bg=WHITE)
        camrow.pack(fill="x", padx=pad, pady=(S(8), 0))
        tk.Label(camrow, text="Webcam index", bg=WHITE, fg=INK,
                 font=font(10, "bold")).pack(side="left")
        self._cam_combo = ttk.Combobox(
            camrow, textvariable=self._cam_index_var,
            values=[str(i) for i in range(10)], state="readonly",
            width=4, style="Rpi.TCombobox")
        self._cam_combo.pack(side="left", padx=(S(8), 0))
        RoundButton(camrow, "CONNECT", self._reconnect_camera, kind="primary",
                    width=130, height=34).pack(side="left", padx=(S(10), 0))
        self._cam_hint = tk.Label(
            camrow,
            text=("Probing for cameras…  On Pi: run python pi_camera_check.py "
                  "if CSI camera is not detected (usually no sudo needed)"),
            bg=WHITE, fg=SUBTLE, font=font(8), wraplength=S(420), justify="left")
        self._cam_hint.pack(side="left", padx=(S(10), 0))

        # -- Footer (pinned) -------------------------------------------------
        footer = tk.Frame(self, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(14))
        tk.Frame(self, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        RoundButton(footer, "CLOSE", self._on_close, kind="secondary",
                    width=120, height=42).pack(side="left")
        RoundButton(footer, "SAVE SNAPSHOT", self._save_snapshot, kind="primary",
                    width=180, height=42).pack(side="right")

        # -- Stat chips (above footer) --------------------------------------
        stats = tk.Frame(self, bg=WHITE)
        stats.pack(side="bottom", fill="x", padx=pad, pady=(0, S(2)))
        self._chip(stats, "MODEL", "model", "—")
        self._chip(stats, "LATENCY", "ms", "—")
        self._chip(stats, "THROUGHPUT", "fps", "—")
        self._chip(stats, "NOISE vs RAW", "drop", "—", accent=GREEN)

        # -- Video panels ----------------------------------------------------
        body = tk.Frame(self, bg=WHITE)
        body.pack(fill="both", expand=True, padx=pad, pady=(S(14), S(6)))
        body.columnconfigure(0, weight=1, uniform="vid")
        body.columnconfigure(1, weight=1, uniform="vid")
        self._panel(body, 0, "raw", "RAW SENSOR", "noisy input", SUBTLE)
        self._panel(body, 1, "out", "NAS DENOISED", "optimised output", GREEN)

    def _panel(self, parent, col, key, title, sub, accent):
        card = tk.Frame(parent, bg=WHITE, highlightthickness=1,
                        highlightbackground=LINE, highlightcolor=LINE)
        card.grid(row=0, column=col, sticky="nsew", padx=S(6))
        head = tk.Frame(card, bg=WHITE)
        head.pack(fill="x", padx=S(12), pady=(S(10), S(6)))
        tk.Label(head, text=title, bg=WHITE, fg=accent,
                 font=font(12, "bold")).pack(side="left")
        tk.Label(head, text=sub, bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(side="right")
        stage = tk.Frame(card, bg="#0F0F0F")
        stage.pack(fill="both", expand=True, padx=S(12), pady=(0, S(12)))
        lbl = tk.Label(stage, bg="#0F0F0F", fg="#BDBDBD",
                       text="Connecting to camera…", font=font(10))
        lbl.pack(fill="both", expand=True)
        self.panels[key] = lbl

    def _chip(self, parent, label, key, value, accent=INK):
        chip = tk.Frame(parent, bg=FIELD)
        chip.pack(side="left", padx=(0, S(8)))
        inner = tk.Frame(chip, bg=FIELD)
        inner.pack(padx=S(12), pady=S(7))
        tk.Label(inner, text=label, bg=FIELD, fg=SUBTLE,
                 font=font(8, "bold")).pack(anchor="w")
        val = tk.Label(inner, text=value, bg=FIELD, fg=accent,
                       font=font(13, "bold"))
        val.pack(anchor="w")
        self.stat_vals[key] = val

    # -- worker (capture + inference) ---------------------------------------
    def _set_status(self, text):
        with self._lock:
            self._status = text

    def _probe_cameras_bg(self):
        def work():
            try:
                import live as _live
                from nsa.pi_camera import diagnose, on_raspberry_pi
                found = _live.probe_cameras()
                pi = diagnose() if on_raspberry_pi() else None
                self.after(0, lambda: self._on_probe_done(found, pi))
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=work, daemon=True).start()

    def _on_probe_done(self, found: list[int], pi_diag=None):
        if not self.winfo_exists():
            return
        if found:
            vals = [str(i) for i in found]
            self._cam_combo.config(values=vals)
            if self._cam_index_var.get() not in vals:
                self._cam_index_var.set(vals[0])
            self._cam_hint.config(
                text=f"Found webcam(s) at index: {', '.join(vals)} — pick one, CONNECT",
                fg=GREEN)
            # Auto-reconnect to the first detected camera if we're on sim.
            with self._lock:
                on_sim = "simulated" in self._status.lower()
            if on_sim:
                self._reconnect_camera()
            return
        if pi_diag:
            if pi_diag.get("picamera2_importable"):
                self._cam_hint.config(
                    text="CSI camera via picamera2 is ready — reconnecting…",
                    fg=GREEN)
                self._reconnect_camera()
                return
            if pi_diag.get("rpicam_vid"):
                self._cam_hint.config(
                    text=f"CSI via {Path(pi_diag['rpicam_vid']).name} "
                         "(no picamera2) — click CONNECT",
                    fg=GREEN)
                return
            rec = (pi_diag.get("recommendations") or [""])[0]
            short = rec.split("\n")[0] if rec else "Run: python pi_camera_check.py"
            self._cam_hint.config(text=short[:120], fg=AMBER)
            return
        self._cam_hint.config(
            text="No webcam yet — try index 0–9, close other camera apps, "
                 "check Windows Privacy → Camera, then CONNECT",
            fg=AMBER)

    def _reconnect_camera(self):
        """Ask the worker to reopen the webcam at the selected index."""
        self._reconnect.set()
        try:
            idx = self._cam_index_var.get()
            self._set_status(f"Reconnecting to webcam index {idx}…")
        except Exception:  # noqa: BLE001
            pass

    def _run(self):
        try:
            import live as _live
            import cv2
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Live module unavailable: {exc}")
            return
        # Make checkpoint/output paths absolute so it works regardless of cwd.
        _live.OUT = ROOT / "outputs"
        _live.CKPT = _live.OUT / "model.pt"

        try:
            self._set_status("Loading compiled model…")
            model, ck = _live.load_model(_live.make_args(source=self.source))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Could not load model: {exc}")
            return
        model_name = str(ck.get("model", {}).get("family", "model")).upper()
        self._model_name = model_name
        sensor_key = ck.get("sensor", "imx662")
        gain = int(ck.get("gain", 512))

        import time as _t
        fps = 0.0

        while not self._stop.is_set():
            try:
                idx = int(self._cam_index_var.get())
            except ValueError:
                idx = 0
            args = _live.make_args(source=self.source, camera_index=idx)
            self._set_status(f"Connecting to webcam index {idx}…")
            try:
                cam = _live.open_camera(args, sensor_key, gain)
            except SystemExit as exc:
                self._set_status(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Camera error: {exc}")
                return

            is_sim = cam.__class__.__name__ == "SimCam"
            if is_sim:
                self._set_status(
                    f"No camera at index {idx} — simulated stream. "
                    "Try another index + CONNECT, or close apps using the camera.")
            else:
                self._set_status(f"Live: {cam.name}")

            self._reconnect.clear()
            t_prev = _t.perf_counter()
            try:
                while not self._stop.is_set() and not self._reconnect.is_set():
                    raw = cam.read()
                    if raw is None:
                        self._set_status("Camera returned no frame — try CONNECT again.")
                        break
                    if raw.ndim == 2:
                        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    elif raw.shape[2] == 4:
                        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    h, w = raw.shape[:2]
                    tw = 432
                    if w > tw:
                        raw = cv2.resize(raw, (tw, int(round(h * tw / w))))

                    out, dt_ms = _live.denoise_bgr(model, raw)
                    n_in, n_out = _live.noise_level(raw), _live.noise_level(out)

                    now = _t.perf_counter()
                    inst = 1.0 / max(now - t_prev, 1e-6)
                    fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                    t_prev = now
                    drop = max(0.0, (1.0 - n_out / n_in) * 100.0) if n_in > 1e-6 else 0.0

                    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._latest = (raw_rgb, out_rgb,
                                        {"fps": fps, "ms": dt_ms, "drop": drop,
                                         "model": model_name})
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Live testing error: {exc}")
            finally:
                try:
                    cam.close()
                except Exception:  # noqa: BLE001
                    pass

            if self._stop.is_set():
                break
            # _reconnect set — loop opens the camera again at the new index.

    # -- painting (Tk thread) -----------------------------------------------
    def _paint(self):
        if self._stop.is_set() or not self.winfo_exists():
            return
        with self._lock:
            latest = self._latest
            status = self._status
        try:
            self.src_lbl.config(text=status)
        except Exception:  # noqa: BLE001
            return
        if latest is not None and Image is not None:
            raw_rgb, out_rgb, st = latest
            self._set_panel("raw", raw_rgb)
            self._set_panel("out", out_rgb)
            self.stat_vals["model"].config(text=st["model"])
            self.stat_vals["ms"].config(text=f"{st['ms']:.0f} ms")
            self.stat_vals["fps"].config(text=f"{st['fps']:.1f} FPS")
            self.stat_vals["drop"].config(text=f"-{st['drop']:.0f}%")
        self.after(45, self._paint)

    def _set_panel(self, key, rgb):
        try:
            im = Image.fromarray(rgb)
            pw = S(self.PANEL_W)
            ph = max(1, int(round(pw * im.height / im.width)))
            im = im.resize((pw, ph), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            self._imgs[key] = photo
            self.panels[key].config(image=photo, text="")
        except Exception:  # noqa: BLE001
            pass

    def _save_snapshot(self):
        with self._lock:
            latest = self._latest
        if latest is None or Image is None:
            messagebox.showinfo("Snapshot", "No frame yet — wait for the stream to "
                                "start, then try again.")
            return
        raw_rgb, out_rgb, _ = latest
        try:
            a, b = Image.fromarray(raw_rgb), Image.fromarray(out_rgb)
            gap = 6
            combo = Image.new("RGB", (a.width + gap + b.width, a.height), RASPBERRY)
            combo.paste(a, (0, 0))
            combo.paste(b, (a.width + gap, 0))
            out_dir = ROOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "live_preview.png"
            combo.save(path)
            messagebox.showinfo("Snapshot saved",
                                f"Saved the raw-vs-denoised frame to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Snapshot", str(exc))

    def _on_close(self):
        self._stop.set()
        try:
            if getattr(self, "_worker", None):
                self._worker.join(timeout=1.5)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        _resolve_font_family()
        self.title("NAS — Neural Architecture Search")
        self.configure(bg=WHITE)
        self._apply_geometry()
        # On Windows we already scale fonts via FT(); applying Tk's own scaling on
        # top double-counts on Linux, so only enable it on the Windows DPI path.
        try:
            self.tk.call("tk", "scaling", SCALE if USE_TK_SCALING else 1.0)
        except Exception:
            pass
        try:
            if LOGO_PATH.exists() and ImageTk:
                self.iconphoto(True, ImageTk.PhotoImage(Image.open(LOGO_PATH)))
        except Exception:
            pass

        self._init_style()
        self.proc = None
        self.q: queue.Queue[str] = queue.Queue()
        self.input_raw = None
        self.dataset_path = None
        self.upload_files = []
        self.hf_model_id = None
        self.hf_weight = None

        self._build_chrome()

    def _build_chrome(self):
        """(Re)build the sidebar + main area — used on launch and on rescale."""
        for w in self.winfo_children():
            w.destroy()
        self.sidebar = Sidebar(self)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.reset()

        self.main = tk.Frame(self, bg=WHITE)
        self.main.pack(side="left", fill="both", expand=True)
        self._build_form()

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Rpi.TCombobox", fieldbackground=FIELD, background=FIELD,
                        foreground=INK, arrowcolor=RASPBERRY, bordercolor=LINE,
                        lightcolor=LINE, darkcolor=LINE, relief="flat", padding=S(6))
        style.map("Rpi.TCombobox",
                  fieldbackground=[("readonly", FIELD)],
                  selectbackground=[("readonly", FIELD)],
                  selectforeground=[("readonly", INK)])
        style.configure("Rpi.Horizontal.TProgressbar", troughcolor=FIELD,
                        background=RASPBERRY, bordercolor=FIELD, thickness=S(6))
        self.option_add("*TCombobox*Listbox.font", font(10))

    # -- Form widgets ---------------------------------------------------------
    def _make_scrollable(self, parent):
        """Return an inner frame inside a vertically scrollable canvas."""
        canvas = tk.Canvas(parent, bg=WHITE, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=WHITE)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        def _wheel(e):
            step = -1 if (getattr(e, "num", 0) == 5 or getattr(e, "delta", 0) < 0) else 1
            canvas.yview_scroll(-step, "units")

        def _bind(_e):
            canvas.bind_all("<MouseWheel>", _wheel)
            canvas.bind_all("<Button-4>", _wheel)
            canvas.bind_all("<Button-5>", _wheel)

        def _unbind(_e):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(seq)

        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)
        return inner

    def _section(self, parent, text):
        wrap = tk.Frame(parent, bg=WHITE)
        wrap.pack(fill="x", pady=(S(14), S(2)))
        tk.Label(wrap, text=text, bg=WHITE, fg=RASPBERRY,
                 font=font(9, "bold")).pack(anchor="w")
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(S(2), S(2)))

    def _badge(self, parent, text, bg=FIELD, fg=SUBTLE):
        tk.Label(parent, text=f" {text} ", bg=bg, fg=fg,
                 font=font(8, "bold")).pack(side="left", padx=(S(10), 0))

    def _radio(self, parent, text, value, enabled, badge=None,
               variable=None, desc=None, command=None):
        variable = variable if variable is not None else self.mode_var
        fr = tk.Frame(parent, bg=WHITE)
        fr.pack(fill="x", pady=S(4))
        top = tk.Frame(fr, bg=WHITE); top.pack(fill="x")
        rb = tk.Radiobutton(
            top, text="  " + text, variable=variable, value=value,
            bg=WHITE, fg=(INK if enabled else "#B6B6B6"), selectcolor=WHITE,
            activebackground=WHITE, activeforeground=INK,
            font=font(11, "bold" if enabled else "normal"),
            state=("normal" if enabled else "disabled"),
            anchor="w", highlightthickness=0, bd=0, takefocus=enabled,
            command=command)
        rb.pack(side="left")
        if badge:
            self._badge(top, badge)
        if desc:
            tk.Label(fr, text="     " + desc, bg=WHITE,
                     fg=(SUBTLE if enabled else "#C4C4C4"),
                     font=font(9), wraplength=S(560),
                     justify="left").pack(anchor="w")

    def _coming_soon(self, parent, text, desc, badge="COMING SOON"):
        fr = tk.Frame(parent, bg=WHITE)
        fr.pack(fill="x", pady=S(6))
        var = tk.BooleanVar(value=False)
        top = tk.Frame(fr, bg=WHITE); top.pack(fill="x")
        cb = tk.Checkbutton(
            top, text="  " + text, variable=var, state="disabled",
            bg=WHITE, fg="#B6B6B6", selectcolor=WHITE, activebackground=WHITE,
            font=font(11), anchor="w", highlightthickness=0, bd=0)
        cb.pack(side="left")
        self._badge(top, badge)
        tk.Label(fr, text="     " + desc, bg=WHITE, fg="#C4C4C4",
                 font=font(9)).pack(anchor="w")

    def _check(self, parent, text, desc, variable, command=None):
        fr = tk.Frame(parent, bg=WHITE)
        fr.pack(fill="x", pady=S(6))
        cb = tk.Checkbutton(
            fr, text="  " + text, variable=variable, onvalue=True, offvalue=False,
            bg=WHITE, fg=INK, selectcolor=WHITE, activebackground=WHITE,
            activeforeground=INK, font=font(11, "bold"), anchor="w",
            highlightthickness=0, bd=0, command=command)
        cb.pack(anchor="w")
        if desc:
            tk.Label(fr, text="     " + desc, bg=WHITE, fg=SUBTLE,
                     font=font(9), wraplength=S(560),
                     justify="left").pack(anchor="w")
        return cb

    def _entry_row(self, parent, key, title, desc, default=""):
        row = tk.Frame(parent, bg=WHITE)
        row.pack(fill="x", pady=S(6))
        row.columnconfigure(0, weight=1)
        left = tk.Frame(row, bg=WHITE); left.grid(row=0, column=0, sticky="w")
        title_lbl = tk.Label(left, text=title, bg=WHITE, fg=INK, font=font(11, "bold"))
        title_lbl.pack(anchor="w")
        tk.Label(left, text=desc, bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(row, textvariable=var, width=18, font=font(10))
        ent.grid(row=0, column=1, sticky="e", padx=(S(8), 0))
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))
        self.entries[key] = var
        self.entry_widgets[key] = (ent, title_lbl)
        return var

    def _set_entry_enabled(self, key: str, on: bool):
        pair = getattr(self, "entry_widgets", {}).get(key)
        if not pair:
            return
        ent, lbl = pair
        try:
            ent.config(state="normal" if on else "disabled")
            lbl.config(fg=INK if on else "#B6B6B6")
        except tk.TclError:
            pass

    # -- Form view (step-by-step wizard) -------------------------------------
    def _add_rows(self, parent, specs):
        for key, title, desc, values, default in specs:
            row = ConfigRow(parent, title, desc, values, default)
            row.pack(fill="x", pady=S(6))
            self.rows[key] = row

    def _build_form(self):
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()

        pad = S(34)
        # Persistent state (kept alive across steps; steps only show/hide).
        self.rows = {}
        self.entries = {}
        self.entry_widgets = {}
        self.mode_var = tk.StringVar(value="single")
        self.source_var = tk.StringVar(value="real")
        self.sim_noise_var = tk.BooleanVar(value=False)
        self.quantize_var = tk.BooleanVar(value=True)
        self.qat_var = tk.BooleanVar(value=False)
        self.eval_var = tk.StringVar(value="single")     # single | sweep
        self.all_sensors_var = tk.BooleanVar(value=False)
        self._nafnet_topo = {"enc": "", "mid": "", "dec": ""}

        # Wizard chrome: a header (step indicator) + a content area + nav footer.
        self._wiz_header = tk.Frame(self.main, bg=WHITE)
        self._wiz_header.pack(fill="x", padx=pad, pady=(S(20), S(4)))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(8), 0))

        self._build_footer()                              # pinned nav at the bottom

        self.content = tk.Frame(self.main, bg=WHITE)
        self.content.pack(fill="both", expand=True, padx=pad, pady=(S(4), 0))

        # Step registry: (key, title, subtitle, builder).
        # The run type (single vs sweep) is chosen on the home screen, so the
        # wizard does not repeat it and starts straight at the sensor step.
        self._step_defs = [
            ("sensor", "Image sensor",
             "Pick the camera module you're optimising for — everything adapts to it.",
             self._step_sensor),
            ("data", "Capture source & data",
             "Where the frames come from, the ground truth, and the run mode.",
             self._step_data),
            ("model", "Model architecture",
             "The network family and its parameters (Level 3).",
             self._step_model),
            ("hw", "Hardware & calibration",
             "Target chip (Levels 4 & 6) and the calibration / quantization (Level 5).",
             self._step_hw),
            ("review", "Review & run",
             "Check your choices, then launch.",
             self._step_review),
        ]

        self._steps = []
        for key, title, subtitle, builder in self._step_defs:
            holder = tk.Frame(self.content, bg=WHITE)
            body = self._make_scrollable(holder)
            builder(body)
            self._steps.append({"key": key, "title": title,
                                "subtitle": subtitle, "holder": holder})

        self._home = tk.Frame(self.content, bg=WHITE)
        home_body = self._make_scrollable(self._home)
        tk.Label(home_body, text="Compile with your current settings",
                 bg=WHITE, fg=INK, font=font(15, "bold")).pack(anchor="w",
                                                                pady=(S(4), S(2)))
        tk.Label(home_body,
                 text="     Uses config.yaml plus the defaults below. "
                      "Press Edit Config to walk through every option.",
                 bg=WHITE, fg=SUBTLE, font=font(10), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(0, S(10)))
        eval_box = tk.Frame(home_body, bg=WHITE)
        eval_box.pack(fill="x", pady=(0, S(10)))
        self._section(eval_box, "RUN TYPE")
        self._radio(eval_box, "Single model compile", "single", enabled=True,
                    variable=self.eval_var, badge="COMPILE",
                    desc="Train and export one architecture with a full report.",
                    command=self._on_eval_change)
        self._radio(eval_box, "Architecture sweep (rank all families)", "sweep",
                    enabled=True, variable=self.eval_var, badge="SWEEP",
                    desc="Train every family, rank by Pareto fitness, pick a winner.",
                    command=self._on_eval_change)
        self._home_summary = tk.Frame(home_body, bg=WHITE)
        self._home_summary.pack(fill="x")

        # Initialise dependent UI state, then show the quick-run home screen.
        self._wizard_mode = "home"
        self._apply_denoise_hw_defaults()
        self._step = 0
        self._on_sensor_change()
        self._on_mode_change()
        self._on_source_change()
        self._on_eval_change()
        self._show_home()

    # -- Individual wizard steps ---------------------------------------------
    def _step_sensor(self, body):
        self._section(body, "PRIMARY INPUT · IMAGE SENSOR")
        tk.Label(body, text="     Pick the camera module you're optimising for. "
                 "Everything after this (noise model, data, network) adapts to it.",
                 bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(0, S(8)))
        sensor_sel = SensorSelector(body, SENSOR_CARDS, "imx219",
                                    command=self._on_sensor_change)
        sensor_sel.pack(fill="x", pady=(0, S(4)))
        self.rows["sensor"] = sensor_sel

        self.all_sensors_cb = self._check(
            body, "Test ALL sensor profiles (sweep only)",
            "Sweep mode only: also vary the sensor (IMX219 · IMX662 · IMX-NG) so the "
            "leaderboard shows which model suits which camera. Slower (3× the runs).",
            self.all_sensors_var, command=self._on_eval_change)

        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(10), 0))
        self._section(body, "LEVEL 1 · SENSOR GAIN")
        self.sensor_echo = tk.Label(
            body, text="", bg=WHITE, fg=RASPBERRY, font=font(9, "bold"),
            wraplength=S(560), justify="left")
        self.sensor_echo.pack(anchor="w", pady=(0, S(4)))
        self._add_rows(body, [
            ("gain", "Sensor Gain", "Challenge-frame analog gain", [256, 512], 256),
        ])

    def _step_data(self, body):
        self._section(body, "LEVEL 1 · CAPTURE SOURCE")
        self._radio(body, "Simulated capture", "sim", enabled=True,
                    variable=self.source_var,
                    desc="Synthesise a noisy frame from the sensor's noise physics.",
                    command=self._on_source_change)
        self._radio(body, "Real captures", "real", enabled=True,
                    variable=self.source_var, badge="REAL DATA",
                    desc="Load real frames; paired noisy/gt folders are auto-detected.",
                    command=self._on_source_change)

        ds_row = tk.Frame(body, bg=WHITE)
        ds_row.pack(fill="x", pady=(S(8), S(2)))
        ds_row.columnconfigure(0, weight=1)
        ds_left = tk.Frame(ds_row, bg=WHITE); ds_left.grid(row=0, column=0, sticky="w")
        tk.Label(ds_left, text="Image Source", bg=WHITE, fg=INK,
                 font=font(11, "bold")).pack(anchor="w")
        self.dataset_label = tk.Label(
            ds_left, text="using config.yaml dataset_path", bg=WHITE,
            fg=SUBTLE, font=font(9))
        self.dataset_label.pack(anchor="w")
        self.dataset_hint = tk.Label(
            ds_left, text="", bg=WHITE, fg=AMBER, font=font(9),
            wraplength=S(520), justify="left")
        self.dataset_hint.pack(anchor="w")
        btns = tk.Frame(ds_row, bg=WHITE); btns.grid(row=0, column=1, sticky="e")
        self.dataset_btn = RoundButton(btns, "CHOOSE FOLDER", self._choose_dataset,
                                       kind="secondary", width=160, height=36)
        self.dataset_btn.pack(side="left", padx=(0, S(8)))
        self.upload_btn = RoundButton(btns, "UPLOAD IMAGES", self._upload_images,
                                      kind="secondary", width=160, height=36)
        self.upload_btn.pack(side="left")
        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))

        self.filter_var = self._entry_row(
            body, "filter", "Dataset Filter",
            "Keyword filter for folders (e.g. imx219 ag12)", "imx219 ag12")
        self.sim_noise_cb = self._check(
            body, "Simulate sensor noise on loaded frames",
            "Inject the selected sensor's physics on top of the real frames.",
            self.sim_noise_var, command=self._on_source_change)
        self.noise_std_var = self._entry_row(
            body, "noise_std", "Noise Std (read-noise e-)",
            "Injected Gaussian noise std, denoise-hw style. Blank = sensor default; "
            "higher = noisier (e.g. 8 to stress-test)", "")

        self._section(body, "LEVEL 2 · GROUND TRUTH / DATA")
        self._add_rows(body, [
            ("frames", "Temporal Frames",
             "Synthetic GT only — averaged reads for simulated capture", [64, 128, 256], 256),
        ])
        tk.Label(body, text="     With real paired noisy/gt folders, ground truth comes "
                 "from disk — temporal frames and gain apply only to synthetic capture "
                 "(or when “Simulate sensor noise” is on).",
                 bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(0, S(4)))
        tk.Label(body, text="     Paired noisy/gt folders auto-detected · "
                 "detail-scored patch selection (denoise-hw logic).",
                 bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w", pady=(0, S(4)))

        self._section(body, "RUN MODE")
        self._radio(body, "Single Frame Calibration", "single", enabled=True,
                    desc="Calibrate + evaluate on one frame.",
                    command=self._on_mode_change)
        self._radio(body, "Batch Folder Calibration", "batch", enabled=True,
                    badge="MULTI-IMAGE",
                    desc="Train across many frames in a folder; metrics are averaged.",
                    command=self._on_mode_change)
        self._radio(body, "Temporal Video Denoise", "temporal", enabled=True,
                    badge="VIDEO",
                    desc="Recursive burst denoising of a frame sequence (IIR blend).",
                    command=self._on_mode_change)
        self.batch_var = self._entry_row(
            body, "batch", "Batch Size", "Frames to load in batch mode", "6")
        self.burst_var = self._entry_row(
            body, "burst", "Temporal Burst", "Frames in a temporal-denoise burst", "8")

        cache_row = tk.Frame(body, bg=WHITE)
        cache_row.pack(fill="x", pady=S(6))
        cache_row.columnconfigure(0, weight=1)
        cache_left = tk.Frame(cache_row, bg=WHITE); cache_left.grid(row=0, column=0, sticky="w")
        tk.Label(cache_left, text="Patch-cache training set builder", bg=WHITE, fg=INK,
                 font=font(11, "bold")).pack(anchor="w")
        tk.Label(cache_left, text="Pre-scan the chosen dataset into detail-scored "
                 "patches (needs a real dataset/folder).", bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(anchor="w")
        RoundButton(cache_row, "BUILD CACHE", self._build_cache, kind="secondary",
                    width=160, height=36).grid(row=0, column=1, sticky="e")

    def _step_model(self, body):
        self._section(body, "LEVEL 3 · MODEL ARCHITECTURE")
        self.model_intro = tk.Label(
            body, text="", bg=WHITE, fg=RASPBERRY, font=font(9, "bold"),
            wraplength=S(560), justify="left")
        self.model_intro.pack(anchor="w", pady=(0, S(4)))
        fam_row = ConfigRow(body, "Model Family",
                            "CNN · DnCNN · U-Net · RED-Net · RIDNet · NAFNet · "
                            "FFDNet · DRUNet · Restormer",
                            ["cnn", "dncnn", "unet", "rednet", "ridnet", "nafnet",
                             "ffdnet", "drunet", "restormer"],
                            "nafnet", command=self._on_family_change)
        fam_row.pack(fill="x", pady=S(6))
        self.rows["model_family"] = fam_row
        self.model_box = tk.Frame(body, bg=WHITE)
        self.model_box.pack(fill="x")
        self._render_model_options()

        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(14), 0))
        self._section(body, "EXTERNAL MODELS · HUGGING FACE HUB")
        tk.Label(body, text="     Source pretrained denoisers from the Hub (Apache-2.0 / "
                 "MIT only). Search, freeze the commit SHA, then click "
                 "DOWNLOAD & USE — the next RUN COMPILE will load those weights "
                 "(ONNX or PyTorch) instead of training from scratch.",
                 bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(0, S(6)))
        hf_row = tk.Frame(body, bg=WHITE); hf_row.pack(fill="x", pady=S(4))
        RoundButton(hf_row, "BROWSE HUGGING FACE", self._hf_browser,
                    kind="secondary", width=220, height=38).pack(side="left")
        self.hf_active_lbl = tk.Label(hf_row, text="", bg=WHITE, fg=GREEN,
                                      font=font(9), wraplength=S(320), justify="left")
        self.hf_active_lbl.pack(side="left", padx=(S(12), 0))
        self._refresh_hf_active_label()

    def _step_hw(self, body):
        self._section(body, "LEVELS 4 & 6 · HARDWARE COMPILER TARGET")
        self._add_rows(body, [
            ("hardware", "Target Hardware", "Pi 5 CPU · Hailo-8 (est.) · DeepX (est.)",
             ["rpi5_cpu", "hailo8", "deepx"], "hailo8"),
        ])
        self.hw_note = tk.Label(body, text="", bg=WHITE, fg=SUBTLE, font=font(9),
                                wraplength=S(560), justify="left")
        self.hw_note.pack(anchor="w", pady=(0, S(2)))
        dep_row = tk.Frame(body, bg=WHITE)
        dep_row.pack(fill="x", pady=S(6))
        dep_row.columnconfigure(0, weight=1)
        dep_left = tk.Frame(dep_row, bg=WHITE); dep_left.grid(row=0, column=0, sticky="w")
        tk.Label(dep_left, text="Deployment Package", bg=WHITE, fg=INK,
                 font=font(11, "bold")).pack(anchor="w")
        tk.Label(dep_left, text="Bundle artifacts + flash instructions "
                 "(device SDK still needed to flash).", bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(anchor="w")
        RoundButton(dep_row, "BUILD PACKAGE", self._build_deploy, kind="secondary",
                    width=170, height=36).grid(row=0, column=1, sticky="e")
        tk.Label(body, text="     Optional — you can also export the transferable "
                 ".zip from the results screen after a compile.", bg=WHITE,
                 fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(S(4), 0))
        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))

        self._section(body, "LEVEL 5 · CALIBRATION & QUANTIZATION")
        self._add_rows(body, [
            ("steps", "Calibration", "Live fit iterations (lower = faster)",
             [150, 300, 500, 800], 300),
        ])

        loss_row = ConfigRow(
            body, "Loss Function",
            "Training objective — parameters below adapt to it",
            ["charbonnier", "l1", "l2", "huber", "ssim", "charbonnier_ssim"],
            "charbonnier", command=self._on_loss_change)
        loss_row.pack(fill="x", pady=S(6))
        self.rows["loss"] = loss_row
        self.loss_box = tk.Frame(body, bg=WHITE)
        self.loss_box.pack(fill="x")
        self._render_loss_options()

        self._check(body, "INT8 quantization (PTQ)",
                    "Quantize weights + activations for the accelerator target.",
                    self.quantize_var)
        raw_row = tk.Frame(body, bg=WHITE)
        raw_row.pack(fill="x", pady=S(6))
        raw_row.columnconfigure(0, weight=1)
        left = tk.Frame(raw_row, bg=WHITE); left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Single RAW (optional)", bg=WHITE, fg=INK,
                 font=font(11, "bold")).pack(anchor="w")
        self.raw_label = tk.Label(left, text="none — using source above", bg=WHITE,
                                  fg=SUBTLE, font=font(9))
        self.raw_label.pack(anchor="w")
        RoundButton(raw_row, "CHOOSE RAW", self._choose_raw, kind="secondary",
                    width=140, height=36).grid(row=0, column=1, sticky="e")
        self._check(body, "Quantization-Aware Training (QAT)",
                    "Train with INT8 fake-quant in the loop (STE) to recover "
                    "quantization loss. Auto-on for gelu→DeepX / non-native acts.",
                    self.qat_var)

    def _step_review(self, body):
        self._section(body, "REVIEW")
        self._review_box = tk.Frame(body, bg=WHITE)
        self._review_box.pack(fill="x", pady=(S(2), 0))
        tk.Label(body, text="     Use BACK to change anything, or press the button "
                 "below to launch. A sweep takes a few minutes; a single compile is "
                 "quicker.", bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(S(10), 0))

    # -- Wizard navigation ----------------------------------------------------
    def _show_home(self):
        """Quick-run landing: prominent Run, optional full config wizard."""
        self._wizard_mode = "home"
        for st in self._steps:
            st["holder"].pack_forget()
        self._home.pack(fill="both", expand=True)
        self._refresh_home_summary()
        self._render_wiz_header()
        self._render_nav()
        try:
            self.sidebar.reset()
        except Exception:
            pass

    def _enter_config_wizard(self):
        self._wizard_mode = "steps"
        self._home.pack_forget()
        self._goto_step(0)

    def _refresh_home_summary(self):
        if not hasattr(self, "_home_summary"):
            return
        for w in self._home_summary.winfo_children():
            w.destroy()
        for label, val, col in self._config_summary_rows():
            rr = tk.Frame(self._home_summary, bg=WHITE)
            rr.pack(fill="x", pady=S(4))
            tk.Label(rr, text=label, bg=WHITE, fg=SUBTLE, font=font(10),
                     width=16, anchor="w").pack(side="left")
            tk.Label(rr, text=val, bg=WHITE, fg=col, font=font(11, "bold"),
                     wraplength=S(420), justify="left").pack(side="left")

    def _config_summary_rows(self):
        """Key/value lines for the home screen and review step."""
        sensor_key = self._row_get("sensor", "imx219")
        sensor_card = next((c for c in SENSOR_CARDS if c["key"] == sensor_key), None)
        sensor_name = sensor_card["name"] if sensor_card else sensor_key
        hw_key = self._row_get("hardware", "hailo8")
        hw_name = {"rpi5_cpu": "Raspberry Pi 5 (CPU)", "hailo8": "Pi 5 + Hailo-8",
                   "deepx": "DeepX DX-M1"}.get(hw_key, hw_key)
        fam = self._row_get("model_family", "nafnet")
        model_txt = (f"{fam.upper()}  {self._row_get('base_channels','32')}ch × "
                     f"depth {self._row_get('block_depth','4')}")
        src = "Real captures" if self.source_var.get() == "real" else "Simulated physics"
        mode = {"single": "Single frame", "batch": "Batch folder",
                "temporal": "Temporal video"}.get(self.mode_var.get(), self.mode_var.get())
        q = ("INT8 PTQ" + (" + QAT" if self.qat_var.get() else "")
             if self.quantize_var.get() else "off")
        sweep = self.eval_var.get() == "sweep"
        run_type = ("Architecture sweep (all 9 families)"
                    if sweep else "Single model compile")
        rows = [
            ("Run type", run_type, AMBER if sweep else RASPBERRY),
            ("Image sensor", f"{sensor_name}  @{self._row_get('gain','256')}×", RASPBERRY),
            ("Capture source", src, INK),
            ("Run mode", mode, INK),
            ("Model", model_txt, INK),
            ("Target chip", hw_name, INK),
            ("Calibration", f"{self._row_get('steps','300')} steps · quant {q}", INK),
            ("Loss", self._row_get("loss", "charbonnier"), INK),
        ]
        hint = self._dataset_quality_hint()
        if hint:
            rows.append(("Dataset", hint[0], hint[1]))
        return rows

    def _goto_step(self, i):
        i = max(0, min(i, len(self._steps) - 1))
        self._wizard_mode = "steps"
        if hasattr(self, "_home"):
            self._home.pack_forget()
        self._step = i
        for j, st in enumerate(self._steps):
            if j == i:
                st["holder"].pack(fill="both", expand=True)
            else:
                st["holder"].pack_forget()
        if self._steps[i]["key"] == "review":
            self._refresh_review()
        self._render_wiz_header()
        self._render_nav()
        try:
            self.sidebar.set_active(min(i, len(LEVELS) - 1))
        except Exception:
            pass

    def _render_wiz_header(self):
        for w in self._wiz_header.winfo_children():
            w.destroy()
        if getattr(self, "_wizard_mode", "home") == "home":
            tk.Label(self._wiz_header, text="READY", bg=WHITE, fg=RASPBERRY,
                     font=font(9, "bold")).pack(anchor="w")
            tk.Label(self._wiz_header, text="Neural Architecture Search",
                     bg=WHITE, fg=INK, font=font(19, "bold")).pack(anchor="w")
            tk.Label(self._wiz_header,
                     text="One click to compile with config.yaml — or edit first.",
                     bg=WHITE, fg=SUBTLE, font=font(10)).pack(anchor="w", pady=(S(2), 0))
            return
        st = self._steps[self._step]
        tk.Label(self._wiz_header, text=f"STEP {self._step + 1} OF "
                 f"{len(self._steps)}", bg=WHITE, fg=RASPBERRY,
                 font=font(9, "bold")).pack(anchor="w")
        tk.Label(self._wiz_header, text=st["title"], bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(self._wiz_header, text=st["subtitle"], bg=WHITE, fg=SUBTLE,
                 font=font(10)).pack(anchor="w", pady=(S(2), 0))

    def _render_nav(self):
        for w in self._nav.winfo_children():
            w.destroy()
        if getattr(self, "_wizard_mode", "home") == "home":
            RoundButton(self._nav, "APP OPTIONS", self._app_options, kind="secondary",
                        width=120, height=40).pack(side="left")
            RoundButton(self._nav, "HISTORY", self._show_history, kind="secondary",
                        width=100, height=40).pack(side="left", padx=(S(6), 0))
            run_lbl = ("▶ SWEEP" if self.eval_var.get() == "sweep"
                       else "▶ COMPILE")
            self.run_btn = RoundButton(self._nav, run_lbl, self._run,
                                       kind="primary", width=168, height=44)
            self.run_btn.pack(side="right")
            RoundButton(self._nav, "EDIT CONFIG", self._enter_config_wizard,
                        kind="secondary", width=130, height=40).pack(side="right",
                                                                     padx=(0, S(8)))
            return
        RoundButton(self._nav, "APP OPTIONS", self._app_options, kind="secondary",
                    width=140, height=44).pack(side="left")
        if self._step == 0:
            RoundButton(self._nav, "◀ HOME", self._show_home, kind="secondary",
                        width=120, height=44).pack(side="left", padx=(S(8), 0))
        elif self._step > 0:
            RoundButton(self._nav, "◀ BACK", lambda: self._goto_step(self._step - 1),
                        kind="secondary", width=120,
                        height=44).pack(side="left", padx=(S(8), 0))
        RoundButton(self._nav, "HISTORY", self._show_history, kind="secondary",
                    width=120, height=44).pack(side="left", padx=(S(8), 0))
        last = self._step == len(self._steps) - 1
        if last:
            label = "RUN SWEEP" if self.eval_var.get() == "sweep" else "RUN COMPILE"
            self.run_btn = RoundButton(self._nav, label, self._run, kind="primary",
                                       width=180, height=44)
            self.run_btn.pack(side="right")
        else:
            RoundButton(self._nav, "NEXT ▶", lambda: self._goto_step(self._step + 1),
                        kind="primary", width=150, height=44).pack(side="right")

    def _refresh_review(self):
        if not hasattr(self, "_review_box"):
            return
        for w in self._review_box.winfo_children():
            w.destroy()
        sweep = self.eval_var.get() == "sweep"
        if sweep and self.all_sensors_var.get():
            sensor_name = "All profiles (IMX219 · IMX662 · IMX-NG)"
        else:
            sensor_key = self._row_get("sensor", "imx219")
            sensor_card = next((c for c in SENSOR_CARDS if c["key"] == sensor_key), None)
            sensor_name = sensor_card["name"] if sensor_card else sensor_key
        goal = ("Sweep & rank all 9 model families" if sweep
                else "Compile one specific model")
        fam = self._row_get("model_family", "nafnet")
        if sweep:
            model_txt = (f"swept · width {self._row_get('base_channels','32')}ch, "
                         f"depth varies")
        else:
            model_txt = (f"{fam.upper()}  {self._row_get('base_channels','32')}ch × "
                         f"depth {self._row_get('block_depth','4')}")
        rows = [("Goal", goal, RASPBERRY)]
        for row in self._config_summary_rows():
            if row[0] == "Model":
                rows.append(("Model", model_txt, INK))
            elif row[0] != "Image sensor" or not sweep:
                rows.append(row)
        if sweep and self.all_sensors_var.get():
            for i, row in enumerate(rows):
                if row[0] == "Image sensor":
                    rows[i] = ("Image sensor", sensor_name, RASPBERRY)
                    break
        for label, val, col in rows:
            rr = tk.Frame(self._review_box, bg=WHITE); rr.pack(fill="x", pady=S(3))
            tk.Label(rr, text=label, bg=WHITE, fg=SUBTLE, font=font(10),
                     width=16, anchor="w").pack(side="left")
            tk.Label(rr, text=val, bg=WHITE, fg=col, font=font(11, "bold"),
                     wraplength=S(420), justify="left").pack(side="left")

    def _build_footer(self):
        pad = S(34)
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        self._nav = tk.Frame(self.main, bg=WHITE)
        self._nav.pack(side="bottom", fill="x", padx=pad, pady=S(16))
        # Buttons are (re)created per step by _render_nav().

    def _apply_denoise_hw_defaults(self):
        """Load config.yaml into the form so quick-run uses the right defaults."""
        try:
            from nsa.config import finalize_dataset_config, load_config, project_root
            from nsa.denoise_hw_data import ensure_project_dataset
            ensure_project_dataset(ROOT)
            cfg = load_config(ROOT / "config.yaml")
            finalize_dataset_config(cfg, ROOT)

            if cfg.hardware and "hardware" in self.rows:
                self.rows["hardware"].set(cfg.hardware)
            if cfg.model.model_family and "model_family" in self.rows:
                self.rows["model_family"].set(cfg.model.model_family)
            enc_blocks = list(cfg.model.nafnet_enc_blocks or [])
            self._write_nafnet_topo(
                " ".join(str(x) for x in enc_blocks),
                str(cfg.model.nafnet_middle_blocks or "") if enc_blocks else "",
                " ".join(str(x) for x in (cfg.model.nafnet_dec_blocks or [])),
            )
            if "model_family" in self.rows:
                self._render_model_options()
            for key, val in (
                ("base_channels", cfg.model.base_channels),
                ("block_depth", cfg.model.block_depth),
                ("conv_type", cfg.model.conv_type),
                ("activation", cfg.model.activation),
                ("gain", cfg.sensor.gain),
                ("steps", cfg.optimization.calibration_steps),
                ("frames", cfg.data.temporal_frames),
            ):
                if key in self.rows and val is not None:
                    self.rows[key].set(val)
            self.quantize_var.set(cfg.optimization.quantize)
            self.qat_var.set(cfg.optimization.qat)
            self.mode_var.set(cfg.run.mode)
            if cfg.sensor.real_capture:
                self.source_var.set("real")
            else:
                self.source_var.set("sim")
            if cfg.sensor.dataset_path:
                self.dataset_path = cfg.sensor.dataset_path
            if cfg.sensor.filter and hasattr(self, "filter_var"):
                self.filter_var.set(" ".join(cfg.sensor.filter))
            if hasattr(self, "noise_std_var"):
                self.noise_std_var.set(
                    "" if cfg.sensor.noise_std is None else str(cfg.sensor.noise_std))
            lc = getattr(cfg.optimization, "loss", None)
            if lc is not None and "loss" in self.rows:
                self.rows["loss"].set(lc.name)
                self._render_loss_options()
                for key, val in (("charbonnier_eps", lc.charbonnier_eps),
                                 ("huber_delta", lc.huber_delta),
                                 ("ssim_window", lc.ssim_window),
                                 ("ssim_weight", lc.ssim_weight)):
                    var = self.entries.get(key)
                    if var is not None:
                        var.set(str(val))
            if cfg.sensor.sensor and "sensor" in self.rows:
                self.rows["sensor"].set(cfg.sensor.sensor)
            if hasattr(self, "dataset_label"):
                label = (self.dataset_path or cfg.sensor.dataset_path
                         or "datasets/PI_RAW (denoise-hw)")
                self.dataset_label.config(text=str(label), fg=SUBTLE)
            self._on_source_change()
            self._on_sensor_change()
            if hasattr(self, "_home_summary"):
                self._refresh_home_summary()
        except Exception:
            pass

    def _dataset_root_path(self) -> Path | None:
        path = self.dataset_path
        if path:
            return Path(path)
        default = ROOT / "datasets" / "PI_RAW"
        if default.exists():
            return default
        return None

    def _dataset_quality_hint(self) -> tuple[str, str] | None:
        """(label, color) for home/review when captures are not real DNG PI_RAW."""
        if self.source_var.get() != "real":
            return None
        try:
            from nsa.denoise_hw_data import dataset_summary, is_synthetic_sample_dataset
            root = self._dataset_root_path()
            if root is None:
                return None
            if is_synthetic_sample_dataset(root):
                return ("Synthetic demo PNGs — fetch real PI_RAW for quality", AMBER)
            info = dataset_summary(root)
            if info.get("kind") == "real_png":
                return ("PNG only (no DNG) — real RAW recommended", AMBER)
            if info.get("kind") == "real_dng":
                return (f"Real PI_RAW ({info.get('paired_folders', 0)} scenes)", INK)
        except Exception:
            pass
        return None

    def _refresh_dataset_hint(self):
        if not hasattr(self, "dataset_hint"):
            return
        hint = self._dataset_quality_hint()
        if hint:
            self.dataset_hint.config(text=f"     {hint[0]}", fg=hint[1])
        else:
            self.dataset_hint.config(text="")

    def _on_source_change(self):
        real = self.source_var.get() == "real"
        for attr in ("dataset_btn", "upload_btn"):
            if hasattr(self, attr):
                getattr(self, attr).set_enabled(real)
        if hasattr(self, "dataset_label"):
            self.dataset_label.config(fg=(SUBTLE if real else "#C4C4C4"))
        # Only real captures can be filtered or have their own noise; in simulated
        # mode the scene + noise are always synthesised, so those knobs are moot.
        self._set_entry_enabled("filter", real)
        if hasattr(self, "sim_noise_cb"):
            try:
                self.sim_noise_cb.config(state="normal" if real else "disabled")
            except tk.TclError:
                pass
        # Noise Std only bites when noise is actually injected: simulated capture,
        # or real capture with "simulate sensor noise" enabled.
        noise_simulated = (not real) or bool(self.sim_noise_var.get())
        self._set_entry_enabled("noise_std", noise_simulated)
        # Temporal Frames only builds a synthetic ground-truth average; with real
        # paired gt (real + no simulated noise) it is ignored.
        if "frames" in self.rows and hasattr(self.rows["frames"], "set_enabled"):
            self.rows["frames"].set_enabled(noise_simulated)
        self._refresh_dataset_hint()

    def _on_mode_change(self):
        # Batch size only applies to batch mode; burst only to temporal mode.
        mode = self.mode_var.get()
        self._set_entry_enabled("batch", mode == "batch")
        self._set_entry_enabled("burst", mode == "temporal")

    def _on_sensor_change(self):
        key = self.rows["sensor"].get() if "sensor" in self.rows else "imx662"
        # Keep the dataset filter aligned with the sensor so real captures for
        # the selected module actually load (a stale imx219 filter loads nothing
        # for imx662 and silently falls back to synthetic).
        if hasattr(self, "filter_var"):
            try:
                from nsa.denoise_hw_data import (DEFAULT_FILTERS_BY_SENSOR,
                                                 default_filter_for_sensor)
                current = tuple((self.filter_var.get() or "").split())
                known = {tuple(v) for v in DEFAULT_FILTERS_BY_SENSOR.values()}
                known.add(())
                if current in known:
                    self.filter_var.set(" ".join(default_filter_for_sensor(key)))
            except Exception:  # noqa: BLE001
                pass
        if not hasattr(self, "sensor_echo"):
            return
        card = next((c for c in SENSOR_CARDS if c["key"] == key), None)
        if card:
            self.sensor_echo.config(
                text=f"     Optimising for {card['name']} ({card['family']}) — "
                     f"{card['specs']}.")

    def _row_get(self, key, default):
        r = self.rows.get(key)
        return r.get() if r is not None else default

    def _grab_when_ready(self, win):
        """Make a Toplevel modal once it is actually viewable.

        Calling grab_set() before the window is mapped raises
        'grab failed: window not viewable', so we poll until it's ready.
        """
        try:
            if not win.winfo_exists():
                return
            if win.winfo_viewable():
                win.grab_set()
            else:
                win.after(40, lambda: self._grab_when_ready(win))
        except tk.TclError:
            win.after(40, lambda: self._grab_when_ready(win))

    def _read_nafnet_topo(self) -> tuple[str, str, str]:
        """NAFNet topology strings (enc / mid / dec) from widgets or cached defaults."""
        if hasattr(self, "naf_enc_var"):
            return (
                (self.naf_enc_var.get() or "").strip(),
                (self.naf_mid_var.get() or "").strip(),
                (self.naf_dec_var.get() or "").strip(),
            )
        t = getattr(self, "_nafnet_topo", {"enc": "", "mid": "", "dec": ""})
        return t["enc"], t["mid"], t["dec"]

    def _write_nafnet_topo(self, enc: str, mid: str, dec: str) -> None:
        self._nafnet_topo = {"enc": enc, "mid": mid, "dec": dec}
        if hasattr(self, "naf_enc_var"):
            self.naf_enc_var.set(enc)
            self.naf_mid_var.set(mid)
            self.naf_dec_var.set(dec)

    def _append_nafnet_cli_args(self, cmd: list) -> None:
        if self._row_get("model_family", "") != "nafnet":
            return
        enc_s, mid_s, dec_s = self._read_nafnet_topo()
        enc = enc_s.split()
        dec = dec_s.split()
        if enc and all(t.isdigit() for t in enc):
            cmd += ["--nafnet-enc", *enc]
            if mid_s.isdigit():
                cmd += ["--nafnet-middle", mid_s]
            if dec and all(t.isdigit() for t in dec):
                cmd += ["--nafnet-dec", *dec]

    def _on_family_change(self):
        self._render_model_options()

    def _on_loss_change(self):
        self._render_loss_options()

    def _render_loss_options(self):
        """Show only the parameter fields that the selected loss actually uses."""
        # Preserve any values the user already typed.
        prev = {}
        for k in ("charbonnier_eps", "huber_delta", "ssim_window", "ssim_weight"):
            var = self.entries.get(k)
            if var is not None:
                prev[k] = var.get()
        for w in self.loss_box.winfo_children():
            w.destroy()
        for k in ("charbonnier_eps", "huber_delta", "ssim_window", "ssim_weight"):
            self.entries.pop(k, None)
            self.entry_widgets.pop(k, None)

        name = self.rows["loss"].get() if "loss" in self.rows else "charbonnier"
        defaults = {"charbonnier_eps": "0.001", "huber_delta": "1.0",
                    "ssim_window": "11", "ssim_weight": "0.2"}

        def field(key, title, desc):
            self._entry_row(self.loss_box, key, title, desc,
                            prev.get(key, defaults[key]))

        if name == "charbonnier":
            field("charbonnier_eps", "Charbonnier eps",
                  "L2→L1 transition — smaller = sharper, larger = smoother")
        elif name in ("l1", "l2"):
            tk.Label(self.loss_box,
                     text=f"     {name.upper()} has no tunable parameters.",
                     bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w", pady=(S(2), 0))
        elif name == "huber":
            field("huber_delta", "Huber delta", "L2→L1 crossover threshold")
        elif name == "ssim":
            field("ssim_window", "SSIM window", "Gaussian window size (odd, e.g. 11)")
        elif name == "charbonnier_ssim":
            field("charbonnier_eps", "Charbonnier eps",
                  "Pixel-term L2→L1 transition")
            field("ssim_weight", "SSIM weight",
                  "Blend: (1-w)·charbonnier + w·(1-SSIM), 0..1")
            field("ssim_window", "SSIM window", "Gaussian window size (odd, e.g. 11)")

    def _render_model_options(self):
        """Render only the Level-3 options that apply to the chosen model family.

        e.g. NAFNet has no separate activation (it uses SimpleGate) and uses
        built-in depthwise convs, so those rows are hidden and the multi-scale
        topology fields appear instead.
        """
        prev = {k: self.rows[k].get()
                for k in ("base_channels", "block_depth", "conv_type", "activation")
                if k in self.rows}
        if hasattr(self, "naf_enc_var"):
            self._write_nafnet_topo(
                self.naf_enc_var.get() or "",
                self.naf_mid_var.get() or "",
                self.naf_dec_var.get() or "",
            )
        for w in self.model_box.winfo_children():
            w.destroy()
        for k in ("base_channels", "block_depth", "conv_type", "activation"):
            self.rows.pop(k, None)
        for attr in ("naf_enc_var", "naf_mid_var", "naf_dec_var"):
            if hasattr(self, attr):
                delattr(self, attr)

        fam = self.rows["model_family"].get()
        topo = getattr(self, "_nafnet_topo", {"enc": "", "mid": "", "dec": ""})

        def cfgrow(key, title, desc, values, default):
            r = ConfigRow(self.model_box, title, desc, values, default)
            r.pack(fill="x", pady=S(6))
            self.rows[key] = r

        cfgrow("base_channels", "Base Channels", "Network width",
               [16, 32, 64], prev.get("base_channels", 32))
        depth_vals = [2, 4, 8]
        if fam == "rednet":
            depth_desc = "RED blocks per encoder stage (minimum 2)"
            depth_vals = [2, 4, 8]
        elif fam in ("unet", "drunet"):
            depth_desc = ("Blocks per U-Net stage — internally halved "
                          "(depth 4 → 2 blocks per stage)")
        elif fam == "nafnet":
            depth_desc = ("NAFBlocks in flat mode — ignored when a custom "
                          "topology is set below")
        elif fam == "restormer":
            depth_desc = "Transformer blocks at each scale"
        else:
            depth_desc = "Stack depth (conv blocks or residual groups)"
        cfgrow("block_depth", "Block Depth", depth_desc,
               depth_vals, max(prev.get("block_depth", 4), 2 if fam == "rednet" else 0))

        if fam == "nafnet":
            tk.Label(self.model_box,
                     text="     NAFNet uses a built-in SimpleGate and depthwise "
                          "convs — no separate activation or conv-type to pick. "
                          "Optionally define a multi-scale topology (leave blank "
                          "for a flat NAFNet).",
                     bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                     justify="left").pack(anchor="w", pady=(S(2), 0))
            self.naf_enc_var = self._entry_row(
                self.model_box, "naf_enc", "NAFNet Encoders",
                "Per-level encoder block counts, e.g. 1 2 2", topo["enc"])
            self.naf_mid_var = self._entry_row(
                self.model_box, "naf_mid", "NAFNet Middle",
                "Bottleneck block count, e.g. 4", topo["mid"])
            self.naf_dec_var = self._entry_row(
                self.model_box, "naf_dec", "NAFNet Decoders",
                "Per-level decoder block counts, e.g. 2 2 1", topo["dec"])
        elif fam == "restormer":
            tk.Label(self.model_box,
                     text="     Restormer uses LayerNorm + transposed self-attention "
                          "+ a GELU gated feed-forward — no separate activation or "
                          "conv-type to pick. Best on the Pi 5 CPU; the attention "
                          "graph gets caveats on INT8 NPUs.",
                     bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                     justify="left").pack(anchor="w", pady=(S(2), 0))
        else:
            cfgrow("conv_type", "Convolution", "Standard or depthwise-separable",
                   ["standard", "depthwise"], prev.get("conv_type", "depthwise"))
            cfgrow("activation", "Activation",
                   "gelu on DeepX forces QAT injection",
                   ["relu", "gelu", "silu"], prev.get("activation", "relu"))

    def _on_eval_change(self):
        sweep = self.eval_var.get() == "sweep"
        if hasattr(self, "model_intro"):
            if sweep:
                self.model_intro.config(
                    text="     Sweep mode: all 9 families are trained & ranked. The "
                         "settings below pin the width / conv / activation used for "
                         "every family (depth is swept); the family choice itself is "
                         "ignored.")
            else:
                self.model_intro.config(
                    text="     Pick a family first — the options adapt to it (e.g. "
                         "NAFNet / Restormer have no separate activation). These are "
                         "the exact parameters that get compiled.")
        if hasattr(self, "all_sensors_cb"):
            try:
                self.all_sensors_cb.config(state="normal" if sweep else "disabled")
            except Exception:
                pass
            if not sweep:
                self.all_sensors_var.set(False)
        if hasattr(self, "hw_note"):
            self.hw_note.config(
                text=("     The sweep compiles for this chip, but the leaderboard "
                      "can re-rank by any Pi chip afterwards." if sweep else ""))
        if getattr(self, "_steps", None):
            if self._steps[self._step]["key"] == "review":
                self._refresh_review()
            self._render_nav()
        if getattr(self, "_wizard_mode", "") == "home":
            self._refresh_home_summary()
            self._render_nav()

    def _choose_dataset(self):
        if self.source_var.get() != "real":
            return
        path = filedialog.askdirectory(title="Select a dataset folder (real captures / paired noisy-gt)")
        if path:
            self.dataset_path = path
            self.upload_files = []
            self.dataset_label.config(text=Path(path).name)
            self._refresh_dataset_hint()

    def _upload_images(self):
        if self.source_var.get() != "real":
            return
        files = filedialog.askopenfilenames(
            title="Upload one or more RAW / image frames",
            filetypes=[("RAW / image", "*.npy *.png *.tif *.tiff *.dng *.raw *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if files:
            self.upload_files = list(files)
            self.dataset_path = None
            n = len(self.upload_files)
            self.dataset_label.config(
                text=(Path(self.upload_files[0]).name if n == 1
                      else f"{n} images uploaded"))
            self._refresh_dataset_hint()

    def _choose_raw(self):
        path = filedialog.askopenfilename(
            title="Select IMX662 Bayer RAW frame",
            filetypes=[("RAW / image", "*.npy *.png *.tif *.tiff *.dng *.raw *.jpg"),
                       ("All files", "*.*")])
        if path:
            self.input_raw = path
            self.raw_label.config(text=Path(path).name)

    def _materialise_uploads(self):
        """Copy uploaded files into a temp folder so they pass as one --dataset."""
        if not self.upload_files:
            return None
        import shutil
        dest = ROOT / "outputs" / "_uploads"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        for f in self.upload_files:
            try:
                shutil.copy(f, dest / Path(f).name)
            except Exception:
                pass
        return str(dest)

    def _noop(self):
        pass

    def _set_ui_scale(self, value, win=None):
        """Live-apply a new text/UI scale and rebuild the whole window."""
        global SCALE
        SCALE = max(0.8, min(3.0, float(value)))
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        try:
            self.tk.call("tk", "scaling", SCALE if USE_TK_SCALING else 1.0)
        except Exception:
            pass
        self._apply_geometry()
        self._build_chrome()

    def _apply_geometry(self):
        """Size the window to the preferred dims, clamped to the screen so the
        pinned footer stays visible even at large text scales."""
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        except Exception:
            sw, sh = 1920, 1080
        w = min(S(980), sw - S(40))
        h = min(S(680), sh - S(80))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(min(S(820), w), min(S(560), h))

    def _app_options(self):
        win = tk.Toplevel(self)
        win.title("App Options")
        win.configure(bg=WHITE)
        win.transient(self)
        win.resizable(False, False)
        pad = S(24)
        wrap = tk.Frame(win, bg=WHITE)
        wrap.pack(fill="both", expand=True, padx=pad, pady=pad)

        tk.Label(wrap, text="App Options", bg=WHITE, fg=INK,
                 font=font(16, "bold")).pack(anchor="w")
        tk.Label(wrap, text="Adjust how the interface looks.", bg=WHITE,
                 fg=SUBTLE, font=font(10)).pack(anchor="w", pady=(S(2), S(14)))

        tk.Label(wrap, text="TEXT SIZE", bg=WHITE, fg=RASPBERRY,
                 font=font(9, "bold")).pack(anchor="w")
        tk.Label(wrap, text=f"Current scale: {SCALE:.2f}×  "
                 f"(or set NSA_UI_SCALE before launch)", bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(anchor="w", pady=(S(2), S(8)))
        sizes = tk.Frame(wrap, bg=WHITE)
        sizes.pack(fill="x", pady=(0, S(16)))
        for label, val in [("Small", 1.0), ("Medium", 1.3),
                           ("Large", 1.6), ("Extra Large", 1.9)]:
            RoundButton(sizes, label, lambda v=val: self._set_ui_scale(v, win),
                        kind="secondary", width=128, height=40).pack(
                            side="left", padx=(0, S(8)))

        tk.Frame(wrap, bg=LINE, height=1).pack(fill="x", pady=(0, S(14)))
        actions = tk.Frame(wrap, bg=WHITE)
        actions.pack(fill="x")
        RoundButton(actions, "OPEN OUTPUTS", self._open_outputs,
                    kind="secondary", width=170, height=42).pack(side="left")
        RoundButton(actions, "CLOSE", win.destroy,
                    kind="primary", width=130, height=42).pack(side="right")

        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        self._grab_when_ready(win)

    # -- Run view -------------------------------------------------------------
    def _run(self):
        if self.eval_var.get() == "sweep":
            self._run_command(self._build_sweep_command(), "Searching…",
                              "Training & ranking model variants for this target")
        else:
            self._run_command(self._build_command(), "Compiling…",
                              "Running the 6-level optimization stack")

    def _run_command(self, cmd, title="Working…", subtitle=""):
        self._run_cmd = cmd
        pad = S(34)
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.sidebar.reset()
        for w in self.main.winfo_children():
            w.destroy()

        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(28), S(4)))
        tk.Label(header, text=title, bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        self.status = tk.Label(header, text=subtitle or "Working",
                               bg=WHITE, fg=SUBTLE, font=font(10))
        self.status.pack(anchor="w", pady=(S(2), 0))

        self.pbar = ttk.Progressbar(self.main, mode="indeterminate",
                                    style="Rpi.Horizontal.TProgressbar")
        self.pbar.pack(fill="x", padx=pad, pady=(S(12), S(6)))
        self.pbar.start(12)

        con = tk.Frame(self.main, bg=FIELD)
        con.pack(fill="both", expand=True, padx=pad, pady=(S(6), S(6)))
        mono = "Cascadia Mono" if "Cascadia Mono" in tkfont.families() else "Consolas"
        self.log = tk.Text(con, bg=FIELD, fg=INK, bd=0, relief="flat",
                           font=(mono, FT(9)), wrap="none", padx=S(14), pady=S(12))
        sb = ttk.Scrollbar(con, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("warn", foreground="#C98A1B")
        self.log.tag_configure("ok", foreground=GREEN)
        self.log.tag_configure("rasp", foreground=RASPBERRY)

        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad)
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(fill="x", padx=pad, pady=S(16))
        self.back_btn = RoundButton(footer, "BACK", self._back, kind="secondary",
                                    width=130, height=44)
        self.back_btn.pack(side="left")
        self.open_btn = RoundButton(footer, "OPEN OUTPUTS", self._open_outputs,
                                    kind="primary", width=180, height=44)
        self.open_btn.pack(side="right")
        self.open_btn.set_enabled(False)

        self._start_process()

    def _build_command(self):
        cmd = [sys.executable, str(ROOT / "run_demo.py"), "--no-window"]
        for key in ("sensor", "hardware", "model_family", "base_channels",
                    "block_depth", "gain", "steps", "frames"):
            row = self.rows.get(key)
            if row is None:
                continue
            flag = "--" + key.replace("_", "-")
            cmd += [flag, row.get()]
        fam = self.rows["model_family"].get()
        if fam not in ("nafnet", "restormer"):
            for key in ("conv_type", "activation"):
                row = self.rows.get(key)
                if row is None:
                    continue
                cmd += ["--" + key.replace("_", "-"), row.get()]

        if not self.quantize_var.get():
            cmd += ["--no-quantize"]
        if self.qat_var.get():
            cmd += ["--qat"]
        mode = self.mode_var.get()
        if mode == "batch":
            bs = (self.batch_var.get() or "6").strip()
            cmd += ["--batch", bs if bs.isdigit() else "6"]
        elif mode == "temporal":
            cmd += ["--temporal"]
            bu = (self.burst_var.get() or "8").strip()
            cmd += ["--burst", bu if bu.isdigit() else "8"]

        # Custom NAFNet topology (nafnet family only).
        self._append_nafnet_cli_args(cmd)

        if self.source_var.get() == "real":
            cmd += ["--real"]
            dataset = self.dataset_path or self._materialise_uploads()
            if dataset:
                cmd += ["--dataset", dataset]
            elif (ROOT / "datasets" / "PI_RAW").exists():
                cmd += ["--dataset", str(ROOT / "datasets" / "PI_RAW")]
            if self.sim_noise_var.get():
                cmd += ["--simulate-noise"]
            tokens = (self.filter_var.get() or "").split()
            if tokens:
                cmd += ["--filter", *tokens]
        else:
            cmd += ["--simulated"]

        self._append_noise_std_cli_args(cmd)
        self._append_loss_cli_args(cmd)

        if self.input_raw:
            cmd += ["--input-raw", self.input_raw]
        if self.hf_model_id:
            cmd += ["--hf-model", self.hf_model_id]
            if self.hf_weight:
                cmd += ["--hf-weight", self.hf_weight]
        return cmd

    def _build_sweep_command(self):
        # Search all model families × depths at the chosen width, keeping the
        # chosen conv/activation/width fixed. This bounds the grid to ~18 configs
        # and uses enough calibration steps that PSNR actually separates the
        # models (too few steps floors quality and the sweep just picks the
        # fastest model). Change the width and re-sweep to explore other widths.
        cmd = [sys.executable, str(ROOT / "search.py"),
               "--hardware", self.rows["hardware"].get(),
               "--sensor", self.rows["sensor"].get(),
               "--gain", self.rows["gain"].get(),
               "--base-channels", self._row_get("base_channels", "32"),
               "--conv-type", self._row_get("conv_type", "depthwise"),
               "--activation", self._row_get("activation", "relu"),
               "--search-steps", "120",
               "--patch-size", "192",
               "--top", "10",
               "--no-final-run"]
        if self.all_sensors_var.get():
            cmd += ["--all-sensors"]
        if self.source_var.get() == "real":
            dataset = self.dataset_path or self._materialise_uploads()
            if not dataset and (ROOT / "datasets" / "PI_RAW").exists():
                dataset = str(ROOT / "datasets" / "PI_RAW")
            if dataset:
                cmd += ["--real", "--dataset", dataset]
            if self.sim_noise_var.get():
                cmd += ["--simulate-noise"]
            tokens = (self.filter_var.get() or "").split()
            if tokens:
                cmd += ["--filter", *tokens]
        else:
            cmd += ["--simulated"]
        self._append_noise_std_cli_args(cmd)
        self._append_loss_cli_args(cmd)
        return cmd

    def _append_loss_cli_args(self, cmd):
        """Append --loss and only the parameter flags the chosen loss uses."""
        row = self.rows.get("loss")
        if row is None:
            return
        name = row.get()
        cmd += ["--loss", name]
        param_flags = {
            "charbonnier_eps": "--charbonnier-eps",
            "huber_delta": "--huber-delta",
            "ssim_window": "--ssim-window",
            "ssim_weight": "--ssim-weight",
        }
        for key, flag in param_flags.items():
            var = self.entries.get(key)
            if var is None:
                continue
            raw = (var.get() or "").strip()
            if not raw:
                continue
            try:
                float(raw)
            except ValueError:
                continue
            cmd += [flag, raw]

    def _append_noise_std_cli_args(self, cmd):
        """Append --noise-std when the user typed a valid read-noise override."""
        var = getattr(self, "noise_std_var", None)
        if var is None:
            return
        raw = (var.get() or "").strip()
        if not raw:
            return
        try:
            cmd += ["--noise-std", str(float(raw))]
        except ValueError:
            pass

    def _build_cache(self):
        dataset = self.dataset_path or self._materialise_uploads()
        if not dataset:
            messagebox.showinfo(
                "Patch-cache builder",
                "Pick a dataset first: set Capture Source to 'Real captures' and "
                "choose a folder or upload images, then click BUILD CACHE.")
            return
        cmd = [sys.executable, str(ROOT / "cache.py"), "--dataset", dataset,
               "--sensor", self.rows["sensor"].get(),
               "--gain", self.rows["gain"].get(),
               "--patch", "128", "--per-image", "6"]
        if self.sim_noise_var.get():
            cmd += ["--simulate-noise"]
        tokens = (self.filter_var.get() or "").split()
        if tokens:
            cmd += ["--filter", *tokens]
        self._run_command(cmd, "Building patch cache…",
                          "Scanning the dataset into detail-scored patches")

    # -- Hugging Face Hub browser -------------------------------------------
    def _refresh_hf_active_label(self):
        if not hasattr(self, "hf_active_lbl"):
            return
        if self.hf_model_id:
            txt = f"Active: {self.hf_model_id}"
            if self.hf_weight:
                txt += f" · {self.hf_weight}"
            self.hf_active_lbl.config(text=txt, fg=GREEN)
        else:
            self.hf_active_lbl.config(
                text="No Hub model — built-in arch trains from scratch.",
                fg=SUBTLE)

    def _hf_clear_selection(self):
        self.hf_model_id = None
        self.hf_weight = None
        self._refresh_hf_active_label()

    def _hf_browser(self):
        try:
            from nsa import hub  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Hugging Face",
                                 f"Could not load the Hub module:\n{exc}")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Hugging Face — model sourcing")
        dlg.configure(bg=WHITE)
        dlg.transient(self)
        try:
            dlg.geometry(f"{S(740)}x{S(640)}+{self.winfo_rootx()+S(60)}"
                         f"+{self.winfo_rooty()+S(30)}")
        except Exception:
            pass
        self._grab_when_ready(dlg)
        self._hf_dlg = dlg
        self._hf_q = queue.Queue()
        self._hf_rows = []
        self._hf_busy = False
        self._hf_lock = ROOT / "outputs" / "hf_lock.json"

        pad = tk.Frame(dlg, bg=WHITE)
        pad.pack(fill="both", expand=True, padx=S(20), pady=S(16))
        tk.Label(pad, text="Hugging Face model sourcing", bg=WHITE, fg=INK,
                 font=font(16, "bold")).pack(anchor="w")
        tk.Label(pad, text="1 license-safe  →  2 benchmark small  →  3 test the gap "
                 " →  4 freeze the weights", bg=WHITE, fg=RASPBERRY,
                 font=font(9, "bold")).pack(anchor="w", pady=(S(2), S(8)))

        ctl = tk.Frame(pad, bg=WHITE); ctl.pack(fill="x")
        self._hf_license = tk.StringVar(value="both")
        lf = tk.Frame(ctl, bg=WHITE); lf.pack(fill="x", pady=S(2))
        tk.Label(lf, text="License", bg=WHITE, fg=INK, font=font(10, "bold"),
                 width=9, anchor="w").pack(side="left")
        for val, label in [("apache-2.0", "Apache-2.0"), ("mit", "MIT"),
                           ("both", "Both")]:
            tk.Radiobutton(lf, text=label, variable=self._hf_license, value=val,
                           bg=WHITE, fg=INK, selectcolor=WHITE,
                           activebackground=WHITE, font=font(10),
                           highlightthickness=0, bd=0,
                           command=self._hf_run_search).pack(side="left",
                                                             padx=(0, S(8)))

        self._hf_size = tk.StringVar(value="any")
        self._hf_category = tk.StringVar(value=HF_CATEGORIES[0][0])
        self._hf_query = tk.StringVar(value="")

        # Category picker — pre-loads a relevant model list, no query needed.
        crow = tk.Frame(ctl, bg=WHITE); crow.pack(fill="x", pady=S(2))
        tk.Label(crow, text="Category", bg=WHITE, fg=INK, font=font(10, "bold"),
                 width=9, anchor="w").pack(side="left")
        cat_cb = ttk.Combobox(crow, textvariable=self._hf_category,
                              values=[c[0] for c in HF_CATEGORIES],
                              state="readonly", width=28, style="Rpi.TCombobox")
        cat_cb.pack(side="left")
        cat_cb.bind("<<ComboboxSelected>>", lambda _e: self._hf_run_search())

        srow2 = tk.Frame(ctl, bg=WHITE); srow2.pack(fill="x", pady=S(2))
        tk.Label(srow2, text="Size", bg=WHITE, fg=INK, font=font(10, "bold"),
                 width=9, anchor="w").pack(side="left")
        size_cb = ttk.Combobox(srow2, textvariable=self._hf_size,
                               values=["any", "tiny", "small", "mid", "large"],
                               state="readonly", width=12, style="Rpi.TCombobox")
        size_cb.pack(side="left")
        size_cb.bind("<<ComboboxSelected>>", lambda _e: self._hf_run_search())
        tk.Label(srow2, text="  tiny <1B · small 1-8B  (denoisers are usually tiny)",
                 bg=WHITE, fg=SUBTLE, font=font(8)).pack(side="left")

        # Optional free-text refinement (not required — list loads automatically).
        qrow = tk.Frame(ctl, bg=WHITE); qrow.pack(fill="x", pady=S(2))
        tk.Label(qrow, text="Refine", bg=WHITE, fg=INK, font=font(10, "bold"),
                 width=9, anchor="w").pack(side="left")
        ent = ttk.Entry(qrow, textvariable=self._hf_query, width=28, font=font(10))
        ent.pack(side="left")
        ent.bind("<Return>", lambda _e: self._hf_run_search())
        tk.Label(qrow, text="  optional keyword", bg=WHITE, fg=SUBTLE,
                 font=font(8)).pack(side="left")

        srow = tk.Frame(ctl, bg=WHITE); srow.pack(fill="x", pady=(S(6), 0))
        self._hf_search_btn = RoundButton(srow, "REFRESH LIST", self._hf_run_search,
                                          kind="primary", width=160, height=38)
        self._hf_search_btn.pack(side="left")
        self._hf_status = tk.Label(srow, text="", bg=WHITE, fg=SUBTLE, font=font(9),
                                   wraplength=S(420), justify="left")
        self._hf_status.pack(side="left", padx=(S(10), 0))

        tk.Frame(pad, bg=LINE, height=1).pack(fill="x", pady=(S(10), S(4)))
        hdr = tk.Frame(pad, bg=WHITE); hdr.pack(fill="x")
        for label, w in [("MODEL", 42), ("PARAMS", 8), ("TIER", 6), ("LICENSE", 11)]:
            tk.Label(hdr, text=label, bg=WHITE, fg=SUBTLE, font=font(8, "bold"),
                     width=w, anchor="w").pack(side="left")
        self._hf_results = tk.Frame(pad, bg=WHITE)
        self._hf_results.pack(fill="both", expand=True, pady=(S(2), 0))

        tk.Frame(pad, bg=LINE, height=1).pack(fill="x", pady=(S(6), S(6)))
        ftr = tk.Frame(pad, bg=WHITE); ftr.pack(fill="x")
        RoundButton(ftr, "CLOSE", dlg.destroy, kind="secondary",
                    width=110, height=40).pack(side="left")
        RoundButton(ftr, "OPEN LOCK FILE", self._hf_open_lock, kind="secondary",
                    width=170, height=40).pack(side="left", padx=(S(8), 0))
        tk.Label(ftr, text="frozen → outputs/hf_lock.json", bg=WHITE, fg=SUBTLE,
                 font=font(8)).pack(side="right")

        self._hf_set_status("Loading relevant models…", RASPBERRY)
        self.after(150, lambda: self._hf_poll(dlg))
        # Auto-load a relevant list immediately — no manual search needed.
        self.after(250, self._hf_run_search)

    def _hf_set_status(self, text, color=None):
        if hasattr(self, "_hf_status") and self._hf_status.winfo_exists():
            self._hf_status.config(text=text, fg=color or SUBTLE)

    def _hf_run_search(self):
        if self._hf_busy:
            return
        from nsa import hub
        licenses = (["apache-2.0", "mit"] if self._hf_license.get() == "both"
                    else [self._hf_license.get()])
        size = self._hf_size.get()
        cat_q, task = HF_CATEGORY_MAP.get(self._hf_category.get(),
                                          ("", "image-to-image"))
        refine = self._hf_query.get().strip()
        query = refine or cat_q
        self._hf_busy = True
        self._hf_search_btn.set_enabled(False)
        self._hf_set_status(f"Loading {self._hf_category.get().lower()} models…",
                            RASPBERRY)

        def work():
            try:
                rows = hub.search_models(query=query, licenses=licenses, task=task,
                                         size=size, limit=10)
                self._hf_q.put(("search_ok", rows))
            except Exception as exc:  # noqa: BLE001
                self._hf_q.put(("search_err", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _hf_poll(self, dlg):
        if not dlg.winfo_exists() or getattr(self, "_hf_dlg", None) is not dlg:
            return
        try:
            kind, payload = self._hf_q.get_nowait()
        except queue.Empty:
            self.after(150, lambda: self._hf_poll(dlg))
            return
        if kind == "search_ok":
            self._hf_busy = False
            self._hf_search_btn.set_enabled(True)
            self._hf_rows = payload
            self._hf_render_results()
            self._hf_set_status(
                f"{len(payload)} license-safe model(s). "
                f"Click DOWNLOAD & USE to run one in compile." if payload else
                "No models in this category for that license/size — try another "
                "Category, Size = any, or License = Both.",
                GREEN if payload else AMBER)
        elif kind == "search_err":
            self._hf_busy = False
            self._hf_search_btn.set_enabled(True)
            self._hf_set_status(payload, RASPBERRY)
        elif kind == "freeze_ok":
            self._hf_busy = False
            e = payload
            self._hf_set_status(
                f"Froze {e['id']} @ {e['sha'][:12]} · {e['license']} · "
                f"{e['params_human']}. Click DOWNLOAD & USE to run it.", GREEN)
            self._hf_render_results()
        elif kind == "use_ok":
            self._hf_busy = False
            e, weight = payload
            self.hf_model_id = e["id"]
            self.hf_weight = weight
            self._refresh_hf_active_label()
            self._hf_set_status(
                f"Ready to compile: {e['id']}"
                + (f" · {weight}" if weight else ""), GREEN)
            try:
                messagebox.showinfo(
                    "Hugging Face",
                    f"Hub model set for the next compile:\n\n{e['id']}\n"
                    f"{('Weight: ' + weight) if weight else ''}\n\n"
                    "Click RUN COMPILE (home or results) to load these weights.")
            except Exception:
                pass
        elif kind == "use_err":
            self._hf_busy = False
            self._hf_set_status(payload, RASPBERRY)
        elif kind == "freeze_err":
            self._hf_busy = False
            self._hf_set_status(payload, RASPBERRY)
        self.after(150, lambda: self._hf_poll(dlg))

    def _hf_render_results(self):
        from nsa import hub
        holder = self._hf_results
        for w in holder.winfo_children():
            w.destroy()
        if not self._hf_rows:
            tk.Label(holder, text="No models to show — pick another Category, set "
                     "Size = any, or License = Both.", bg=WHITE, fg=SUBTLE,
                     font=font(9)).pack(anchor="w", pady=S(8))
            return
        locked = {e.get("id") for e in hub.load_lock(self._hf_lock)}
        tcol = {"tiny": SUBTLE, "small": GREEN, "mid": AMBER,
                "large": RASPBERRY, "xl": RASPBERRY}
        for r in self._hf_rows:
            row = tk.Frame(holder, bg=WHITE); row.pack(fill="x", pady=1)
            tk.Label(row, text=r.get("id", ""), bg=WHITE, fg=INK, font=font(9),
                     width=42, anchor="w").pack(side="left")
            tk.Label(row, text=hub.human_params(r.get("params")), bg=WHITE, fg=INK,
                     font=font(9, "bold"), width=8, anchor="w").pack(side="left")
            t = r.get("tier") or "—"
            tk.Label(row, text=t, bg=WHITE, fg=tcol.get(t, SUBTLE), font=font(9),
                     width=6, anchor="w").pack(side="left")
            tk.Label(row, text=r.get("license", "?"), bg=WHITE, fg=GREEN,
                     font=font(9), width=11, anchor="w").pack(side="left")
            if r.get("id") in locked:
                tk.Label(row, text="✓ frozen", bg=WHITE, fg=GREEN,
                         font=font(8, "bold")).pack(side="left", padx=(S(4), 0))
            else:
                RoundButton(row, "FREEZE", lambda rr=r: self._hf_freeze(rr),
                            kind="secondary", width=72, height=30).pack(side="left")
            RoundButton(row, "DOWNLOAD & USE", lambda rr=r: self._hf_download_use(rr),
                        kind="primary", width=130, height=30).pack(side="left", padx=(S(4), 0))

    def _hf_download_use(self, r):
        if self._hf_busy:
            return
        from nsa import hub
        from nsa.hf_runner import pick_weight_file
        mid = r.get("id")
        fam = self._row_get("model_family", "nafnet")
        self._hf_busy = True
        self._hf_set_status(f"Downloading {mid} — fetching pinned snapshot…", RASPBERRY)

        def work():
            try:
                e = hub.freeze_model(mid, lock_path=self._hf_lock, download=True)
                files = []
                if e.get("local_path"):
                    from pathlib import Path
                    files = [p.name for p in Path(e["local_path"]).rglob("*") if p.is_file()]
                if not files:
                    files = hub.model_details(mid).get("files") or []
                weight = pick_weight_file(files, family=fam, hint=mid)
                self._hf_q.put(("use_ok", (e, weight)))
            except Exception as exc:  # noqa: BLE001
                self._hf_q.put(("use_err", f"Download failed: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def _hf_freeze(self, r):
        if self._hf_busy:
            return
        from nsa import hub
        mid = r.get("id")
        self._hf_busy = True
        self._hf_set_status(f"Freezing {mid} — resolving exact commit SHA…", RASPBERRY)

        def work():
            try:
                e = hub.freeze_model(mid, lock_path=self._hf_lock)
                self._hf_q.put(("freeze_ok", e))
            except Exception as exc:  # noqa: BLE001
                self._hf_q.put(("freeze_err", f"Freeze failed: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def _hf_open_lock(self):
        """Show the frozen-model manifest in-app (avoids xdg-open / X11 over SSH)."""
        self._view_text_file(
            ROOT / "outputs" / "hf_lock.json",
            "Hugging Face lock file",
            missing_msg=("No models frozen yet. Search the Hub, then click "
                         "FREEZE on a model to lock its commit hash."))

    def _live_test(self):
        """Live camera: Pi over SSH from AI server, local LiveView elsewhere."""
        if not (ROOT / "outputs" / "model.pt").exists():
            if not messagebox.askyesno(
                "Live testing",
                "No compiled model checkpoint (outputs/model.pt) was found yet.\n\n"
                "Live testing will rebuild and quick-calibrate a model first "
                "(a few seconds). Continue?"):
                return
        try:
            from nsa.pi_remote import (run_live_on_pi, should_use_pi_remote,
                                       pi_live_stream_info)
            if should_use_pi_remote(ROOT):
                err = run_live_on_pi(ROOT)
                if err is None:
                    streaming, url = pi_live_stream_info(ROOT)
                    if streaming:
                        messagebox.showinfo(
                            "Live testing",
                            "Started live.py on the Pi's CSI camera over SSH.\n\n"
                            f"Watch it live in your browser:\n    {url}\n\n"
                            "(Give it a few seconds to warm up. Works from your "
                            "desk — no monitor or VNC on the Pi needed.)\n"
                            "AI-server SSH log: outputs/pi_live.log")
                    else:
                        messagebox.showinfo(
                            "Live testing",
                            "Started live.py on the Pi's CSI camera over SSH.\n\n"
                            "The RAW | DENOISED window opens on the MONITOR "
                            "ATTACHED TO THE PI. Press q or ESC there to stop.\n\n"
                            "Pi in a remote room? Set pi_live.stream: true in "
                            "config.yaml to watch from your browser instead.\n"
                            "AI-server SSH log: outputs/pi_live.log")
                    return
                messagebox.showerror("Pi live testing", err)
                return
            src = "opencv" if sys.platform.startswith("win") else "auto"
            LiveView(self, source=src)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Live testing", str(exc))

    def _build_deploy(self):
        if not (ROOT / "outputs" / "summary.json").exists():
            messagebox.showinfo(
                "Deployment package",
                "Run a compile first (RUN COMPILE) so there are artifacts to "
                "package, then click BUILD PACKAGE.")
            return
        self._run_command([sys.executable, str(ROOT / "deploy.py")],
                          "Building deployment package…",
                          "Bundling artifacts + flash instructions")

    def _export_package(self):
        """Build the transferable hardware package from the last compile."""
        if not (ROOT / "outputs" / "summary.json").exists():
            messagebox.showinfo(
                "Export package",
                "Compile a model first, then export the transferable package.")
            return
        self._run_command([sys.executable, str(ROOT / "deploy.py")],
                          "Exporting transferable package…",
                          "Bundling device binary + ONNX + flash instructions (.zip)")

    def _reveal(self, target):
        """Open the folder containing ``target`` (a file or dir path)."""
        path = Path(target)
        folder = path.parent if path.suffix else path
        if not self._try_open_os_path(folder, "Package"):
            try:
                messagebox.showinfo("Package", f"Transferable package:\n{target}")
            except Exception:
                pass

    def _start_process(self):
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"

        cmd = getattr(self, "_run_cmd", None) or self._build_command()

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, env=env, cwd=str(ROOT))
                for line in self.proc.stdout:
                    self.q.put(line.rstrip("\n"))
                self.proc.wait()
            except Exception as exc:
                self.q.put(f"[error] {exc}")
            self.q.put("__DONE__")

        threading.Thread(target=worker, daemon=True).start()
        self.after(60, self._drain)

    def _drain(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line == "__DONE__":
                    self._finish()
                    return
                self._append(line)
        except queue.Empty:
            pass
        self.after(60, self._drain)

    def _append(self, line: str):
        for i, (num, _name) in enumerate(LEVELS[:6]):
            if f"LEVEL {num} " in line or f"LEVEL {num}  " in line:
                self.sidebar.set_active(i)
                tail = line.split("·")[-1].strip()
                self.status.config(text=tail[:80] or "Compiling…")
        if "PROTOTYPE PERFORMANCE REPORT" in line:
            self.sidebar.all_done()

        tag = None
        if "▲" in line or "WARNING" in line:
            tag = "warn"
        elif "✓" in line:
            tag = "ok"
        elif "LEVEL" in line or "FINAL PARETO" in line:
            tag = "rasp"
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")

    def _finish(self):
        self.pbar.stop()
        self.pbar.pack_forget()
        self.sidebar.all_done()
        # Stash the full streamed log before we rebuild the view.
        try:
            self._full_log = self.log.get("1.0", "end").strip()
        except Exception:
            self._full_log = ""
        cmd = getattr(self, "_run_cmd", []) or []
        is_compile = any("run_demo.py" in str(c) for c in cmd)
        is_sweep = any("search.py" in str(c) for c in cmd)
        self.status.config(
            text="Compilation complete — artifacts written to outputs/" if is_compile
            else ("Sweep complete — ranked leaderboard below" if is_sweep
                  else "Done — see the log above and outputs/ for results"))
        if is_compile:
            self._show_result()
        elif is_sweep:
            self._show_ranking()
        else:
            try:
                self.open_btn.set_enabled(True)
            except Exception:
                pass

    def _load_summary(self) -> dict:
        try:
            return json.loads((ROOT / "outputs" / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _metric_card(self, parent, label, value, accent=INK, sub=None):
        card = tk.Frame(parent, bg=FIELD)
        card.pack(side="left", fill="both", expand=True, padx=(0, S(10)))
        inner = tk.Frame(card, bg=FIELD)
        inner.pack(fill="both", expand=True, padx=S(14), pady=S(12))
        tk.Label(inner, text=label.upper(), bg=FIELD, fg=SUBTLE,
                 font=font(8, "bold")).pack(anchor="w")
        tk.Label(inner, text=value, bg=FIELD, fg=accent,
                 font=font(16, "bold")).pack(anchor="w", pady=(S(2), 0))
        if sub:
            tk.Label(inner, text=sub, bg=FIELD, fg=SUBTLE,
                     font=font(8)).pack(anchor="w")

    def _target_card(self, parent, t):
        """One per-chip suitability row in the results screen."""
        verdict = t.get("verdict", "")
        vcol, vtxt = {
            "SUITABLE": (GREEN, "✓  SUITABLE"),
            "CAVEATS": ("#C98A1B", "▲  WITH CAVEATS"),
            "UNSUITABLE": (RASPBERRY, "✗  NOT RECOMMENDED"),
        }.get(verdict, (INK, verdict))
        selected = t.get("selected")
        bg = "#FCEEF2" if selected else FIELD

        card = tk.Frame(parent, bg=bg, highlightthickness=(2 if selected else 0),
                        highlightbackground=RASPBERRY, highlightcolor=RASPBERRY)
        card.pack(fill="x", pady=S(4))
        inner = tk.Frame(card, bg=bg)
        inner.pack(fill="x", padx=S(12), pady=S(8))
        inner.columnconfigure(0, weight=1)

        left = tk.Frame(inner, bg=bg); left.grid(row=0, column=0, sticky="w")
        title = t.get("label", t.get("key", ""))
        if selected:
            title += "   ◀ selected"
        tk.Label(left, text=title, bg=bg, fg=INK,
                 font=font(11, "bold")).pack(anchor="w")
        budget = t.get("budget_kb", 0)
        if budget and budget < 500000:
            mem = f"{100*t.get('mem_frac',0):.0f}% of {budget:,.0f} KB SRAM"
            if t.get("tiled"):
                mem += " (+tiling)"
        else:
            mem = "fits system RAM"
        specs = (f"{t.get('precision','')}  ·  {mem}  ·  "
                 f"~{t.get('fps','—')} FPS  ·  {t.get('format','')}")
        tk.Label(left, text=specs, bg=bg, fg=SUBTLE,
                 font=font(9)).pack(anchor="w")
        for n in t.get("notes", []):
            tk.Label(left, text="· " + n, bg=bg, fg="#C98A1B", font=font(8),
                     wraplength=S(440), justify="left").pack(anchor="w")

        tk.Label(inner, text=vtxt, bg=bg, fg=vcol,
                 font=font(10, "bold")).grid(row=0, column=1, sticky="e",
                                             padx=(S(8), 0))

    def _show_result(self):
        s = self._load_summary()
        pad = S(34)
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()

        # -- Header ----------------------------------------------------------
        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(22), S(2)))
        tk.Label(header, text="Compilation Complete", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        if s:
            m = s.get("model", {})
            kind = s.get("kind", "compile")
            kind_lbl = "SWEEP" if kind == "sweep" else "COMPILE"
            subtitle = (f"{kind_lbl}   ·   {s.get('hardware_name','')}   ·   "
                        f"{m.get('display') or ''}   ·   "
                        f"{s.get('precision','')}")
        else:
            subtitle = "Results written to outputs/"
        tk.Label(header, text=subtitle, bg=WHITE, fg=SUBTLE,
                 font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        # -- Footer (pinned) -------------------------------------------------
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(12))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        foot_top = tk.Frame(footer, bg=WHITE)
        foot_top.pack(fill="x")
        foot_bot = tk.Frame(footer, bg=WHITE)
        foot_bot.pack(fill="x", pady=(S(6), 0))
        RoundButton(foot_top, "RUN AGAIN", self._back, kind="secondary",
                    width=120, height=40).pack(side="left")
        RoundButton(foot_top, "LIVE TEST", self._live_test, kind="primary",
                    width=120, height=40).pack(side="left", padx=(S(6), 0))
        RoundButton(foot_top, "FULL LOG", self._show_log, kind="secondary",
                    width=110, height=40).pack(side="right")
        if s and s.get("package_zip"):
            RoundButton(foot_top, "OPEN PKG",
                        lambda: self._reveal(s.get("package_zip")), kind="primary",
                        width=120, height=40).pack(side="right", padx=(0, S(6)))
        else:
            RoundButton(foot_top, "EXPORT", self._export_package, kind="primary",
                        width=110, height=40).pack(side="right", padx=(0, S(6)))
        RoundButton(foot_bot, "HISTORY", self._show_history, kind="secondary",
                    width=100, height=40).pack(side="left")
        RoundButton(foot_bot, "OPEN OUTPUTS", self._open_outputs, kind="secondary",
                    width=130, height=40).pack(side="left", padx=(S(6), 0))

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(6), 0))
        body = self._make_scrollable(outer)

        if not s:
            tk.Label(body, text="No summary found. See the full log for details.",
                     bg=WHITE, fg=SUBTLE, font=font(11)).pack(anchor="w", pady=S(10))

        # -- Fitness banner --------------------------------------------------
        if s:
            grade = s.get("grade", "")
            gcol = GRADE_COLORS.get(grade, INK)
            fb = tk.Frame(body, bg=WHITE)
            fb.pack(fill="x", pady=(S(8), S(10)))
            tk.Label(fb, text=f"{s.get('fitness','—')}", bg=WHITE, fg=gcol,
                     font=font(34, "bold")).pack(side="left")
            tk.Label(fb, text="/ 100", bg=WHITE, fg=SUBTLE,
                     font=font(14)).pack(side="left", padx=(S(6), S(12)), anchor="s",
                                         pady=(0, S(8)))
            tk.Label(fb, text=grade, bg=WHITE, fg=gcol,
                     font=font(13, "bold")).pack(side="left", anchor="s",
                                                 pady=(0, S(10)))
            tk.Label(fb, text="  Pareto fitness score", bg=WHITE, fg=SUBTLE,
                     font=font(10)).pack(side="left", anchor="s", pady=(0, S(11)))

            # -- Metric cards -----------------------------------------------
            row1 = tk.Frame(body, bg=WHITE); row1.pack(fill="x", pady=(0, S(10)))
            self._metric_card(row1, "Image quality",
                              f"{s.get('psnr_out','—')} dB",
                              accent=GREEN,
                              sub=f"+{s.get('psnr_gain','—')} dB vs input ({s.get('psnr_in','—')} dB)")
            self._metric_card(row1, "Latency / speed",
                              f"{s.get('latency_ms','—')} ms",
                              sub=f"{s.get('fps','—')} FPS  ·  {s.get('hardware','')}")
            self._metric_card(row1, "INT8 drop",
                              f"{s.get('quant_drop_db','—'):+} dB"
                              if isinstance(s.get('quant_drop_db'), (int, float))
                              else "—",
                              sub="FP32 → INT8")

            row2 = tk.Frame(body, bg=WHITE); row2.pack(fill="x", pady=(0, S(14)))
            self._metric_card(row2, "Weight memory",
                              f"{s.get('weight_kb','—')} KB", sub="storage / flash")
            budget = s.get("sram_budget_kb", 0)
            act = s.get("act_kb", 0)
            sub_sram = (f"{100*act/budget:.0f}% of {budget:,.0f} KB SRAM"
                        if budget and budget < 500000 else "peak activation")
            self._metric_card(row2, "Activation memory",
                              f"{act:,.0f} KB", sub=sub_sram)
            self._metric_card(row2, "Model size",
                              f"{s.get('model',{}).get('params','—'):,}"
                              if isinstance(s.get('model',{}).get('params'), int) else "—",
                              sub="trainable params")

            # -- Config / data details --------------------------------------
            self._section(body, "CONFIGURATION")
            m = s.get("model", {})
            run_mode = s.get("run_mode", "")
            if run_mode == "batch":
                run_txt = f"batch  ·  {s.get('frames','')} frame(s)"
            elif run_mode == "temporal":
                run_txt = f"temporal video  ·  {s.get('temporal_frames_out','')} frames denoised"
            else:
                run_txt = "single frame"
            quant_txt = s.get("quant_scheme", "")
            if s.get("qat"):
                quant_txt += " (QAT, fake-quant in the loop)"
            details = [
                ("Sensor", f"{s.get('sensor','')}  ({s.get('sensor_key','')})  ·  {s.get('gain','')}× gain"),
                ("Capture", s.get("capture_mode", "")),
            ]
            if s.get("frame_source") and s.get("frame_source") != "synthetic":
                details.append(("Frame", s.get("frame_source", "")))
            if s.get("dataset_path"):
                details.append(("Dataset", s.get("dataset_path", "")))
            details += [
                ("Ground truth", s.get("gt_kind", "")),
                ("Run mode", run_txt),
                ("Quantization", quant_txt or "—"),
                ("Target", f"{s.get('hardware_name','')}  [{s.get('precision','')}]"),
            ]
            if m.get("custom_nafnet"):
                details.append(("NAFNet topology",
                                f"enc {m.get('nafnet_enc')} · mid {m.get('nafnet_middle')} "
                                f"· dec {m.get('nafnet_dec')}"))
            for k, v in details:
                r = tk.Frame(body, bg=WHITE); r.pack(fill="x", pady=S(2))
                tk.Label(r, text=k, bg=WHITE, fg=SUBTLE, font=font(10),
                         width=14, anchor="w").pack(side="left")
                tk.Label(r, text=v, bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")

            # -- Cross-chip suitability -------------------------------------
            if s.get("targets"):
                self._section(body, "RUNS ON THESE CHIPS")
                tk.Label(body, text="     Will this exact model deploy on each "
                         "Raspberry Pi-class target? (from each chip's specs)",
                         bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                         justify="left").pack(anchor="w", pady=(0, S(6)))
                for t in s["targets"]:
                    self._target_card(body, t)

            if s.get("warnings"):
                self._section(body, "COMPILER NOTES")
                for w in s["warnings"]:
                    tk.Label(body, text="▲  " + w, bg=WHITE, fg="#C98A1B",
                             font=font(9), wraplength=S(560), justify="left").pack(anchor="w")

            # -- Transferable deployment package ----------------------------
            self._section(body, "TRANSFERABLE PACKAGE")
            if s.get("package_zip"):
                tk.Label(body, text="✓  Hardware-ready package exported",
                         bg=WHITE, fg=GREEN, font=font(11, "bold")).pack(anchor="w")
                tk.Label(body, text=s["package_zip"], bg=WHITE, fg=INK,
                         font=font(9), wraplength=S(580), justify="left").pack(anchor="w")
                tk.Label(body, text="Contains the device binary, ONNX, manifest.json "
                         "and FLASH_INSTRUCTIONS.md — copy the .zip to the target and "
                         "follow the flash steps.", bg=WHITE, fg=SUBTLE,
                         font=font(9), wraplength=S(580), justify="left").pack(anchor="w",
                                                                              pady=(S(2), 0))
            else:
                tk.Label(body, text="Not exported yet — click EXPORT PACKAGE to bundle a "
                         "transferable .zip for the target device.", bg=WHITE, fg=SUBTLE,
                         font=font(9), wraplength=S(580), justify="left").pack(anchor="w")

        # -- Validation panel image -----------------------------------------
        self._section(body, "VALIDATION MATRIX")
        if ImageTk and PANEL_PATH.exists():
            try:
                im = Image.open(PANEL_PATH)
                target_w = S(640)
                ratio = target_w / im.width
                im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
                self._result_img = ImageTk.PhotoImage(im)
                tk.Label(body, image=self._result_img, bg=WHITE).pack(anchor="w", pady=(S(4), S(12)))
            except Exception:
                tk.Label(body, text="(could not render validation_panel.png)",
                         bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")
        else:
            tk.Label(body, text="validation_panel.png not found in outputs/",
                     bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")

        # -- Resolution vs TOPS scaling chart ---------------------------------
        scale_path = None
        if s and s.get("scaling_chart"):
            scale_path = Path(s["scaling_chart"])
        if scale_path is None:
            scale_path = ROOT / "outputs" / "resolution_tops_scaling.png"
        self._section(body, "RESOLUTION vs TOPS SCALING")
        tk.Label(body, text="How effective throughput scales with input pixels "
                 "for each Pi-class target (dashed = peak TOPS).",
                 bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(0, S(6)))
        if ImageTk and scale_path.exists():
            try:
                im = Image.open(scale_path)
                target_w = S(640)
                ratio = target_w / im.width
                im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
                self._scaling_img = ImageTk.PhotoImage(im)
                tk.Label(body, image=self._scaling_img, bg=WHITE).pack(
                    anchor="w", pady=(S(4), S(12)))
            except Exception:
                tk.Label(body, text="(could not render resolution_tops_scaling.png)",
                         bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")
        else:
            tk.Label(body, text="resolution_tops_scaling.png not found — "
                     "re-run a compile to generate it.",
                     bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")

    def _show_ranking(self):
        """Leaderboard of every model the sweep trained (from outputs/pareto.json)."""
        try:
            data = json.loads((ROOT / "outputs" / "pareto.json").read_text(encoding="utf-8"))
        except Exception:
            data = {}
        self._rank_rows = data.get("all_results", [])
        self._rank_winner = data.get("winner", {})
        self._rank_target_label = data.get("target_label", "")
        # Default the filter to the chip the sweep actually compiled for.
        target = data.get("target", "all")
        self._rank_have_chips = any(r.get("chips") for r in self._rank_rows)
        self._rank_filter = target if (self._rank_have_chips and target in CHIP_LABEL) else "all"
        rows = self._rank_rows
        winner = self._rank_winner
        pad = S(34)
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()

        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(22), S(2)))
        tk.Label(header, text="Sweep Complete", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(header, text=f"SWEEP   ·   {self._rank_target_label}  ·  "
                 f"{len(rows)} models trained  ·  "
                 f"✦ = Pareto-optimal  ·  click any row to run it", bg=WHITE,
                 fg=SUBTLE, font=font(10)).pack(anchor="w", pady=(S(2), 0))

        # -- Hardware preference filter --------------------------------------
        if self._rank_have_chips:
            fbar = tk.Frame(self.main, bg=WHITE)
            fbar.pack(fill="x", padx=pad, pady=(S(8), 0))
            tk.Label(fbar, text="Best for:", bg=WHITE, fg=INK,
                     font=font(10, "bold")).pack(side="left", padx=(0, S(8)))
            self._rank_filter_row = tk.Frame(fbar, bg=WHITE)
            self._rank_filter_row.pack(side="left")
            self._render_filter_buttons()

        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        # Footer (pinned).
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(12))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        foot_row = tk.Frame(footer, bg=WHITE)
        foot_row.pack(fill="x")
        RoundButton(foot_row, "BACK", self._back, kind="secondary",
                    width=100, height=40).pack(side="left")
        RoundButton(foot_row, "OPEN OUTPUTS", self._open_outputs, kind="secondary",
                    width=130, height=40).pack(side="left", padx=(S(6), 0))
        RoundButton(foot_row, "FULL LOG", self._show_log, kind="secondary",
                    width=110, height=40).pack(side="right")
        if winner:
            RoundButton(foot_row, "USE WINNER", lambda: self._use_winner(winner),
                        kind="primary", width=130, height=40).pack(side="right",
                                                                   padx=(0, S(6)))

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(6), 0))
        body = self._make_scrollable(outer)

        if not rows:
            tk.Label(body, text="No ranking found. See the full log for details.",
                     bg=WHITE, fg=SUBTLE, font=font(11)).pack(anchor="w", pady=S(10))
            return

        # Table container (re-rendered when the filter changes) + winner callout.
        self._rank_table_holder = tk.Frame(body, bg=WHITE)
        self._rank_table_holder.pack(fill="x")
        self._render_rank_table()

        if winner:
            self._section(body, "OVERALL WINNER (highest fitness)")
            tk.Label(body, text=f"{winner.get('family','').upper()}  "
                     f"{winner.get('base_channels')}ch × depth {winner.get('block_depth')}",
                     bg=WHITE, fg=GREEN, font=font(13, "bold")).pack(anchor="w")
            tk.Label(body, text=f"PSNR {winner.get('psnr')} dB   ·   "
                     f"{winner.get('latency_ms')} ms   ·   "
                     f"fitness {winner.get('fitness')} / 100 ({winner.get('grade')})",
                     bg=WHITE, fg=INK, font=font(10)).pack(anchor="w", pady=(S(2), 0))
            tk.Label(body, text="Use the 'Best for' filter above to re-rank by which "
                     "Raspberry Pi chip a model is suitable for. Click USE WINNER to "
                     "load this config, or click any row to run that exact model.",
                     bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(620),
                     justify="left").pack(anchor="w", pady=(S(4), S(10)))

    def _render_filter_buttons(self):
        for w in self._rank_filter_row.winfo_children():
            w.destroy()
        for key in ("all", "rpi5_cpu", "hailo8", "deepx"):
            sel = (key == self._rank_filter)
            RoundButton(self._rank_filter_row, CHIP_LABEL[key],
                        lambda k=key: self._set_rank_filter(k),
                        kind="primary" if sel else "secondary",
                        width=120, height=34).pack(side="left", padx=(0, S(6)))

    def _set_rank_filter(self, key: str):
        self._rank_filter = key
        self._render_filter_buttons()
        self._render_rank_table()

    def _rank_standouts(self, rows: list) -> dict:
        """Identify the standout model in each axis so each row can say *why*."""
        def _num(r, k):
            v = r.get(k)
            return v if isinstance(v, (int, float)) else None
        tags = {}
        psnr_rows = [r for r in rows if _num(r, "psnr") is not None]
        lat_rows = [r for r in rows if _num(r, "latency_ms") is not None]
        par_rows = [r for r in rows if _num(r, "params") is not None]
        if psnr_rows:
            tags["sharp"] = id(max(psnr_rows,
                                    key=lambda r: r.get("psnr_gain", r.get("psnr", 0))))
        if lat_rows:
            tags["fast"] = id(min(lat_rows, key=lambda r: r["latency_ms"]))
        if par_rows:
            tags["lean"] = id(min(par_rows, key=lambda r: r["params"]))
        if rows:
            tags["best"] = id(max(rows, key=lambda r: r.get("fitness", 0)))
        return tags

    def _why_tag(self, r: dict, standouts: dict):
        """Return (text, colour) describing what this model is best at."""
        rid = id(r)
        if standouts.get("best") == rid:
            return "top pick", GREEN
        if standouts.get("sharp") == rid:
            return "sharpest", "#3F6FB0"
        if standouts.get("fast") == rid:
            return "fastest", "#3F6FB0"
        if standouts.get("lean") == rid:
            return "leanest", AMBER
        return "", SUBTLE

    def _render_rank_table(self):
        holder = self._rank_table_holder
        for w in holder.winfo_children():
            w.destroy()
        rows = list(self._rank_rows)
        flt = self._rank_filter
        chip_mode = flt != "all" and self._rank_have_chips
        multi_sensor = len({r.get("sensor") for r in rows if r.get("sensor")}) > 1
        sensor_col = [("SENSOR", 6)] if multi_sensor else []
        have_gain = any(r.get("psnr_gain") is not None for r in rows)
        gain_col = [("GAIN", 7)] if have_gain else []
        mw = 11 if (multi_sensor or have_gain) else 14
        sw = 7 if (multi_sensor or have_gain) else 10

        if chip_mode:
            def _key(r):
                c = (r.get("chips") or {}).get(flt, {})
                vr = VERDICT_RANK.get(c.get("verdict"), 3)
                return (vr, -float(r.get("fitness", 0)))
            rows.sort(key=_key)
            cols = ([("#", 3), ("MODEL", mw)] + sensor_col +
                    [("PARAMS", 7)] + gain_col + [("PSNR", 7), ("FPS", 6), ("FIT", 6),
                     (f"ON {CHIP_LABEL[flt]}", 11), ("STANDOUT", sw)])
        else:
            rows.sort(key=lambda r: -float(r.get("fitness", 0)))
            cols = ([("#", 3), ("MODEL", mw)] + sensor_col +
                    [("PARAMS", 7)] + gain_col + [("PSNR", 7), ("LATENCY", 8), ("FIT", 6),
                     ("RATING", 7), ("STANDOUT", sw)])

        if chip_mode:
            tk.Label(holder, text=f"Ranked by suitability for "
                     f"{CHIP_LABEL[flt]} (best fit first), then fitness.",
                     bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w", pady=(S(6), 0))

        hrow = tk.Frame(holder, bg=WHITE); hrow.pack(fill="x", pady=(S(6), S(2)))
        for label, w in cols:
            tk.Label(hrow, text=label, bg=WHITE, fg=SUBTLE, font=font(8, "bold"),
                     width=w, anchor="w").pack(side="left")
        tk.Frame(holder, bg=LINE, height=1).pack(fill="x", pady=(0, S(2)))

        standouts = self._rank_standouts(self._rank_rows)
        for i, r in enumerate(rows, 1):
            best = (i == 1)
            bg = FIELD if best else WHITE
            row = tk.Frame(holder, bg=bg, cursor="hand2"); row.pack(fill="x", pady=1)
            star = "✦ " if r.get("pareto") else "  "
            model = (f"{r.get('family','').upper()} {r.get('base_channels')}ch×"
                     f"{r.get('block_depth')}")
            g = r.get("grade", "")
            why, why_col = self._why_tag(r, standouts)
            sensor_cell = ([(SENSOR_SHORT.get(r.get("sensor"), r.get("sensor", "—")),
                             7, SUBTLE, False)] if multi_sensor else [])
            gain = r.get("psnr_gain")
            if gain is None and r.get("psnr_in") is not None and r.get("psnr") is not None:
                gain = float(r["psnr"]) - float(r["psnr_in"])
            if gain is not None:
                gain_txt = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
                gain_col = (f"{gain_txt} dB", 7,
                            GREEN if gain >= 8 else AMBER if gain >= 3 else RASPBERRY,
                            gain >= 3)
            else:
                gain_col = None

            if chip_mode:
                c = (r.get("chips") or {}).get(flt, {})
                verdict = c.get("verdict", "")
                vlabel = VERDICT_LABEL.get(verdict, "—")
                vcol = VERDICT_COLORS.get(verdict, SUBTLE)
                fps = c.get("fps")
                fps_txt = f"{fps:.0f}" if isinstance(fps, (int, float)) else "—"
                cells = [
                    (f"{i}", 3, INK, False),
                    (star + model, mw, RASPBERRY if best else INK, False),
                ] + sensor_cell + [
                    (f"{r.get('params',0)/1000:.1f}K", 7, SUBTLE, False),
                ] + ([gain_col] if gain_col else []) + [
                    (f"{r.get('psnr','—')} dB", 7, INK, False),
                    (fps_txt, 6, INK, False),
                    (f"{r.get('fitness','—')}", 6, GRADE_COLORS.get(g, INK), False),
                    (vlabel, 11, vcol, True),
                    (why, sw, why_col, True),
                ]
            else:
                cells = [
                    (f"{i}", 3, INK, False),
                    (star + model, mw, RASPBERRY if best else INK, False),
                ] + sensor_cell + [
                    (f"{r.get('params',0)/1000:.1f}K", 7, SUBTLE, False),
                ] + ([gain_col] if gain_col else []) + [
                    (f"{r.get('psnr','—')} dB", 7, INK, False),
                    (f"{r.get('latency_ms','—')} ms", 8, SUBTLE, False),
                    (f"{r.get('fitness','—')}", 6, GRADE_COLORS.get(g, INK), False),
                    (g, 8, GRADE_COLORS.get(g, INK), True),
                    (why, sw, why_col, True),
                ]

            cell_widgets = [row]
            for text, w, fg, bold in cells:
                lbl = tk.Label(row, text=text, bg=bg, fg=fg,
                               font=font(9, "bold" if (best or bold) else "normal"),
                               width=w, anchor="w")
                lbl.pack(side="left")
                cell_widgets.append(lbl)
            hi = "#F0F2F8"
            for wdg in cell_widgets:
                wdg.bind("<Button-1>", lambda _e, rr=r: self._ranking_row_clicked(rr))
                wdg.bind("<Enter>", lambda _e, ws=cell_widgets: [x.configure(bg=hi) for x in ws])
                wdg.bind("<Leave>", lambda _e, ws=cell_widgets, b=bg: [x.configure(bg=b) for x in ws])

    def _use_winner(self, winner: dict):
        self._use_config(winner)

    def _use_config(self, r: dict):
        """Load a ranking row's parameters into the form (single-compile mode)."""
        self.sidebar.reset()
        self._build_form()                     # rebuilds self.rows with defaults
        self.eval_var.set("single")
        self._on_eval_change()
        sensor = r.get("sensor")
        if sensor and sensor != "all" and "sensor" in self.rows:
            self.rows["sensor"].set(sensor)
            self._on_sensor_change()
        fam = r.get("family")
        if fam and "model_family" in self.rows:
            self.rows["model_family"].set(fam)
        # .set() doesn't fire the combobox callback, so rebuild the detail rows
        # for this family before applying its specific parameters.
        self._render_model_options()
        for key in ("base_channels", "block_depth", "conv_type", "activation"):
            val = r.get(key)
            if val is not None and key in self.rows:
                self.rows[key].set(val)
        # Jump straight to the review step so the loaded config is visible.
        try:
            self._goto_step(len(self._steps) - 1)
        except Exception:
            pass

    def _ranking_row_clicked(self, r: dict):
        """Popup: show a clicked model's config + options to run it."""
        dlg = tk.Toplevel(self)
        dlg.title("Run this model")
        dlg.configure(bg=WHITE)
        dlg.transient(self)
        try:
            dlg.geometry(f"{S(500)}x{S(500)}+{self.winfo_rootx()+S(120)}"
                         f"+{self.winfo_rooty()+S(90)}")
        except Exception:
            pass
        self._grab_when_ready(dlg)

        pad = tk.Frame(dlg, bg=WHITE)
        pad.pack(fill="both", expand=True, padx=S(22), pady=S(18))
        tk.Label(pad, text="Run this configuration", bg=WHITE, fg=INK,
                 font=font(15, "bold")).pack(anchor="w")
        title = (f"{r.get('family','').upper()}  {r.get('base_channels')}ch × "
                 f"depth {r.get('block_depth')}")
        tk.Label(pad, text=title, bg=WHITE, fg=RASPBERRY,
                 font=font(13, "bold")).pack(anchor="w", pady=(S(8), 0))
        fam = r.get("family", "")
        if fam in ("nafnet", "restormer"):
            sub = ("SimpleGate + depthwise" if fam == "nafnet"
                   else "transposed attention + GELU FFN")
        else:
            sub = f"{r.get('conv_type','')} conv  ·  {r.get('activation','')} activation"
        tk.Label(pad, text=sub, bg=WHITE, fg=INK, font=font(10)).pack(anchor="w")

        g = r.get("grade", "")
        gcol = GRADE_COLORS.get(g, INK)
        metrics = []
        if r.get("sensor") and r.get("sensor") != "all":
            metrics.append(("Sensor", r.get("sensor"), RASPBERRY))
        metrics += [
            ("Sweep PSNR", f"{r.get('psnr','—')} dB", INK),
            ("Sweep latency", f"{r.get('latency_ms','—')} ms", INK),
            ("Parameters", f"{r.get('params',0):,}", INK),
            ("Sweep fitness", f"{r.get('fitness','—')} / 100  ({g})", gcol),
        ]
        for label, val, col in metrics:
            rr = tk.Frame(pad, bg=WHITE); rr.pack(fill="x", pady=S(2))
            tk.Label(rr, text=label, bg=WHITE, fg=SUBTLE, font=font(9),
                     width=14, anchor="w").pack(side="left")
            tk.Label(rr, text=val, bg=WHITE, fg=col,
                     font=font(10, "bold")).pack(side="left")

        # -- Per-chip suitability for this exact model -----------------------
        chips = r.get("chips") or {}
        if chips:
            tk.Label(pad, text="Runs on these chips", bg=WHITE, fg=INK,
                     font=font(10, "bold")).pack(anchor="w", pady=(S(8), S(2)))
            for key in ("rpi5_cpu", "hailo8", "deepx"):
                c = chips.get(key)
                if not c:
                    continue
                v = c.get("verdict", "")
                vcol = VERDICT_COLORS.get(v, SUBTLE)
                fps = c.get("fps")
                fps_txt = f"{fps:.0f} FPS" if isinstance(fps, (int, float)) else "—"
                cr = tk.Frame(pad, bg=WHITE); cr.pack(fill="x", pady=S(1))
                tk.Label(cr, text=CHIP_LABEL.get(key, key), bg=WHITE, fg=INK,
                         font=font(9, "bold"), width=11, anchor="w").pack(side="left")
                tk.Label(cr, text=VERDICT_LABEL.get(v, "—"), bg=WHITE, fg=vcol,
                         font=font(9, "bold"), width=14, anchor="w").pack(side="left")
                tk.Label(cr, text=fps_txt, bg=WHITE, fg=SUBTLE,
                         font=font(9), width=9, anchor="w").pack(side="left")

        tk.Label(pad, text="Sweep values come from a fast calibration. Run the full "
                 "pipeline to get final artifacts, the validation panel and the "
                 "per-chip suitability report.", bg=WHITE, fg=SUBTLE, font=font(8),
                 wraplength=S(410), justify="left").pack(anchor="w", pady=(S(8), S(12)))

        btns = tk.Frame(pad, bg=WHITE); btns.pack(fill="x", side="bottom")

        def _run_now():
            dlg.destroy()
            self._use_config(r)
            self._run()

        def _load_only():
            dlg.destroy()
            self._use_config(r)

        RoundButton(btns, "CANCEL", dlg.destroy, kind="secondary",
                    width=110, height=40).pack(side="left")
        RoundButton(btns, "LOAD INTO FORM", _load_only, kind="secondary",
                    width=170, height=40).pack(side="left", padx=(S(8), 0))
        RoundButton(btns, "RUN THIS", _run_now, kind="primary",
                    width=130, height=40).pack(side="right")

    def _view_text_file(self, path, title, missing_msg=None):
        """Open a read-only in-app viewer for a text/JSON file."""
        p = Path(path)
        if not p.exists():
            messagebox.showinfo(title, missing_msg or f"File not found:\n{p}")
            return
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(title, f"Could not read {p}:\n{exc}")
            return
        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg=WHITE)
        win.transient(self)
        try:
            win.geometry(f"{S(720)}x{S(520)}+{self.winfo_rootx()+S(40)}"
                         f"+{self.winfo_rooty()+S(40)}")
        except Exception:
            pass
        pad = tk.Frame(win, bg=WHITE)
        pad.pack(fill="both", expand=True, padx=S(16), pady=S(14))
        tk.Label(pad, text=title, bg=WHITE, fg=INK,
                 font=font(15, "bold")).pack(anchor="w")
        tk.Label(pad, text=str(p.resolve()), bg=WHITE, fg=SUBTLE,
                 font=font(9), wraplength=S(640), justify="left").pack(
                     anchor="w", pady=(S(2), S(8)))
        con = tk.Frame(pad, bg=FIELD)
        con.pack(fill="both", expand=True)
        mono = ("Cascadia Mono" if "Cascadia Mono" in tkfont.families()
                else ("DejaVu Sans Mono" if "DejaVu Sans Mono" in tkfont.families()
                      else "Courier"))
        txt = tk.Text(con, bg=FIELD, fg=INK, bd=0, relief="flat",
                      font=(mono, FT(9)), wrap="none", padx=S(12), pady=S(10))
        sb = ttk.Scrollbar(con, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        foot = tk.Frame(pad, bg=WHITE)
        foot.pack(fill="x", pady=(S(10), 0))
        RoundButton(foot, "CLOSE", win.destroy, kind="secondary",
                    width=110, height=38).pack(side="right")

    def _show_log(self):
        win = tk.Toplevel(self)
        win.title("NAS — Full compilation log")
        win.configure(bg=WHITE)
        win.geometry(f"{S(820)}x{S(560)}")
        con = tk.Frame(win, bg=FIELD)
        con.pack(fill="both", expand=True, padx=S(12), pady=S(12))
        mono = "Cascadia Mono" if "Cascadia Mono" in tkfont.families() else \
               ("DejaVu Sans Mono" if "DejaVu Sans Mono" in tkfont.families() else "Courier")
        txt = tk.Text(con, bg=FIELD, fg=INK, bd=0, relief="flat",
                      font=(mono, FT(9)), wrap="none", padx=S(12), pady=S(10))
        sb = ttk.Scrollbar(con, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", getattr(self, "_full_log", "") or "(no log captured)")
        txt.configure(state="disabled")

    def _back(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.sidebar.reset()
        self._build_form()

    def _open_outputs(self):
        path = ROOT / "outputs"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if self._try_open_os_path(path, "Outputs folder"):
            return
        try:
            messagebox.showinfo("Outputs folder", f"Results are saved in:\n{path.resolve()}")
        except Exception:
            pass

    def _open_path(self, target):
        """Open a file or folder in the OS, or show its path if that fails."""
        p = Path(target)
        if not self._try_open_os_path(p, "Open"):
            try:
                messagebox.showinfo("Open", str(p.resolve()))
            except Exception:
                pass

    def _try_open_os_path(self, path, fallback_title="Path") -> bool:
        """Try the OS file manager / default app. False if unavailable (e.g. SSH)."""
        path = Path(path)
        target = path if path.exists() else path.parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
                return True
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
                return True
            # Linux: xdg-open needs X11; over SSH it often fails with
            # "connection rejected due to wrong authentication".
            if not _has_display():
                return False
            subprocess.Popen(
                ["xdg-open", str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    # -- Run history ---------------------------------------------------------
    def _show_history(self):
        """Browse past compiles + sweeps saved under outputs/history/."""
        try:
            from nsa.history import load_history
            rows = load_history(ROOT / "outputs" / "history")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("History", f"Could not load history:\n{exc}")
            return

        pad = S(34)
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()
        try:
            self.sidebar.all_done()
        except Exception:
            pass

        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(22), S(2)))
        tk.Label(header, text="Run history", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(header, text=f"{len(rows)} saved run(s) · refer back to past results "
                 "and models without re-running anything", bg=WHITE, fg=SUBTLE,
                 font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(14))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        RoundButton(footer, "◀ BACK", self._back, kind="secondary",
                    width=130, height=44).pack(side="left")
        RoundButton(footer, "OPEN HISTORY FOLDER",
                    lambda: self._open_path(ROOT / "outputs" / "history"),
                    kind="secondary", width=210, height=44).pack(side="left",
                                                                 padx=(S(8), 0))

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(8), 0))
        body = self._make_scrollable(outer)

        if not rows:
            tk.Label(body, text="No runs saved yet. Run a compile or sweep and it will "
                     "be archived here automatically (model + report + artifacts).",
                     bg=WHITE, fg=SUBTLE, font=font(11), wraplength=S(560),
                     justify="left").pack(anchor="w", pady=S(20))
            return
        for rec in rows:
            self._history_card(body, rec)

    def _history_card(self, parent, rec):
        is_sweep = rec.get("kind") == "sweep"
        card = tk.Frame(parent, bg=FIELD)
        card.pack(fill="x", pady=S(5))
        inner = tk.Frame(card, bg=FIELD)
        inner.pack(fill="x", padx=S(14), pady=S(10))

        top = tk.Frame(inner, bg=FIELD); top.pack(fill="x")
        tk.Label(top, text=rec.get("time", ""), bg=FIELD, fg=INK,
                 font=font(11, "bold")).pack(side="left")
        kind_txt = "SWEEP" if is_sweep else "COMPILE"
        kind_col = AMBER if is_sweep else RASPBERRY
        tk.Label(top, text=f"  {kind_txt} ", bg=FIELD, fg=kind_col,
                 font=font(8, "bold")).pack(side="left", padx=(S(6), 0))
        g = rec.get("grade")
        if g:
            tk.Label(top, text=f" {g} ", bg=FIELD, fg=GRADE_COLORS.get(g, INK),
                     font=font(8, "bold")).pack(side="right")

        prof = rec.get("profile", "")
        sub = f"{prof}   ·   {rec.get('hardware_name', rec.get('hardware',''))}"
        sensor = rec.get("sensor")
        if sensor:
            sub += f"   ·   {sensor}"
        if rec.get("gain"):
            sub += f" @{rec.get('gain')}x"
        tk.Label(inner, text=sub, bg=FIELD, fg=RASPBERRY, font=font(10, "bold"),
                 wraplength=S(620), justify="left").pack(anchor="w", pady=(S(3), 0))

        # Metrics line
        bits = []
        if rec.get("psnr_out") is not None:
            if rec.get("psnr_in") is not None:
                bits.append(f"PSNR {rec['psnr_in']:.1f} -> {rec['psnr_out']:.1f} dB")
            else:
                bits.append(f"PSNR {rec['psnr_out']:.1f} dB")
        if rec.get("fps"):
            bits.append(f"{rec['fps']:.0f} FPS")
        if rec.get("latency_ms"):
            bits.append(f"{rec['latency_ms']:.1f} ms")
        if rec.get("fitness") is not None:
            bits.append(f"fit {rec['fitness']}")
        if rec.get("params"):
            bits.append(f"{rec['params']/1000:.1f}K params")
        if is_sweep and rec.get("n_evaluated"):
            bits.append(f"{rec['n_evaluated']} models tried")
        if bits:
            tk.Label(inner, text="   ·   ".join(bits), bg=FIELD, fg=INK,
                     font=font(9)).pack(anchor="w", pady=(S(3), 0))

        # Action buttons
        btns = tk.Frame(inner, bg=FIELD); btns.pack(anchor="w", pady=(S(8), 0))
        RoundButton(btns, "OPEN FOLDER", lambda r=rec: self._open_path(r.get("dir")),
                    kind="secondary", width=140, height=34).pack(side="left")
        if rec.get("panel"):
            RoundButton(btns, "VIEW PANEL",
                        lambda r=rec: self._open_path(r.get("panel")),
                        kind="secondary", width=130,
                        height=34).pack(side="left", padx=(S(6), 0))
        if not is_sweep and rec.get("model_pt"):
            RoundButton(btns, "USE FOR LIVE",
                        lambda r=rec: self._history_use_model(r),
                        kind="primary", width=140,
                        height=34).pack(side="left", padx=(S(6), 0))
        RoundButton(btns, "LOAD CONFIG",
                    lambda r=rec: self._history_load_config(r),
                    kind="secondary", width=140,
                    height=34).pack(side="left", padx=(S(6), 0))

    def _history_use_model(self, rec):
        """Copy a past model back to outputs/model.pt so live testing uses it."""
        import shutil
        src = rec.get("model_pt")
        if not src or not Path(src).exists():
            messagebox.showinfo("Use model", "This run has no saved model checkpoint.")
            return
        try:
            dest = ROOT / "outputs" / "model.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Use model", str(exc))
            return
        if messagebox.askyesno(
                "Model ready for live testing",
                f"Loaded the {rec.get('family','').upper()} model from "
                f"{rec.get('time','')} as the active model.\n\n"
                "Open live camera testing now?"):
            self._live_test()

    def _history_load_config(self, rec):
        """Reload a past run's configuration into the wizard (no re-run needed)."""
        m = rec.get("model", {}) or {}
        win = rec.get("winner", {}) or {}
        cfg = {
            "family": m.get("family") or win.get("family"),
            "base_channels": m.get("base_channels") or win.get("base_channels"),
            "block_depth": m.get("block_depth") or win.get("block_depth"),
            "conv_type": m.get("conv_type") or win.get("conv_type"),
            "activation": m.get("activation") or win.get("activation"),
            "sensor": rec.get("sensor_key") or rec.get("sensor"),
        }
        self._use_config(cfg)


def _has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if __name__ == "__main__":
    if not _has_display():
        print("No graphical display detected (headless session).\n"
              "The desktop UI needs a display (X / Wayland). Options:\n"
              "  • Run the CLI instead:   python run_demo.py --no-window\n"
              "  • Or enable X over SSH:  ssh -X <host>  (and install python3-tk)\n")
        sys.exit(1)
    try:
        App().mainloop()
    except tk.TclError as exc:
        print(f"Could not open the GUI: {exc}\n"
              "Tk may be missing — install it with: sudo apt install python3-tk\n"
              "Or run headless:  python run_demo.py --no-window")
        sys.exit(1)
