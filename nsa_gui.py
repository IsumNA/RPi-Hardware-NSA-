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


def _choice_int(val, default: int) -> int:
    """Coerce a ConfigRow / history value to int (comboboxes store strings)."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


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


# Chrome allowances (title bar + desktop panel/taskbar) when clamping to screen.
_SCREEN_MARGIN_W = 40
_SCREEN_MARGIN_H = 96


def fit_scale_to_screen(win, *, ref_w: int = 1180, ref_h: int = 820) -> None:
    """Shrink the global UI SCALE so the largest window still fits the display.

    The default scale is tuned for readability, not for small laptop panels
    (common on Linux). If the biggest window (the CTT wizard, ~ref_w×ref_h
    logical px) wouldn't fit at the current scale, drop the scale just enough
    that it does — so windows open on-screen and stay resizable. Never scales
    UP, and never below 0.8×."""
    global SCALE
    try:
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    except Exception:  # noqa: BLE001
        return
    fit = min((sw - _SCREEN_MARGIN_W) / ref_w, (sh - _SCREEN_MARGIN_H) / ref_h)
    if fit < SCALE:
        SCALE = max(0.8, fit)


def place_window(win, w_logical: float, h_logical: float, *, master=None,
                 min_w: float | None = None, min_h: float | None = None,
                 resizable: bool = True) -> None:
    """Size, position and constrain a window so it ALWAYS fits the screen.

    Scales the requested logical size by S(), clamps it (and the minsize) to the
    visible screen area, keeps the whole window on-screen, and makes it
    resizable. Using this everywhere means no window can open bigger than the
    display or with a minsize the screen can't satisfy — the root cause of the
    'can't resize / window off-screen' behaviour on Linux."""
    try:
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    except Exception:  # noqa: BLE001
        sw, sh = 1920, 1080
    avail_w, avail_h = sw - S(_SCREEN_MARGIN_W), sh - S(_SCREEN_MARGIN_H)
    w = max(240, min(S(w_logical), avail_w))
    h = max(180, min(S(h_logical), avail_h))
    if master is not None:
        try:
            x = master.winfo_rootx() + S(24)
            y = master.winfo_rooty() + S(20)
        except Exception:  # noqa: BLE001
            x, y = (sw - w) // 2, (sh - h) // 3
    else:
        x, y = (sw - w) // 2, (sh - h) // 3
    x = max(0, min(x, sw - w))          # keep fully on-screen
    y = max(0, min(y, sh - h))
    win.geometry(f"{w}x{h}+{x}+{y}")
    mw = min(S(min_w) if min_w else w, avail_w)
    mh = min(S(min_h) if min_h else h, avail_h)
    win.minsize(max(240, mw), max(180, mh))
    try:
        win.resizable(bool(resizable), bool(resizable))
    except Exception:  # noqa: BLE001
        pass


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "rpi_logo.png"


def _load_scaled_photo(path, target_px: int):
    """Load an image scaled to fit ``target_px``, or ``None`` if unavailable.

    Works with or without Pillow: Pillow gives high-quality aspect-preserving
    resizing; the Tk fallback (used when Pillow is missing) can only read PNG/GIF
    and shrinks by an integer ``subsample`` factor so large assets don't render
    at full native resolution.
    """
    path = Path(path)
    if not path.exists():
        return None
    if Image is not None and ImageTk is not None:
        try:
            im = Image.open(path).convert("RGBA")
            im.thumbnail((target_px, target_px), Image.LANCZOS)
            return ImageTk.PhotoImage(im)
        except Exception:  # noqa: BLE001
            pass
    try:
        photo = tk.PhotoImage(file=str(path))
        native = max(photo.width(), photo.height())
        # Ceil division: pick the smallest integer factor that keeps the image
        # within target_px (subsample can only shrink by whole numbers).
        factor = max(1, -(-native // max(1, target_px)))
        if factor > 1:
            photo = photo.subsample(factor, factor)
        return photo
    except Exception:  # noqa: BLE001
        return None


def _load_logo_photo(size: int | None = None):
    """Load ``assets/rpi_logo.png`` as a Tk photo, or ``None`` if unavailable."""
    return _load_scaled_photo(LOGO_PATH, size if size is not None else S(58))


def _draw_logo_fallback(canvas, cx: int, cy: int, radius: int) -> None:
    """Simple raspberry disc when the PNG logo cannot be loaded."""
    canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                       fill=RASPBERRY, outline="")
    canvas.create_text(cx, cy, text="π", fill="white", font=font(12, "bold"))
PANEL_PATH = ROOT / "outputs" / "validation_panel.png"
# Remembers the last-used wizard settings across launches (loaded on top of
# config.yaml defaults; keeps config.yaml's documentation comments intact).
GUI_STATE_PATH = ROOT / ".nsa_gui_state.json"

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

        logo_x, logo_y = S(34), S(46)
        if self._logo_img is None:
            self._logo_img = _load_logo_photo(S(58))
        if self._logo_img is not None:
            self.create_image(logo_x, logo_y, image=self._logo_img)
            tx = S(70)
        else:
            _draw_logo_fallback(self, logo_x, logo_y, S(20))
            tx = S(70)
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
        return _load_scaled_photo(path, S(self.THUMB))

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
        self._noise_on = tk.BooleanVar(value=False)
        self._noise_sigma_var = tk.StringVar(value="20")
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
        place_window(self, 840, 700, master=master, min_w=560, min_h=460)
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
        self._logo_img = _load_logo_photo(S(40))
        if self._logo_img is not None:
            tk.Label(header, image=self._logo_img, bg=WHITE).pack(
                side="left", padx=(0, S(12)))
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

        # -- Inject synthetic noise into the original (stress-test denoiser) --
        noiserow = tk.Frame(self, bg=WHITE)
        noiserow.pack(fill="x", padx=pad, pady=(S(6), 0))
        tk.Checkbutton(
            noiserow, text="Add noise to original", variable=self._noise_on,
            bg=WHITE, fg=INK, activebackground=WHITE, selectcolor=WHITE,
            font=font(10, "bold")).pack(side="left")
        tk.Label(noiserow, text="sigma (8-bit)", bg=WHITE, fg=INK,
                 font=font(10)).pack(side="left", padx=(S(12), S(6)))
        ttk.Spinbox(
            noiserow, from_=0, to=100, increment=5, width=5,
            textvariable=self._noise_sigma_var, style="Rpi.TCombobox").pack(side="left")
        tk.Label(noiserow,
                 text="Injects Gaussian noise on the raw feed before denoising "
                      "— watch the model clean a noisier input.",
                 bg=WHITE, fg=SUBTLE, font=font(8),
                 wraplength=S(420), justify="left").pack(side="left", padx=(S(10), 0))

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

                    if self._noise_on.get():
                        try:
                            sigma = float(self._noise_sigma_var.get())
                        except (ValueError, tk.TclError):
                            sigma = 0.0
                        if sigma > 0:
                            raw = _live.add_noise_bgr(raw, sigma)

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
        master = self.master
        if getattr(master, "_live_view", None) is self:
            master._live_view = None
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            pass


class Imx662DataStudio(tk.Toplevel):
    """Browse IMX662 dataset layout: what's needed, what's on disk, GT capture help."""

    _SECTION_LABELS = {
        "on_disk": "ON DISK — manager PI_RAW captures (do not delete)",
        "noise_pipeline": "YOU ADD — noise calibration shoots (bias / dark / flat)",
        "imx662_targets": "TO GENERATE — IMX662 night-vision pairs (synthesis)",
    }
    _STATUS_ICON = {
        "complete": "✓",
        "partial": "◐",
        "missing": "○",
    }
    _STATUS_FG = {
        "complete": GREEN,
        "partial": AMBER,
        "missing": SUBTLE,
    }

    def __init__(self, master):
        super().__init__(master, bg=WHITE)
        self.app = master
        self.title("NAS  ·  IMX662 Dataset Studio")
        self.configure(bg=WHITE)
        self._audit: dict = {}
        self._thumb_refs: list = []

        from nsa.dataset_layout import find_best_project_root
        default_root = find_best_project_root()
        if default_root is None:
            default_root = ROOT / "datasets" / "imx662_project"
        self.root_var = tk.StringVar(value=str(default_root))
        self.gain_var = tk.StringVar(value="256")
        self.ag_tag_var = tk.StringVar(value="ag12")

        self._build_chrome()
        self._refresh()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.transient(master)
        place_window(self, 980, 700, master=master, min_w=640, min_h=520)
        master._grab_when_ready(self)

    def _build_chrome(self):
        pad = S(20)
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(14), S(4)))
        tk.Label(header, text="IMX662 Dataset Studio", bg=WHITE, fg=INK,
                 font=font(17, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text=("Point at your PI_RAW folder (e.g. /opt/datasets/PI_RAW). "
                  "Top section shows what your manager already captured "
                  "(cabinet_*, colour_stripes, imx219_ag*). Lower sections show "
                  "what you still need for IMX662 noise synthesis."),
            bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(900), justify="left",
        ).pack(anchor="w", pady=(S(2), 0))

        path_row = tk.Frame(self, bg=WHITE)
        path_row.pack(fill="x", padx=pad, pady=(S(8), 0))
        path_row.columnconfigure(1, weight=1)
        tk.Label(path_row, text="PI_RAW root", bg=WHITE, fg=INK,
                 font=font(10, "bold"), width=12, anchor="w").grid(row=0, column=0)
        ttk.Entry(path_row, textvariable=self.root_var, font=font(10)).grid(
            row=0, column=1, sticky="ew", padx=(S(4), S(4)))
        RoundButton(path_row, "…", self._browse_root, kind="secondary",
                    width=40, height=30).grid(row=0, column=2)
        RoundButton(path_row, "REFRESH", self._refresh, kind="secondary",
                    width=90, height=30).grid(row=0, column=3, padx=(S(6), 0))

        self.summary_lbl = tk.Label(self, text="", bg=WHITE, fg=INK, font=font(10),
                                    justify="left", wraplength=S(900))
        self.summary_lbl.pack(anchor="w", padx=pad, pady=(S(6), 0))

        prog_fr = tk.Frame(self, bg=FIELD)
        prog_fr.pack(fill="x", padx=pad, pady=(S(6), 0))
        self.prog = ttk.Progressbar(prog_fr, mode="determinate", length=S(400))
        self.prog.pack(side="left", padx=S(10), pady=S(8))
        self.prog_lbl = tk.Label(prog_fr, text="", bg=FIELD, fg=SUBTLE, font=font(9))
        self.prog_lbl.pack(side="left", padx=(S(8), S(10)))

        body = tk.Frame(self, bg=WHITE)
        body.pack(fill="both", expand=True, padx=pad, pady=(S(8), 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=WHITE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, S(8)))
        tk.Label(left, text="Checklist", bg=WHITE, fg=INK,
                 font=font(10, "bold")).pack(anchor="w")
        tree_fr = tk.Frame(left, bg=WHITE)
        tree_fr.pack(fill="both", expand=True, pady=(S(4), 0))
        self.tree = ttk.Treeview(tree_fr, columns=("status", "count"), show="tree headings",
                                 height=16)
        self.tree.heading("#0", text="Scene / test folder")
        self.tree.heading("status", text="")
        self.tree.heading("count", text="Files")
        self.tree.column("#0", width=S(220), stretch=True)
        self.tree.column("status", width=S(28), stretch=False)
        self.tree.column("count", width=S(72), stretch=False)
        sy = ttk.Scrollbar(tree_fr, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sy.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sy.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = tk.Frame(body, bg=FIELD, highlightthickness=1,
                         highlightbackground=LINE)
        right.grid(row=0, column=1, sticky="nsew")
        self.detail_title = tk.Label(right, text="Select a slot", bg=FIELD, fg=INK,
                                     font=font(12, "bold"), anchor="w")
        self.detail_title.pack(fill="x", padx=S(12), pady=(S(10), S(2)))
        self.detail_path = tk.Label(right, text="", bg=FIELD, fg=SUBTLE,
                                    font=font(9), anchor="w")
        self.detail_path.pack(fill="x", padx=S(12))
        self.detail_purpose = tk.Label(right, text="", bg=FIELD, fg=INK,
                                       font=font(9), wraplength=S(480), justify="left",
                                       anchor="w")
        self.detail_purpose.pack(fill="x", padx=S(12), pady=(S(6), 0))
        tk.Label(right, text="How to capture", bg=FIELD, fg=INK,
                 font=font(9, "bold"), anchor="w").pack(fill="x", padx=S(12),
                                                        pady=(S(8), 0))
        cap_fr = tk.Frame(right, bg=WHITE)
        cap_fr.pack(fill="both", expand=True, padx=S(12), pady=(S(4), S(8)))
        self.capture_txt = tk.Text(cap_fr, height=10, wrap="word", font=font(9),
                                   bg=WHITE, fg=INK, relief="flat",
                                   highlightthickness=1, highlightbackground=LINE)
        self.capture_txt.pack(fill="both", expand=True)
        self.capture_txt.config(state="disabled")

        thumb_lbl_fr = tk.Frame(right, bg=FIELD)
        thumb_lbl_fr.pack(fill="x", padx=S(12))
        tk.Label(thumb_lbl_fr, text="On disk", bg=FIELD, fg=INK,
                 font=font(9, "bold")).pack(side="left")
        self.thumb_count = tk.Label(thumb_lbl_fr, text="", bg=FIELD, fg=SUBTLE,
                                    font=font(9))
        self.thumb_count.pack(side="left", padx=(S(6), 0))
        self.thumb_row = tk.Frame(right, bg=FIELD)
        self.thumb_row.pack(fill="x", padx=S(12), pady=(S(4), S(10)))

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(6), 0))
        foot = tk.Frame(self, bg=WHITE)
        foot.pack(fill="x", padx=pad, pady=S(12))
        RoundButton(foot, "USE EXISTING GT", self._export_gt_from_pi_raw,
                    kind="primary", width=160, height=38).pack(side="left")
        RoundButton(foot, "ADD CALIB FOLDERS", self._scaffold,
                    kind="secondary", width=150, height=38).pack(side="left",
                                                                   padx=(S(8), 0))
        RoundButton(foot, "GT FROM BURST…", self._build_gt,
                    kind="secondary", width=130, height=38).pack(side="left",
                                                                 padx=(S(4), 0))
        RoundButton(foot, "OPEN FOLDER", self._open_slot,
                    kind="secondary", width=130, height=38).pack(side="left",
                                                                   padx=(S(8), 0))
        RoundButton(foot, "NOISE WIZARD", self._open_wizard,
                    kind="secondary", width=130, height=38).pack(side="right")

    def _browse_root(self):
        p = filedialog.askdirectory(
            title="PI_RAW dataset root (folder containing Data/)",
            initialdir=self.root_var.get() or str(ROOT / "datasets"),
        )
        if p:
            self.root_var.set(p)
            self._refresh()

    def _refresh(self):
        from nsa.dataset_layout import IMX662_TARGET_AG_TAGS, audit_project
        root = Path(self.root_var.get().strip() or ".")
        try:
            gain = int(self.gain_var.get())
        except ValueError:
            gain = 256
        self._audit = audit_project(
            root, gain=gain, imx662_ag_tags=IMX662_TARGET_AG_TAGS,
        )

        sm = self._audit.get("summary", {})
        inv = self._audit.get("pi_raw_inventory", {})
        cal = self._audit.get("calibration_pipeline") or {}
        cal_ok = "READY" if cal.get("ready") else "needed"
        self.prog["value"] = min(
            100.0,
            100.0 * sm.get("imx662_pairs_on_disk", 0)
            / max(1, len(IMX662_TARGET_AG_TAGS) * max(1, sm.get("scenes_on_disk", 1))),
        )
        self.prog_lbl.config(
            text=(f"{sm.get('paired_on_disk', 0)} paired folders on disk  ·  "
                  f"IMX662 targets missing: {sm.get('imx662_targets_missing', 0)}"))
        exists = "found" if self._audit.get("exists") else "not found"
        ag_on_disk = ", ".join(inv.get("ag_tags", [])[:8]) or "—"
        self.summary_lbl.config(
            text=(
                f"PI_RAW {exists}: {self._audit.get('pi_raw_root', root)}  ·  "
                f"{sm.get('scenes_on_disk', 0)} scenes  ·  "
                f"tags on disk: {ag_on_disk}  ·  "
                f"noise calibration: {cal_ok}"
            ),
            fg=GREEN if cal.get("ready") else AMBER,
        )

        for item in self.tree.get_children():
            self.tree.delete(item)
        self._tree_items: dict[str, dict] = {}
        by_section = self._audit.get("by_section", {})
        for section, slots in by_section.items():
            sec_id = self.tree.insert(
                "", "end", text=self._SECTION_LABELS.get(section, section),
                values=("", ""),
            )
            for sl in slots:
                icon = self._STATUS_ICON.get(sl["status"], "?")
                count = f"{sl.get('found', 0)}"
                if sl.get("required"):
                    count = f"{sl.get('found', 0)}/{sl['required']}"
                iid = self.tree.insert(
                    sec_id, "end", text=sl["title"],
                    values=(icon, count),
                )
                self._tree_items[iid] = sl
                for child in sl.get("children") or []:
                    files = child.get("files") or {}
                    nfiles = len(files)
                    cicon = "✓" if child.get("has_pair") else "○"
                    label = child.get("folder_name", "?")
                    cid = self.tree.insert(
                        iid, "end", text=label,
                        values=(cicon, str(nfiles)),
                    )
                    self._tree_items[cid] = {
                        **child,
                        "title": label,
                        "rel_path": child.get("rel_path", ""),
                        "section": "on_disk",
                        "status": "complete" if child.get("has_pair") else "missing",
                        "purpose": (
                            f"{child.get('sensor', '?')} @ {child.get('ag_tag', '?')} — "
                            f"manager capture. Files: "
                            + ", ".join(sorted(files.keys())) or "none"
                        ),
                        "how_to_capture": (
                            "Already on disk. Each test folder should contain "
                            "noisy.dng, noisy.png, gt.dng, gt.png (not all required, "
                            "but noisy + gt pair must exist)."
                        ),
                        "files": list(files.values()),
                    }
            missing = sum(
                1 for s in slots
                if s.get("status") != "complete" and not s.get("optional")
            )
            self.tree.item(sec_id, open=(section != "on_disk" and missing > 0)
                          or section == "on_disk")

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        sl = self._tree_items.get(iid)
        if sl is None:
            return
        self.detail_title.config(text=sl["title"])
        self.detail_path.config(text=sl["rel_path"])
        self.detail_purpose.config(text=sl["purpose"])
        self.capture_txt.config(state="normal")
        self.capture_txt.delete("1.0", "end")
        self.capture_txt.insert("1.0", sl["how_to_capture"])
        self.capture_txt.config(state="disabled")
        st = sl["status"]
        self.detail_title.config(fg=self._STATUS_FG.get(st, INK))

        for w in self.thumb_row.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        pi_raw = Path(self._audit.get("pi_raw_root", self.root_var.get()))
        proj = Path(self._audit.get("project_root", pi_raw.parent))
        files = sl.get("files") or []
        self.thumb_count.config(text=f"({len(files)} file(s))")
        for rel in files[:6]:
            fp = pi_raw / rel
            if not fp.is_file():
                fp = proj / rel
            if not fp.is_file():
                continue
            cell = tk.Frame(self.thumb_row, bg=FIELD)
            cell.pack(side="left", padx=(0, S(6)))
            photo = _load_scaled_photo(fp, S(72))
            if photo is not None:
                self._thumb_refs.append(photo)
                tk.Label(cell, image=photo, bg=FIELD).pack()
            else:
                tk.Label(cell, text=fp.suffix, bg=WHITE, fg=SUBTLE,
                         font=font(8), width=8, height=4).pack()
            tk.Label(cell, text=fp.name[:14], bg=FIELD, fg=SUBTLE,
                     font=font(7)).pack()

    def _selected_slot_path(self) -> Path | None:
        sel = self.tree.selection()
        if not sel:
            return None
        sl = self._tree_items.get(sel[0])
        if not sl:
            return None
        from nsa.dataset_layout import resolve_layout
        proj, pi = resolve_layout(self._audit.get("project_root", self.root_var.get()))
        if sl.get("section") == "on_disk" or sl.get("section") == "imx662_targets":
            return pi / sl["rel_path"]
        return proj / sl["rel_path"]

    def _open_slot(self):
        p = self._selected_slot_path()
        if p is None:
            messagebox.showinfo("Open folder", "Select a slot in the checklist first.")
            return
        p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])  # noqa: S603
            else:
                subprocess.Popen(["xdg-open", str(p)])  # noqa: S603
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder", str(exc))

    def _scaffold(self):
        from nsa.dataset_layout import scaffold_imx662_project
        root = self.root_var.get().strip()
        if not root:
            messagebox.showerror("Template", "Choose a project root path first.")
            return
        try:
            gain = int(self.gain_var.get())
        except ValueError:
            gain = 256
        try:
            scaffold_imx662_project(root, gain=gain)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Template", str(exc))
            return
        messagebox.showinfo(
            "Template created",
            f"Folder tree and CAPTURE.md guides written under:\n{root}\n\n"
            "Shoot your frames into the labelled folders, then click REFRESH.")
        self._refresh()

    def _build_gt(self):
        from nsa.gt_capture import burst_folder_to_gt
        root = Path(self.root_var.get().strip())
        burst = filedialog.askdirectory(
            title="Burst folder (sequential RAW frames)",
            initialdir=str(root / "bursts") if (root / "bursts").is_dir() else str(root),
        )
        if not burst:
            return
        out = filedialog.asksaveasfilename(
            title="Save ground-truth image",
            defaultextension=".png",
            initialdir=str(root / "clean_scenes"),
            initialfile="gt_01.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if not out:
            return
        try:
            manifest = burst_folder_to_gt(burst, out, min_frames=8)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ground truth", str(exc))
            return
        messagebox.showinfo(
            "Ground truth saved",
            f"Averaged {manifest['frames_used']} frames →\n{manifest['output']}")
        self._refresh()

    def _export_gt_from_pi_raw(self):
        from nsa.dataset_layout import export_clean_gt_from_pi_raw, resolve_layout
        proj, pi = resolve_layout(self.root_var.get().strip())
        clean = proj / "clean_scenes"
        try:
            written = export_clean_gt_from_pi_raw(pi, clean)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Use existing GT", str(exc))
            return
        if not written:
            messagebox.showwarning(
                "Use existing GT",
                "No gt.* files found in PI_RAW scenes.\n"
                "Check that PI_RAW/Data/<scene>/imx219_ag12_test/ exists.")
            return
        messagebox.showinfo(
            "Ground truth copied",
            f"Copied {len(written)} gt file(s) to:\n{clean}\n\n"
            "Use clean_scenes/ as input in the Noise Dataset Wizard.")
        self._refresh()

    def _open_wizard(self):
        try:
            NoiseDatasetWizard(self.app)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Noise wizard", str(exc))


class NoiseDatasetWizard(tk.Toplevel):
    """5-phase IMX662 noise workflow: calibrate → synthesize PI_RAW dataset."""

    def __init__(self, master):
        super().__init__(master, bg=WHITE)
        self.app = master
        self.title("NAS  ·  Noise dataset builder")
        self.configure(bg=WHITE)
        self._step = 0
        self._busy = False

        self.calib_dir = tk.StringVar()
        self.model_out = tk.StringVar(value=str(ROOT / "models" / "noise" / "imx662_gain256.json"))
        self.sensor_var = tk.StringVar(value="imx662")
        self.gain_var = tk.StringVar(value="256")
        self.temp_var = tk.StringVar()
        from nsa.dataset_layout import find_best_project_root, resolve_layout
        _pi = find_best_project_root() or (ROOT / "datasets" / "PI_RAW")
        _proj, _pi_raw = resolve_layout(_pi)
        self.clean_dir = tk.StringVar(
            value=str(_proj / "clean_scenes") if (_proj / "clean_scenes").is_dir() else "")
        self.calib_dir.set(
            str(_proj / "calibration" / "imx662_gain256")
            if (_proj / "calibration" / "imx662_gain256").is_dir() else "")
        self.dataset_out = tk.StringVar(value=str(_pi_raw))
        self.calib_json = tk.StringVar()
        self.ag_tag_var = tk.StringVar(value="ag12")
        self.layout_var = tk.StringVar(value="auto")
        self.temporal_var = tk.StringVar(value="64")
        self.write_config_var = tk.BooleanVar(value=True)
        self.use_legacy_var = tk.BooleanVar(value=False)

        self._build_chrome()
        self._show_step(0)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.transient(master)
        place_window(self, 720, 640, master=master, min_w=560, min_h=480)
        master._grab_when_ready(self)

    def _build_chrome(self):
        pad = S(24)
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(16), S(4)))
        tk.Label(header, text="Build training dataset", bg=WHITE, fg=INK,
                 font=font(17, "bold")).pack(anchor="w")
        self.step_lbl = tk.Label(
            header,
            text="Step 1 of 2  ·  Phases 1–4: calibrate IMX662 noise model",
            bg=WHITE, fg=SUBTLE, font=font(10))
        self.step_lbl.pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(8), 0))

        self.body = tk.Frame(self, bg=WHITE)
        self.body.pack(fill="both", expand=True, padx=pad, pady=(S(8), 0))

        self.page_calib = tk.Frame(self.body, bg=WHITE)
        self.page_synth = tk.Frame(self.body, bg=WHITE)
        self._build_calib_page(self.page_calib)
        self._build_synth_page(self.page_synth)

        log_fr = tk.Frame(self, bg=FIELD)
        log_fr.pack(fill="x", padx=pad, pady=(S(6), 0))
        self.status_lbl = tk.Label(log_fr, text="Ready.", bg=FIELD, fg=SUBTLE,
                                   font=font(9), wraplength=S(640), justify="left")
        self.status_lbl.pack(anchor="w", padx=S(10), pady=S(8))

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(6), 0))
        foot = tk.Frame(self, bg=WHITE)
        foot.pack(fill="x", padx=pad, pady=S(12))
        self.back_btn = RoundButton(foot, "BACK", self._prev_step, kind="secondary",
                                    width=100, height=38)
        self.back_btn.pack(side="left")
        self.next_btn = RoundButton(foot, "NEXT", self._next_step, kind="secondary",
                                    width=100, height=38)
        self.next_btn.pack(side="left", padx=(S(6), 0))
        self.run_btn = RoundButton(foot, "RUN", self._run_current, kind="primary",
                                   width=140, height=38)
        self.run_btn.pack(side="right")
        self.all_btn = RoundButton(foot, "RUN ALL", self._run_all, kind="primary",
                                   width=120, height=38)
        self.all_btn.pack(side="right", padx=(0, S(6)))

    def _path_row(self, parent, label, var, browse_cmd):
        row = tk.Frame(parent, bg=WHITE)
        row.pack(fill="x", pady=S(4))
        row.columnconfigure(1, weight=1)
        tk.Label(row, text=label, bg=WHITE, fg=INK, font=font(10, "bold"),
                 width=16, anchor="w").grid(row=0, column=0, sticky="w")
        ent = ttk.Entry(row, textvariable=var, font=font(10))
        ent.grid(row=0, column=1, sticky="ew", padx=(S(4), S(4)))
        RoundButton(row, "…", browse_cmd, kind="secondary",
                    width=44, height=30).grid(row=0, column=2)

    def _build_calib_page(self, parent):
        tk.Label(
            parent,
            text=("Organise Phase-1 captures: bias/ (lens capped, min exposure), "
                  "dark/ (lens capped, normal exposure), flat/level_XX/ pairs "
                  "at 10–15 brightness levels. Repeat per gain and temperature."),
            bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(620),
            justify="left").pack(anchor="w", pady=(0, S(10)))
        self._path_row(parent, "Calibration folder", self.calib_dir, self._browse_calib)
        self._path_row(parent, "Noise model JSON", self.model_out, self._browse_model_out)

        opts = tk.Frame(parent, bg=WHITE)
        opts.pack(fill="x", pady=(S(8), 0))
        tk.Label(opts, text="Sensor", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Combobox(opts, textvariable=self.sensor_var, width=10,
                     values=["imx662", "imx219", "imxng"], state="readonly",
                     style="Rpi.TCombobox").pack(side="left", padx=(S(8), S(16)))
        tk.Label(opts, text="Gain", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Combobox(opts, textvariable=self.gain_var, width=6,
                     values=["256", "512"], state="readonly",
                     style="Rpi.TCombobox").pack(side="left", padx=(S(8), S(16)))
        tk.Label(opts, text="Temp °C", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Entry(opts, textvariable=self.temp_var, width=6, font=font(10)).pack(side="left",
                                                                                  padx=(S(8), 0))

    def _build_synth_page(self, parent):
        tk.Label(
            parent,
            text=("Phase 5: inject calibrated noise on clean ground-truth images "
                  "and write PI_RAW/Data/<scene>/<sensor_test>/noisy.png + gt.png."),
            bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(620),
            justify="left").pack(anchor="w", pady=(0, S(10)))
        self._path_row(parent, "Clean images", self.clean_dir, self._browse_clean)
        self._path_row(parent, "Calibration JSON", self.calib_json, self._browse_calib_json)
        self._path_row(parent, "Output PI_RAW", self.dataset_out, self._browse_dataset_out)

        opts = tk.Frame(parent, bg=WHITE)
        opts.pack(fill="x", pady=(S(8), 0))
        tk.Label(opts, text="Layout", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Combobox(opts, textvariable=self.layout_var, width=8,
                     values=["auto", "flat", "scenes"], state="readonly",
                     style="Rpi.TCombobox").pack(side="left", padx=(S(8), S(12)))
        tk.Label(opts, text="AG tag", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Entry(opts, textvariable=self.ag_tag_var, width=8, font=font(10)).pack(
            side="left", padx=(S(8), S(12)))
        tk.Label(opts, text="GT frames", bg=WHITE, fg=INK, font=font(10, "bold")).pack(side="left")
        ttk.Entry(opts, textvariable=self.temporal_var, width=6, font=font(10)).pack(
            side="left", padx=(S(8), 0))

        tk.Checkbutton(
            parent, text="  Update config.yaml to use the new dataset",
            variable=self.write_config_var, bg=WHITE, fg=INK, selectcolor=WHITE,
            activebackground=WHITE, font=font(10)).pack(anchor="w", pady=(S(8), 0))
        tk.Checkbutton(
            parent, text="  Skip calibration — use datasheet noise model (legacy)",
            variable=self.use_legacy_var, bg=WHITE, fg=INK, selectcolor=WHITE,
            activebackground=WHITE, font=font(10),
            command=self._on_legacy_toggle).pack(anchor="w", pady=(S(4), 0))

    def _on_legacy_toggle(self):
        if self.use_legacy_var.get():
            self.calib_json.set("")

    def _show_step(self, step: int):
        self._step = max(0, min(step, 1))
        self.page_calib.pack_forget()
        self.page_synth.pack_forget()
        if self._step == 0:
            self.page_calib.pack(fill="both", expand=True)
            self.step_lbl.config(
                text="Step 1 of 2  ·  Phases 1–4: calibrate IMX662 noise model")
            self.run_btn.set_text("CALIBRATE")
        else:
            self.page_synth.pack(fill="both", expand=True)
            if not self.calib_json.get() and self.model_out.get():
                self.calib_json.set(self.model_out.get())
            self.step_lbl.config(
                text="Step 2 of 2  ·  Phase 5: synthesize PI_RAW training pairs")
            self.run_btn.set_text("BUILD DATASET")
        self.back_btn.set_enabled(self._step > 0)

    def _prev_step(self):
        self._show_step(self._step - 1)

    def _next_step(self):
        self._show_step(self._step + 1)

    def _set_status(self, text: str, ok: bool = False):
        self.status_lbl.config(text=text, fg=GREEN if ok else SUBTLE)

    def _browse_calib(self):
        p = filedialog.askdirectory(title="Phase-1 calibration folder (bias/dark/flat)")
        if p:
            self.calib_dir.set(p)

    def _browse_model_out(self):
        p = filedialog.asksaveasfilename(
            title="Save noise model JSON",
            defaultextension=".json",
            initialfile=Path(self.model_out.get() or "imx662.json").name,
            initialdir=str(ROOT / "models" / "noise"),
            filetypes=[("JSON", "*.json")])
        if p:
            self.model_out.set(p)

    def _browse_clean(self):
        p = filedialog.askdirectory(title="Folder of clean ground-truth images")
        if p:
            self.clean_dir.set(p)

    def _browse_calib_json(self):
        p = filedialog.askopenfilename(
            title="Calibrated noise model JSON",
            initialdir=str(ROOT / "models" / "noise"),
            filetypes=[("JSON", "*.json")])
        if p:
            self.calib_json.set(p)
            self.use_legacy_var.set(False)

    def _browse_dataset_out(self):
        p = filedialog.askdirectory(title="Output PI_RAW dataset root")
        if p:
            self.dataset_out.set(p)

    def _set_busy(self, on: bool):
        self._busy = on
        self.run_btn.set_enabled(not on)
        self.all_btn.set_enabled(not on)

    def _run_current(self):
        if self._step == 0:
            self._run_calibrate()
        else:
            self._run_synthesize()

    def _run_calibrate(self):
        if self._busy:
            return
        calib = self.calib_dir.get().strip()
        out = self.model_out.get().strip()
        if not calib or not Path(calib).is_dir():
            messagebox.showerror("Calibrate", "Pick a calibration folder (bias/dark/flat).")
            return
        if not out:
            messagebox.showerror("Calibrate", "Choose where to save the noise model JSON.")
            return
        temp = self.temp_var.get().strip()
        temp_c = float(temp) if temp else None

        def work():
            try:
                from nsa.noise_calib import run_calibration_pipeline
                model, validation = run_calibration_pipeline(
                    calib, out,
                    sensor=self.sensor_var.get(),
                    gain=int(self.gain_var.get()),
                    temperature_c=temp_c,
                )
                ok = validation.get("ok", True)
                msg = (f"Calibration saved → {out}\n"
                       f"shot a={model.shot_a:.4g}  read={model.read_dist.kind}  "
                       f"validation={'PASS' if ok else 'CHECK'}")
                self.after(0, lambda: self._calibrate_done(out, msg, ok))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._calibrate_fail(str(exc)))

        self._set_busy(True)
        self._set_status("Running Phases 2–4 (extract → fit → validate)…")
        threading.Thread(target=work, daemon=True).start()

    def _calibrate_done(self, out: str, msg: str, ok: bool):
        self._set_busy(False)
        self.calib_json.set(out)
        self._set_status(msg, ok=ok)
        if ok:
            messagebox.showinfo("Calibration complete", msg + "\n\nClick Next to build the dataset.")
        else:
            messagebox.showwarning("Calibration finished with warnings", msg)

    def _calibrate_fail(self, err: str):
        self._set_busy(False)
        self._set_status(f"Calibration failed: {err}")
        messagebox.showerror("Calibration failed", err)

    def _run_synthesize(self):
        if self._busy:
            return
        clean = self.clean_dir.get().strip()
        out = self.dataset_out.get().strip()
        if not clean or not Path(clean).is_dir():
            messagebox.showerror("Build dataset", "Pick a folder of clean images.")
            return
        if not out:
            messagebox.showerror("Build dataset", "Choose an output PI_RAW folder.")
            return
        calib = self.calib_json.get().strip() or None
        if not self.use_legacy_var.get() and not calib:
            messagebox.showerror("Build dataset",
                                 "Pick a calibration JSON or enable legacy datasheet noise.")
            return
        if calib and not Path(calib).is_file():
            messagebox.showerror("Build dataset", f"Calibration file not found:\n{calib}")
            return
        try:
            temporal = max(1, int(self.temporal_var.get() or "64"))
        except ValueError:
            temporal = 64

        def work():
            try:
                from nsa.dataset_sim import build_dataset
                manifest = build_dataset(
                    clean, out,
                    sensor=self.sensor_var.get(),
                    gain=int(self.gain_var.get()),
                    layout=self.layout_var.get(),
                    ag_tag=self.ag_tag_var.get().strip() or None,
                    temporal_frames=temporal,
                    calibration=None if self.use_legacy_var.get() else calib,
                    overwrite=True,
                )
                if self.write_config_var.get():
                    from nsa.denoise_hw_data import patch_config_dataset
                    patch_config_dataset(ROOT / "config.yaml", Path(out))
                n = manifest.get("pairs_written", 0)
                wf = manifest.get("workflow", "")
                self.after(0, lambda: self._synth_done(out, n, wf))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._synth_fail(str(exc)))

        self._set_busy(True)
        self._set_status("Phase 5: synthesizing noisy/gt pairs…")
        threading.Thread(target=work, daemon=True).start()

    def _synth_done(self, out: str, pairs: int, workflow: str):
        self._set_busy(False)
        msg = f"Wrote {pairs} pair(s) → {out}  ({workflow})"
        self._set_status(msg, ok=True)
        self.app.dataset_path = out
        if hasattr(self.app, "dataset_label"):
            self.app.dataset_label.config(text=out, fg=GREEN)
        self.app.source_var.set("real")
        self.app._on_source_change()
        messagebox.showinfo(
            "Dataset ready",
            msg + "\n\nconfig.yaml updated. Use Real captures + Extended training to compile.")
        self._show_step(1)

    def _synth_fail(self, err: str):
        self._set_busy(False)
        self._set_status(f"Build failed: {err}")
        messagebox.showerror("Build dataset failed", err)

    def _run_all(self):
        if self._busy:
            return
        if self.use_legacy_var.get():
            if not self.clean_dir.get().strip():
                messagebox.showinfo("Run all", "Set the clean images folder on Step 2.")
                self._show_step(1)
                return
            self._run_synthesize()
            return
        if not self.calib_dir.get().strip():
            messagebox.showinfo("Run all", "Set the calibration folder on Step 1 first.")
            self._show_step(0)
            return
        if not self.clean_dir.get().strip():
            messagebox.showinfo("Run all", "Set the clean images folder on Step 2 first.")
            self._show_step(1)
            return

        def chain():
            try:
                from nsa.noise_calib import run_calibration_pipeline
                from nsa.dataset_sim import build_dataset
                from nsa.denoise_hw_data import patch_config_dataset

                calib = self.calib_dir.get().strip()
                out_model = self.model_out.get().strip()
                temp = self.temp_var.get().strip()
                temp_c = float(temp) if temp else None
                self.after(0, lambda: self._set_status("Phases 1–4: calibrating…"))
                _model, validation = run_calibration_pipeline(
                    calib, out_model,
                    sensor=self.sensor_var.get(),
                    gain=int(self.gain_var.get()),
                    temperature_c=temp_c,
                )
                self.after(0, lambda: self.calib_json.set(out_model))
                clean = self.clean_dir.get().strip()
                out_ds = self.dataset_out.get().strip()
                temporal = max(1, int(self.temporal_var.get() or "64"))
                self.after(0, lambda: self._set_status("Phase 5: building dataset…"))
                manifest = build_dataset(
                    clean, out_ds,
                    sensor=self.sensor_var.get(),
                    gain=int(self.gain_var.get()),
                    layout=self.layout_var.get(),
                    ag_tag=self.ag_tag_var.get().strip() or None,
                    temporal_frames=temporal,
                    calibration=out_model,
                    overwrite=True,
                )
                if self.write_config_var.get():
                    patch_config_dataset(ROOT / "config.yaml", Path(out_ds))
                n = manifest.get("pairs_written", 0)
                ok = validation.get("ok", True)
                self.after(0, lambda: self._all_done(out_ds, n, ok))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._synth_fail(str(exc)))

        self._set_busy(True)
        threading.Thread(target=chain, daemon=True).start()

    def _all_done(self, out: str, pairs: int, calib_ok: bool):
        self._set_busy(False)
        msg = f"Pipeline complete — {pairs} pair(s) at {out}"
        self._set_status(msg, ok=calib_ok)
        self.app.dataset_path = out
        if hasattr(self.app, "dataset_label"):
            self.app.dataset_label.config(text=out, fg=GREEN)
        self.app.source_var.set("real")
        self.app._on_source_change()
        if calib_ok:
            messagebox.showinfo("Noise dataset ready", msg)
        else:
            messagebox.showwarning("Dataset built (check calibration)",
                                   msg + "\n\nPhase-4 validation had warnings — review the model.")


class CttCaptureWizard(tk.Toplevel):
    """Guided **Camera Tuning Tool → NSA** capture.

    Drives a Raspberry Pi CTT server over its HTTP API station-by-station
    (bias → dark → flat levels → scene bursts), shows a live preview + readback
    so you can set up the rig, fires each burst, pulls the DNGs back, and files
    them into the NSA calibration + bursts layout. A thin GUI over the tested
    ``nsa_ctt_capture`` backend; captures run on worker threads so the window
    stays responsive.
    """

    PANEL_W = 680  # ~1.77:1, matching the IMX662's 1936x1096 sensor aspect
    PANEL_H = 385

    def __init__(self, master):
        super().__init__(master, bg=WHITE)
        self.app = master
        self.title("NAS  ·  Camera capture")
        self.configure(bg=WHITE)

        import types
        import nsa_ctt_capture as backend
        from nsa.dataset_layout import MANAGER_SCENES
        self.backend = backend
        self._types = types

        self._client = None
        self._transfer = None
        self._plan: list = []
        self._idx = 0
        self._recorded: list = []
        self._project_root = None
        self._controls_range: dict = {}
        self._busy = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._imgs = {}

        # Connection / plan parameters.
        self.host_var = tk.StringVar(value="10.3.195.212")
        self.port_var = tk.StringVar(value="5000")
        self.project_var = tk.StringVar(value="imx662")
        self.root_var = tk.StringVar(value=str(ROOT / "datasets" / "imx662_project"))
        self.gain_var = tk.StringVar(value="256")
        self.mode_var = tk.StringVar(value="real pairs")   # 'real pairs' | 'calibration'
        self.agtag_var = tk.StringVar(value="ag24")
        # Real mode sweeps this analogue-gain series per scene (one pair per gain,
        # filed into imx662_ag<gain>_test).
        self.gains_var = tk.StringVar(
            value=", ".join(str(g) for g in backend.DEFAULT_GAIN_SWEEP))
        self.flatlevels_var = tk.StringVar(value="12")
        self.burst_var = tk.StringVar(value="48")
        self.scenes_var = tk.StringVar(value=", ".join(MANAGER_SCENES))
        self.transfer_var = tk.StringVar(value="archive")
        self.ssh_var = tk.StringVar(value="pi@10.3.195.212")
        self.workspace_var = tk.StringVar(value="~/ctt-server-workspace")
        self.autostart_var = tk.BooleanVar(value=True)
        self.cttcmd_var = tk.StringVar(value="ctt-server")
        self.autolight_var = tk.BooleanVar(value=True)
        # Copy the finished PI_RAW pairs to the AI-server dataset root (and verify)
        # so training picks them up. Default = the path NSA auto-detects.
        self.publish_var = tk.BooleanVar(value=True)
        self.publish_path_var = tk.StringVar(value=str(backend.SYSTEM_PI_RAW))
        self._lightbox_present = False
        self._lightbox_illums = []      # driver's channel names, for the override combo
        self._light_manual_override = False  # user hand-set the light for this station
        self._capture_lux = None  # actual lux to send with capture() — CTT requires >0 for macbeth

        self._build_chrome()
        self._show_connect()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.transient(master)
        place_window(self, 1180, 760, master=master, min_w=760, min_h=560)
        master._grab_when_ready(self)
        self.after(100, self._paint)

    # -- chrome --------------------------------------------------------------
    def _build_chrome(self):
        pad = S(24)
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(16), S(4)))
        tk.Label(header, text="Camera capture", bg=WHITE, fg=INK,
                 font=font(17, "bold")).pack(anchor="w")
        self.step_lbl = tk.Label(
            header, text="Connect to the CTT server on the Pi",
            bg=WHITE, fg=SUBTLE, font=font(10))
        self.step_lbl.pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(8), 0))

        self.body = tk.Frame(self, bg=WHITE)
        self.body.pack(fill="both", expand=True, padx=pad, pady=(S(8), 0))
        self.page_connect = tk.Frame(self.body, bg=WHITE)
        self.page_station = tk.Frame(self.body, bg=WHITE)
        self._build_connect_page(self.page_connect)
        self._build_station_page(self.page_station)

        log_fr = tk.Frame(self, bg=FIELD)
        log_fr.pack(fill="x", padx=pad, pady=(S(6), 0))
        self.status_lbl = tk.Label(log_fr, text="Ready.", bg=FIELD, fg=SUBTLE,
                                   font=font(9), wraplength=S(880), justify="left")
        self.status_lbl.pack(anchor="w", padx=S(10), pady=S(8))

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(6), 0))
        foot = tk.Frame(self, bg=WHITE)
        foot.pack(fill="x", padx=pad, pady=S(12))
        self.back_btn = RoundButton(foot, "BACK", self._prev_station,
                                    kind="secondary", width=100, height=38)
        self.back_btn.pack(side="left")
        self.skip_btn = RoundButton(foot, "SKIP", self._skip_station,
                                    kind="secondary", width=100, height=38)
        self.skip_btn.pack(side="left", padx=(S(6), 0))
        self.connect_btn = RoundButton(foot, "CONNECT", self._connect,
                                       kind="primary", width=150, height=38)
        self.connect_btn.pack(side="right")
        self.capture_btn = RoundButton(foot, "CAPTURE", self._capture,
                                       kind="primary", width=170, height=38)
        self.apply_btn = RoundButton(foot, "RE-APPLY", self._apply_current,
                                     kind="secondary", width=130, height=38)

    def _build_connect_page(self, parent):
        tk.Label(
            parent,
            text=("Point this at the CTT server running on the Pi. It captures the "
                  "bias/dark/flat frames for the noise model and static scene bursts "
                  "for ground truth, then files the DNGs into the NSA layout."),
            bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(860),
            justify="left").pack(anchor="w", pady=(0, S(10)))

        grid = tk.Frame(parent, bg=WHITE)
        grid.pack(fill="x")
        for c in (1, 3):
            grid.columnconfigure(c, weight=1)

        def row(r, label, var, col=0, hint=None):
            tk.Label(grid, text=label, bg=WHITE, fg=INK, font=font(10, "bold"),
                     anchor="w").grid(row=r, column=col*2, sticky="w", pady=S(4),
                                      padx=(0, S(6)))
            ttk.Entry(grid, textvariable=var, font=font(10)).grid(
                row=r, column=col*2+1, sticky="ew", padx=(0, S(16)))

        row(0, "Host / IP", self.host_var, 0)
        row(0, "Port", self.port_var, 1)
        row(1, "CTT project", self.project_var, 0)
        row(1, "Operating gain", self.gain_var, 1)  # bias/dark/flat all shot at this
        row(2, "Flat levels", self.flatlevels_var, 0)
        row(2, "Burst frames", self.burst_var, 1)

        # Capture mode + real-pairs folder tag.
        tk.Label(grid, text="Mode", bg=WHITE, fg=INK, font=font(10, "bold")).grid(
            row=3, column=0, sticky="w", pady=S(4))
        moderow = tk.Frame(grid, bg=WHITE)
        moderow.grid(row=3, column=1, columnspan=3, sticky="w")
        ttk.Combobox(moderow, textvariable=self.mode_var, width=14, state="readonly",
                     values=["real pairs", "calibration"],
                     style="Rpi.TCombobox").pack(side="left")
        tk.Label(moderow, text="Gains", bg=WHITE, fg=INK, font=font(10)).pack(
            side="left", padx=(S(12), S(6)))
        ttk.Entry(moderow, textvariable=self.gains_var, width=28, font=font(10)).pack(
            side="left")
        tk.Label(grid, text="real pairs = for EACH scene, sweep the Gains series (one "
                            "genuine noisy+gt pair per gain) into PI_RAW/Data/<scene>/"
                            "imx662_ag<gain>_test — one rig setup, all gains captured "
                            "back-to-back.  calibration = bias/dark/flat noise model + "
                            "synthesis.", bg=WHITE,
                 fg=SUBTLE, font=font(8), wraplength=S(820), justify="left").grid(
                     row=4, column=0, columnspan=4, sticky="w", pady=(S(2), S(2)))

        tk.Label(grid, text="NSA project root", bg=WHITE, fg=INK,
                 font=font(10, "bold")).grid(row=5, column=0, sticky="w", pady=S(4))
        rootrow = tk.Frame(grid, bg=WHITE)
        rootrow.grid(row=5, column=1, columnspan=3, sticky="ew")
        rootrow.columnconfigure(0, weight=1)
        ttk.Entry(rootrow, textvariable=self.root_var, font=font(10)).grid(
            row=0, column=0, sticky="ew")
        RoundButton(rootrow, "…", self._browse_root, kind="secondary",
                    width=44, height=30).grid(row=0, column=1, padx=(S(4), 0))

        tk.Label(grid, text="Scenes", bg=WHITE, fg=INK,
                 font=font(10, "bold")).grid(row=6, column=0, sticky="w", pady=S(4))
        ttk.Entry(grid, textvariable=self.scenes_var, font=font(10)).grid(
            row=6, column=1, columnspan=3, sticky="ew")

        # Transfer method.
        tfr = tk.Frame(parent, bg=WHITE)
        tfr.pack(fill="x", pady=(S(12), 0))
        tk.Label(tfr, text="Transfer", bg=WHITE, fg=INK,
                 font=font(10, "bold")).pack(side="left")
        ttk.Combobox(tfr, textvariable=self.transfer_var, width=10,
                     values=["archive", "rsync"], state="readonly",
                     style="Rpi.TCombobox").pack(side="left", padx=(S(8), S(16)))
        tk.Label(tfr, text="SSH (rsync)", bg=WHITE, fg=INK,
                 font=font(10)).pack(side="left")
        ttk.Entry(tfr, textvariable=self.ssh_var, width=20, font=font(10)).pack(
            side="left", padx=(S(6), S(12)))
        tk.Label(tfr, text="Pi workspace", bg=WHITE, fg=INK,
                 font=font(10)).pack(side="left")
        ttk.Entry(tfr, textvariable=self.workspace_var, width=22, font=font(10)).pack(
            side="left", padx=(S(6), 0))
        tk.Label(
            parent,
            text=("archive = pull the project ZIP over HTTPS (no setup). "
                  "rsync = incremental pull over SSH (needs key access to the Pi)."),
            bg=WHITE, fg=SUBTLE, font=font(8), wraplength=S(860),
            justify="left").pack(anchor="w", pady=(S(6), 0))

        # Auto-start: SSH in and launch ctt-server if it isn't already running.
        asr = tk.Frame(parent, bg=WHITE)
        asr.pack(fill="x", pady=(S(10), 0))
        tk.Checkbutton(
            asr, text="  Auto-start ctt-server on the Pi via SSH if it's not running",
            variable=self.autostart_var, bg=WHITE, fg=INK, selectcolor=WHITE,
            activebackground=WHITE, font=font(10)).pack(side="left")
        tk.Label(asr, text="CTT command", bg=WHITE, fg=INK,
                 font=font(10)).pack(side="left", padx=(S(12), S(6)))
        ttk.Entry(asr, textvariable=self.cttcmd_var, width=24, font=font(10)).pack(
            side="left")

        # Lightbox: if a lightSTUDIO-S is attached to the Pi, auto-select the
        # illuminant from each scene's name (cabinet_D50_100 → D50) at a default
        # intensity. Intensity is set as a % — no target-lux metering.
        lbr = tk.Frame(parent, bg=WHITE)
        lbr.pack(fill="x", pady=(S(6), 0))
        tk.Checkbutton(
            lbr, text="  Auto-select lightbox illuminant from scene name",
            variable=self.autolight_var, bg=WHITE, fg=INK, selectcolor=WHITE,
            activebackground=WHITE, font=font(10)).pack(side="left")
        tk.Label(
            parent,
            text=("Needs a lightSTUDIO-S plugged into the Pi. Scenes named "
                  "<name>_<illuminant>_<lux> (e.g. cabinet_D50_100) get that "
                  "illuminant switched on at the default intensity; set the exact "
                  "intensity % per station with the SET button. Other scene names "
                  "are left to manual lighting."),
            bg=WHITE, fg=SUBTLE, font=font(8), wraplength=S(860),
            justify="left").pack(anchor="w", pady=(S(2), 0))

        # Publish: copy the finished PI_RAW pairs onto the AI-server dataset root
        # and read them back to confirm they landed (so training sees them).
        pub = tk.Frame(parent, bg=WHITE)
        pub.pack(fill="x", pady=(S(8), 0))
        tk.Checkbutton(
            pub, text="  Copy finished pairs to AI-server dataset",
            variable=self.publish_var, bg=WHITE, fg=INK, selectcolor=WHITE,
            activebackground=WHITE, font=font(10)).pack(side="left")
        tk.Label(pub, text="dataset root", bg=WHITE, fg=INK,
                 font=font(10)).pack(side="left", padx=(S(12), S(6)))
        ttk.Entry(pub, textvariable=self.publish_path_var, width=30,
                  font=font(10)).pack(side="left", fill="x", expand=True)
        tk.Label(
            parent,
            text=("After the pairs are built they're copied here and verified "
                  "(size-checked read-back). This is the path NSA training "
                  "auto-detects on the AI machine — leave it as /opt/datasets/PI_RAW "
                  "unless your dataset lives elsewhere."),
            bg=WHITE, fg=SUBTLE, font=font(8), wraplength=S(860),
            justify="left").pack(anchor="w", pady=(S(2), 0))

    def _build_station_page(self, parent):
        cols = tk.Frame(parent, bg=WHITE)
        cols.pack(fill="both", expand=True)
        cols.columnconfigure(0, weight=0)
        cols.columnconfigure(1, weight=1)

        # Left: live preview + readback. Fixed-pixel holder so the (imageless)
        # label doesn't balloon to character-unit dimensions before frame 1.
        left = tk.Frame(cols, bg=WHITE)
        left.grid(row=0, column=0, sticky="nw", padx=(0, S(16)))
        holder = tk.Frame(left, bg="#111111", width=S(self.PANEL_W), height=S(self.PANEL_H))
        holder.pack()
        holder.pack_propagate(False)
        self.preview_lbl = tk.Label(holder, text="live preview…", bg="#111111",
                                    fg="#888888", font=font(9))
        self.preview_lbl.pack(fill="both", expand=True)
        self.readback_lbl = tk.Label(left, text="", bg=WHITE, fg=INK,
                                     font=font(9), justify="left", anchor="w")
        self.readback_lbl.pack(anchor="w", pady=(S(8), 0))
        self.clip_lbl = tk.Label(left, text="", bg=WHITE, fg=SUBTLE,
                                 font=font(9), justify="left", anchor="w")
        self.clip_lbl.pack(anchor="w")

        # Right: station title + setup instructions + applied settings.
        right = tk.Frame(cols, bg=WHITE)
        right.grid(row=0, column=1, sticky="new")
        self.station_title = tk.Label(right, text="", bg=WHITE, fg=RASPBERRY,
                                      font=font(13, "bold"), justify="left", anchor="w")
        self.station_title.pack(anchor="w")
        self.setup_lbl = tk.Label(right, text="", bg=WHITE, fg=INK, font=font(10),
                                  wraplength=S(360), justify="left", anchor="w")
        self.setup_lbl.pack(anchor="w", pady=(S(8), 0))
        self.applied_lbl = tk.Label(right, text="", bg=FIELD, fg=INK, font=font(9),
                                    wraplength=S(360), justify="left", anchor="w")
        self.applied_lbl.pack(anchor="w", fill="x", pady=(S(12), 0), ipady=S(6),
                              ipadx=S(6))

        # Lightbox panel — only shown for scene stations, when a lightSTUDIO-S
        # is attached. Shows the auto-metered result and lets you override it.
        self.light_fr = tk.Frame(right, bg=WHITE)
        self.light_lbl = tk.Label(self.light_fr, text="", bg=WHITE, fg=SUBTLE,
                                  font=font(9), wraplength=S(360), justify="left",
                                  anchor="w")
        self.light_lbl.pack(anchor="w")
        override = tk.Frame(self.light_fr, bg=WHITE)
        override.pack(anchor="w", pady=(S(4), 0))
        tk.Label(override, text="Illuminant", bg=WHITE, fg=INK,
                 font=font(9)).pack(side="left")
        self.light_illum_var = tk.StringVar(value="")
        self.light_illum_combo = ttk.Combobox(
            override, textvariable=self.light_illum_var, width=10, state="readonly",
            style="Rpi.TCombobox")
        self.light_illum_combo.pack(side="left", padx=(S(4), S(10)))
        tk.Label(override, text="intensity %", bg=WHITE, fg=INK,
                 font=font(9)).pack(side="left")
        self.light_pct_var = tk.StringVar(value="100")
        ttk.Entry(override, textvariable=self.light_pct_var, width=5,
                 font=font(9)).pack(side="left", padx=(S(4), S(8)))
        RoundButton(override, "SET", self._set_light_manual, kind="secondary",
                   width=64, height=26).pack(side="left")

        self.progress_lbl = tk.Label(right, text="", bg=WHITE, fg=SUBTLE,
                                     font=font(9), anchor="w")
        self.progress_lbl.pack(anchor="w", pady=(S(10), 0))

    # -- page switching ------------------------------------------------------
    def _show_connect(self):
        self.page_station.pack_forget()
        self.page_connect.pack(fill="both", expand=True)
        self.step_lbl.config(text="Connect to the CTT server on the Pi")
        self.connect_btn.pack(side="right")
        self.capture_btn.pack_forget()
        self.apply_btn.pack_forget()
        self.back_btn.set_enabled(False)
        self.skip_btn.set_enabled(False)

    def _show_station_page(self):
        self.page_connect.pack_forget()
        self.page_station.pack(fill="both", expand=True)
        self.connect_btn.pack_forget()
        self.apply_btn.pack(side="right", padx=(0, S(6)))
        self.capture_btn.pack(side="right")

    # -- helpers -------------------------------------------------------------
    def _set_status(self, text: str, kind: str = "info"):
        colour = {"ok": GREEN, "warn": AMBER, "err": RASPBERRY}.get(kind, SUBTLE)
        self.status_lbl.config(text=text, fg=colour)

    def _set_busy(self, on: bool):
        self._busy = on
        for b in (self.capture_btn, self.apply_btn, self.back_btn,
                  self.skip_btn, self.connect_btn):
            b.set_enabled(not on)

    def _browse_root(self):
        p = filedialog.askdirectory(title="NSA project root (PI_RAW / calibration / bursts)")
        if p:
            self.root_var.set(p)

    def _args_namespace(self):
        scenes = [s.strip() for s in self.scenes_var.get().split(",") if s.strip()]
        return self._types.SimpleNamespace(
            gain=int(self.gain_var.get() or "256"),
            # 0 → calibrate bias/dark/flat at the target gain (the real operating
            # gain), so the fitted noise model matches the low-light regime.
            analogue_gain=0.0,
            bias_frames=8, dark_frames=5, dark_exposure_ms=20.0,
            flat_levels=int(self.flatlevels_var.get() or "12"),
            flat_gain=0.0, flat_min_ms=1.0, flat_max_ms=30.0,
            burst_frames=int(self.burst_var.get() or "48"),
            scenes=scenes, colour_temp=5000, lux=None,
            mode="real" if self.mode_var.get().startswith("real") else "calib",
            ag_tag=(self.agtag_var.get().strip() or "ag24"),
            gain_sweep=self.backend.parse_gain_sweep(self.gains_var.get()),
        )

    # -- connect -------------------------------------------------------------
    def _connect(self):
        if self._busy:
            return
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Connect", "Port must be a number.")
            return
        from nsa.dataset_layout import resolve_layout, scaffold_imx662_project
        args = self._args_namespace()

        ssh = self.ssh_var.get().strip() or None
        autostart = bool(self.autostart_var.get())
        ctt_cmd = self.cttcmd_var.get().strip() or "ctt-server"
        workspace = self.workspace_var.get().strip() or None

        def status(msg):
            self.after(0, lambda: self._set_status(msg))

        def work():
            try:
                project_root, _ = resolve_layout(self.root_var.get().strip())
                scaffold_imx662_project(
                    project_root, gain=args.gain,
                    scenes=tuple(args.scenes),
                    flat_levels=max(2, args.flat_levels))
                client = self.backend.CTTClient(host, port)
                # SSH in and start ctt-server if it isn't answering yet.
                self.backend.ensure_server(
                    client, ssh=ssh, ctt_cmd=ctt_cmd, port=port,
                    workspace=workspace, autostart=autostart, status=status)
                h = client.health()
                if not h.get("camera"):
                    raise self.backend.CTTError(
                        f"CTT is up but reports no camera: {h.get('error', 'unknown')}")
                client.ensure_project(self.project_var.get().strip())
                controls = client.get_controls()
                plan = self.backend.build_plan(project_root, args, controls)
                self.after(0, lambda: self._connected(client, project_root, plan))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._connect_fail(str(e)))

        self._set_busy(True)
        self._set_status(f"Connecting to https://{host}:{port} …")
        threading.Thread(target=work, daemon=True).start()

    def _connect_fail(self, err: str):
        self._set_busy(False)
        self._set_status(f"Connect failed: {err}", "err")
        messagebox.showerror("Connect failed", err)

    def _connected(self, client, project_root, plan):
        self._set_busy(False)
        self._client = client
        self._project_root = project_root
        self._plan = plan
        self._idx = 0
        self._recorded = []
        # Build the transfer backend.
        mirror = project_root / ".ctt_mirror"
        try:
            if self.transfer_var.get() == "rsync":
                self._transfer = self.backend.RsyncTransfer(
                    self.ssh_var.get().strip(), self.workspace_var.get().strip(),
                    self.project_var.get().strip(), mirror)
            else:
                self._transfer = self.backend.ArchiveTransfer(
                    client, self.project_var.get().strip(), mirror)
        except self.backend.CTTError as exc:
            messagebox.showerror("Transfer", str(exc))
            return
        # Probe the lightbox once so the station page knows whether to offer
        # auto-lighting / manual override.
        try:
            lb = client.lightbox_status() or {}
        except Exception:  # noqa: BLE001
            lb = {}
        self._lightbox_present = bool(lb.get("present"))
        self._lightbox_illums = list(lb.get("illuminants", {}).values())
        if self._lightbox_illums:
            self.light_illum_combo.config(values=self._lightbox_illums)

        status = f"Connected — camera ready. {len(plan)} stations planned."
        if self._lightbox_present:
            status += f"  Lightbox: {lb.get('model', 'detected')}."
        self._set_status(status, "ok")
        self._show_station_page()
        self._start_preview()
        self._poll_controls()
        self._show_station(0)

    # -- live preview + readback --------------------------------------------
    def _start_preview(self):
        threading.Thread(target=self._preview_worker, daemon=True).start()

    def _preview_worker(self):
        url = f"{self._client.base}/api/preview"
        try:
            r = self._client.s.get(url, stream=True, timeout=15)
            buf = b""
            for chunk in r.iter_content(chunk_size=16384):
                if self._stop.is_set():
                    break
                buf += chunk
                a = buf.find(b"\xff\xd8")
                b_ = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                if a != -1 and b_ != -1:
                    with self._lock:
                        self._latest_jpeg = buf[a:b_ + 2]
                    buf = buf[b_ + 2:]
                if len(buf) > 4_000_000:  # guard against runaway buffering
                    buf = b""
        except Exception:  # noqa: BLE001
            pass

    def _paint(self):
        if self._stop.is_set():
            return
        import io
        with self._lock:
            jpg = self._latest_jpeg
        if jpg and Image is not None:
            try:
                im = Image.open(io.BytesIO(jpg)).convert("RGB")
                im.thumbnail((S(self.PANEL_W), S(self.PANEL_H)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(im)
                self._imgs["preview"] = photo
                self.preview_lbl.config(image=photo, text="")
            except Exception:  # noqa: BLE001
                pass
        self.after(120, self._paint)

    def _poll_controls(self):
        if self._stop.is_set() or self._client is None:
            return

        def work():
            try:
                c = self._client.get_controls()
                h = self._client.histogram()
            except Exception:  # noqa: BLE001
                return
            self.after(0, lambda: self._update_readback(c, h))

        threading.Thread(target=work, daemon=True).start()
        self.after(1300, self._poll_controls)

    def _update_readback(self, c: dict, h: dict):
        self.readback_lbl.config(text=(
            f"exposure  {c.get('exposure', 0)/1000:.2f} ms      "
            f"gain  {c.get('gain', 0):g}×\n"
            f"colour temp  {c.get('colour_temp', 0)} K      "
            f"lux  {c.get('lux', 0)}      focus  {c.get('focus_fom', 0)}\n"
            f"auto-exposure  {'ON' if c.get('auto_exposure') else 'off (locked)'}"))
        clip = h.get("clipping") or {}
        if clip:
            hot = any(float(v) > 1.0 for v in clip.values())
            self.clip_lbl.config(
                text="clipping  " + "  ".join(f"{k} {v}%" for k, v in clip.items()),
                fg=AMBER if hot else SUBTLE)

    # -- station flow --------------------------------------------------------
    def _show_station(self, idx: int):
        if idx >= len(self._plan):
            self._finish()
            return
        self._idx = max(0, idx)
        st = self._plan[self._idx]
        self.step_lbl.config(text=f"Station {self._idx+1} of {len(self._plan)}")
        self.station_title.config(text=st.title)
        self.setup_lbl.config(text=st.setup)
        if st.meta.get("gain_sweep"):
            gains = st.meta["gain_sweep"]
            self.progress_lbl.config(
                text=(f"→ sweeps gains {', '.join(str(g) for g in gains)} into  "
                      f"{st.meta['pair_root']}/imx662_ag<gain>_test"))
        else:
            self.progress_lbl.config(text=f"→ files land in  {st.dest}")
        self.back_btn.set_enabled(self._idx > 0)
        self.skip_btn.set_enabled(True)

        # Lightbox panel only makes sense for scene stations with a device attached.
        self._light_manual_override = False
        self._capture_lux = st.lux  # CTT requires a positive lux for macbeth captures
        is_scene = bool(st.meta.get("is_real_pair"))
        if is_scene and self._lightbox_present:
            illum, _ = self.backend.parse_scene_light(st.meta.get("scene", ""))
            self.light_illum_var.set(illum or (self._lightbox_illums[0]
                                                if self._lightbox_illums else ""))
            self.light_lbl.config(text="Lightbox ready.", fg=SUBTLE)
            self.light_fr.pack(anchor="w", fill="x", pady=(S(10), 0))
        else:
            self.light_fr.pack_forget()

        self._apply_current()

    def _prev_station(self):
        if not self._busy:
            self._show_station(self._idx - 1)

    def _skip_station(self):
        if not self._busy:
            self._set_status(f"Skipped {self._plan[self._idx].station_id}.", "warn")
            self._show_station(self._idx + 1)

    def _apply_current(self):
        if self._busy or not self._plan:
            return
        st = self._plan[self._idx]
        auto_light = (bool(st.meta.get("is_real_pair")) and self._lightbox_present
                     and bool(self.autolight_var.get())
                     and not self._light_manual_override)

        try:
            pct = float(self.light_pct_var.get())
        except (ValueError, tk.TclError):
            pct = self.backend.DEFAULT_LIGHTBOX_PERCENT

        def work():
            try:
                light_info = None
                # Light the scene BEFORE the camera auto-meters, otherwise the
                # locked exposure is metered against the wrong brightness. The
                # box is driven purely by intensity % (no target-lux search).
                if auto_light:
                    illum, _ = self.backend.parse_scene_light(st.meta.get("scene", ""))
                    if illum:
                        self._client.set_lightbox(illum, pct)
                        meas = self._client._settled_lux()
                        light_info = (illum, pct, meas)
                applied = self.backend._apply_controls(self._client, st)
                self.after(0, lambda: self._applied(applied, light_info))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._set_status(f"Apply failed: {e}", "err"))

        self._set_busy(True)
        self._set_status("Applying camera settings…" if not auto_light
                         else "Setting lightbox, then applying camera settings…")
        threading.Thread(target=work, daemon=True).start()

    def _applied(self, applied: dict, light_info: tuple | None = None):
        self._set_busy(False)
        st = self._plan[self._idx]
        mode = "auto-metered then locked" if st.controls is None else "locked"
        # Bias/dark stations run at minimal/zero light, so the preview is black
        # by design — call that out so it doesn't read as a broken feed.
        dark_note = ""
        if st.image_type == "dark":
            dark_note = ("\nThe live preview is BLACK here on purpose — this station "
                         "measures the sensor with the lens capped / minimal exposure.")
        self.applied_lbl.config(text=(
            f"Applied ({mode}):  exposure {applied.get('exposure', 0)/1000:.2f} ms  ·  "
            f"gain {applied.get('gain', 0):g}×\n"
            f"Set up the rig as above, then press CAPTURE "
            f"({st.frames} frame(s))." + dark_note))
        if light_info:
            illum, pct, meas = light_info
            self.light_illum_var.set(illum)
            self.light_pct_var.set(f"{pct:.0f}")
            if meas > 0:
                self._capture_lux = int(round(meas))  # measured lux, metadata only
            self.light_lbl.config(
                text=(f"Set {illum} to {pct:.0f}% → measured {meas:.0f} lux. "
                     "Adjust the intensity % and press SET if needed."),
                fg=SUBTLE)
        elif st.meta.get("is_real_pair") and self._lightbox_present:
            self.light_lbl.config(
                text="Auto-light off or scene name unparseable — set manually below.",
                fg=SUBTLE)
        self._set_status("Ready to capture.", "ok")

    # -- lightbox manual override --------------------------------------------
    def _set_light_manual(self):
        if self._busy or self._client is None:
            return
        illum = self.light_illum_var.get().strip()
        if not illum:
            messagebox.showerror("Lightbox", "Pick an illuminant first.")
            return
        try:
            pct = float(self.light_pct_var.get())
        except ValueError:
            messagebox.showerror("Lightbox", "Percent must be a number.")
            return
        self._light_manual_override = True

        def work():
            try:
                self._client.set_lightbox(illum, pct)
                meas = self._client._settled_lux()
                if meas > 0:
                    self._capture_lux = int(round(meas))  # metadata only
                self.after(0, lambda: self._set_status(
                    f"Lightbox set to {illum} at {pct:.0f}% (measured {meas:.0f} lux). "
                    "Press RE-APPLY to re-lock exposure to this light.", "ok"))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._set_status(
                    f"Lightbox set failed: {e}", "err"))

        threading.Thread(target=work, daemon=True).start()

    def _capture(self):
        if self._busy or not self._plan:
            return
        st = self._plan[self._idx]
        project = self.project_var.get().strip()
        incremental = isinstance(self._transfer, self.backend.RsyncTransfer)
        capture_lux = self._capture_lux if self._capture_lux else st.lux

        # Real-pair scenes sweep the whole gain series after this single setup.
        if st.meta.get("gain_sweep"):
            self._set_busy(True)
            self._set_status("Metering, then sweeping the gain series…")
            self._capture_sweep(st, project, incremental)
            return

        def work():
            try:
                added = self._client.capture(
                    project, image_type=st.image_type, frames=st.frames,
                    colour_temp=st.colour_temp, lux=capture_lux)
                fnames = [a["filename"] for a in added
                          if a.get("filename", "").lower().endswith(".dng")]
                rec = self.backend.Recorded(station=st, ctt_filenames=fnames)
                self._recorded.append(rec)
                placed = 0
                if incremental:
                    mirror = self._transfer.fetch(fnames)
                    placed = self.backend._place_files(mirror, rec)
                self.after(0, lambda: self._captured(len(fnames), placed, incremental))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._capture_fail(str(e)))

        self._set_busy(True)
        self._set_status(f"Capturing {st.frames} frame(s)…")
        threading.Thread(target=work, daemon=True).start()

    def _capture_sweep(self, st, project: str, incremental: bool):
        def status(msg, kind="info"):
            self.after(0, lambda: self._set_status(msg, kind))

        def work():
            try:
                recs = self.backend.capture_gain_sweep(
                    self._client, project, st, burst_frames=st.frames,
                    status=status, stop=self._stop.is_set)
                self._recorded.extend(recs)
                placed = 0
                if incremental:
                    for rec in recs:
                        mirror = self._transfer.fetch(rec.ctt_filenames)
                        placed += self.backend._place_files(mirror, rec)
                self.after(0, lambda: self._swept(len(recs), placed, incremental))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._capture_fail(str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _swept(self, n_gains: int, placed: int, incremental: bool):
        self._set_busy(False)
        if incremental:
            self._set_status(f"Swept {n_gains} gain(s), filed {placed} DNG(s) → advancing.", "ok")
        else:
            self._set_status(f"Swept {n_gains} gain(s) (pulled at finish) → advancing.", "ok")
        self.after(600, lambda: self._show_station(self._idx + 1))

    def _capture_fail(self, err: str):
        self._set_busy(False)
        self._set_status(f"Capture failed: {err}", "err")
        messagebox.showerror("Capture failed", err)

    def _captured(self, n: int, placed: int, incremental: bool):
        self._set_busy(False)
        if incremental:
            self._set_status(f"Captured {n} DNG(s), filed {placed} → advancing.", "ok")
        else:
            self._set_status(f"Captured {n} DNG(s) (pulled at finish) → advancing.", "ok")
        self.after(600, lambda: self._show_station(self._idx + 1))

    # -- finish --------------------------------------------------------------
    def _finish(self):
        self.step_lbl.config(text="Capture complete")
        self.station_title.config(text="All stations captured")
        n_scenes = sum(1 for r in self._recorded if r.station.station_id.startswith("burst_"))
        self.setup_lbl.config(text=(
            f"Captured {len(self._recorded)} station(s), {n_scenes} scene burst(s).\n"
            "Finalising the transfer and filing DNGs into the NSA layout…"))
        self.capture_btn.pack_forget()
        self.apply_btn.pack_forget()
        self.skip_btn.set_enabled(False)

        def work():
            try:
                mirror = self._transfer.finalize()
                total = 0
                for rec in self._recorded:
                    have = len(list(rec.station.dest.glob("*.dng"))) \
                        if rec.station.dest.is_dir() else 0
                    if have >= len(rec.ctt_filenames) and have > 0:
                        continue
                    total += self.backend._place_files(mirror, rec)
                self.after(0, lambda: self._finished(total))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._set_status(f"Finalise failed: {e}", "err"))

        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _finished(self, placed: int):
        self._set_busy(False)
        self.back_btn.set_enabled(False)
        rawpy_ok = self.backend._have_rawpy()
        real = any(r.station.meta.get("is_real_pair") for r in self._recorded)

        if real:
            # Derive real noisy/gt pairs (temporal average) into PI_RAW.
            if not rawpy_ok:
                messagebox.showwarning(
                    "rawpy not installed",
                    "The burst DNGs are captured, but averaging them into gt.png "
                    "needs rawpy:\n\n    pip install rawpy")
            self.applied_lbl.config(text=(
                f"Filed {placed} DNG(s). Building real noisy/gt pairs "
                "(temporal average)…"))
            self._set_status("Deriving real pairs…")

            def work():
                out = []
                for rec in self._recorded:
                    meta = rec.station.meta
                    if not meta.get("is_real_pair"):
                        continue
                    try:
                        res = self.backend.derive_real_pair(
                            rec.station.dest, meta["pair_dest"],
                            min_frames=min(8, int(self.burst_var.get() or "48")))
                        self.backend.write_gain_sidecar(meta["pair_dest"], meta)
                        tag = (f" ag{meta['requested_gain']}→{meta.get('actual_gain', '?')}×"
                               if meta.get("requested_gain") is not None else "")
                        out.append(f"{meta['scene']}{tag}: {res['noisy']} + {res['gt']} "
                                   f"(gt from {res['frames_used']} frames)")
                    except Exception as exc:  # noqa: BLE001
                        out.append(f"{meta['scene']}: FAILED — {exc}")
                self.after(0, lambda: self._pairs_done(out))

            self._set_busy(True)
            threading.Thread(target=work, daemon=True).start()
            return

        # Calibration mode → average each scene's burst into a clean GT
        # reference (clean_scenes/<scene>/gt_01.png), same as the CLI's
        # post_process() step, THEN hand off to the noise dataset builder.
        # Without this, "Clean images" in step 2 points at an empty folder.
        gain = self.gain_var.get()
        cal_dir = self._project_root / f"calibration/imx662_gain{gain}"
        clean_dir = self._project_root / "clean_scenes"
        if not rawpy_ok:
            messagebox.showwarning(
                "rawpy not installed",
                "The DNGs are captured and filed, but building the clean GT "
                "references and noise model needs rawpy to decode raw "
                "DNGs:\n\n    pip install rawpy")
        self.applied_lbl.config(text=(
            f"Filed {placed} DNG(s).\n"
            f"Calibration → {cal_dir}\n"
            "Averaging scene bursts into clean GT references…"))
        self._set_status("Building clean GT references…")

        def work():
            from nsa.gt_capture import burst_folder_to_gt
            out = []
            for rec in self._recorded:
                if not rec.station.station_id.startswith("burst_"):
                    continue
                scene = rec.station.meta.get("scene")
                gt_path = clean_dir / scene / "gt_01.png"
                try:
                    manifest = burst_folder_to_gt(
                        str(rec.station.dest), str(gt_path),
                        min_frames=min(8, int(self.burst_var.get() or "48")))
                    out.append(f"{scene}: gt_01.png ({manifest['frames_used']} frames)")
                except Exception as exc:  # noqa: BLE001
                    out.append(f"{scene}: FAILED — {exc}")
            self.after(0, lambda: self._clean_scenes_done(cal_dir, clean_dir, out))

        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _clean_scenes_done(self, cal_dir: Path, clean_dir: Path, lines: list):
        self._set_busy(False)
        self.applied_lbl.config(text=(
            "Clean GT references built:\n" + "\n".join(lines) +
            f"\n\nCalibration → {cal_dir}\nClean scenes → {clean_dir}"))
        self._set_status("Ready — build the noise model + dataset next.", "ok")
        build = RoundButton(self, "BUILD NOISE DATASET", self._open_builder,
                            kind="primary", width=240, height=42)
        build.pack(pady=(0, S(10)))
        self._builder_prefill = (str(cal_dir), str(clean_dir))

    def _pairs_done(self, lines: list):
        self._set_busy(False)
        pi_raw = self._project_root / "PI_RAW" / "Data"
        self.applied_lbl.config(text=(
            "Real noisy/gt pairs written under\n"
            f"{pi_raw}\n\n" + "\n".join(lines)))
        self._set_status(f"Done — {len(lines)} real pair(s) in PI_RAW.", "ok")
        self._publish_pairs()

    def _publish_pairs(self):
        """Copy the finished PI_RAW pairs to the AI-server dataset root and
        confirm (read-back) that they actually landed there."""
        if not self.publish_var.get():
            return
        dest = self.publish_path_var.get().strip()
        if not dest:
            return
        src = self._project_root / "PI_RAW"
        self._set_status(f"Publishing to AI server ({dest})…")
        self._set_busy(True)

        def work():
            summary = self.backend.publish_pi_raw(src, dest)
            self.after(0, lambda: self._published(summary))

        threading.Thread(target=work, daemon=True).start()

    def _published(self, summary: dict):
        self._set_busy(False)
        dest = summary["dest"]
        prior = self.applied_lbl.cget("text")
        if summary["error"]:
            self.applied_lbl.config(text=prior + f"\n\nAI-server copy FAILED:\n{summary['error']}")
            self._set_status(f"NOT saved to AI server: {summary['error']}", "err")
            messagebox.showerror(
                "AI-server copy failed",
                f"The pairs are saved locally, but could NOT be copied to\n{dest}\n\n"
                f"{summary['error']}")
            return
        verified = summary["verified"]
        total = verified + len(summary["failures"])
        if summary["failures"]:
            self.applied_lbl.config(text=(
                prior + f"\n\nAI server ({dest}): verified {verified}/{total}; "
                f"FAILED: {', '.join(summary['failures'][:4])}"))
            self._set_status(f"Partly saved to AI server: {verified}/{total} verified.", "warn")
        else:
            self.applied_lbl.config(text=(
                prior + f"\n\nConfirmed on AI server:\n{dest}\n"
                f"{verified} file(s) present (copied {summary['copied']}, "
                f"already-there {summary['skipped']})."))
            self._set_status(
                f"Confirmed — {verified} file(s) saved on AI server ({dest}).", "ok")

    def _open_builder(self):
        # Hand off to the existing noise dataset wizard, prefilled.
        try:
            wiz = NoiseDatasetWizard(self.app)
            cal_dir, clean_dir = self._builder_prefill
            wiz.calib_dir.set(cal_dir)
            wiz.gain_var.set(self.gain_var.get())
            if Path(clean_dir).is_dir():
                wiz.clean_dir.set(clean_dir)
            self._on_close()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Noise dataset wizard", str(exc))

    def _on_close(self):
        self._stop.set()
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        _resolve_font_family()
        # Shrink the default scale to the actual display before building anything,
        # so windows fit and stay resizable on small Linux panels.
        if not os.environ.get("NSA_UI_SCALE"):
            fit_scale_to_screen(self)
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
            icon = _load_logo_photo()
            if icon is not None:
                self.iconphoto(True, icon)
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
        self._live_view = None          # in-app LiveView window (if open)

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
        self.extended_train_var = tk.BooleanVar(value=False)
        self.extended_steps_var = tk.StringVar(value="1500")
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

        self._build_goal_card(home_body)
        self._build_noise_dataset_card(home_body)

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
        self._apply_gui_state(self._load_gui_state())   # last-used settings win
        self._step = 0
        self._on_sensor_change()
        self._on_mode_change()
        self._on_source_change()
        self._on_eval_change()
        self._show_home()
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

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

        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))
        self._check(body, "Extended training on the full dataset",
                    "After the quick calibration, keep training on EVERY paired "
                    "image in the dataset (all of PI_RAW). Much stronger denoising, "
                    "but slower. Recommended when you have real captures.",
                    self.extended_train_var, command=self._on_extended_toggle)
        ext_row = tk.Frame(body, bg=WHITE)
        ext_row.pack(fill="x", pady=(0, S(4)))
        ext_row.columnconfigure(0, weight=1)
        ext_left = tk.Frame(ext_row, bg=WHITE); ext_left.grid(row=0, column=0, sticky="w")
        self._ext_steps_lbl = tk.Label(ext_left, text="     Extended steps",
                                       bg=WHITE, fg=INK, font=font(10))
        self._ext_steps_lbl.pack(anchor="w")
        tk.Label(ext_left, text="     more steps = better quality, longer wait "
                 "(typical 1000–3000)", bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(anchor="w")
        self._ext_steps_entry = ttk.Entry(ext_row, textvariable=self.extended_steps_var,
                                          width=18, font=font(10))
        self._ext_steps_entry.grid(row=0, column=1, sticky="e", padx=(S(8), 0))
        self._on_extended_toggle()

    def _on_extended_toggle(self):
        on = bool(self.extended_train_var.get())
        try:
            self._ext_steps_entry.config(state="normal" if on else "disabled")
            self._ext_steps_lbl.config(fg=INK if on else "#B6B6B6")
        except (tk.TclError, AttributeError):
            pass

    def _step_review(self, body):
        self._section(body, "REVIEW")
        self._review_box = tk.Frame(body, bg=WHITE)
        self._review_box.pack(fill="x", pady=(S(2), 0))
        tk.Label(body, text="     Use BACK to change anything, or press the button "
                 "below to launch. A sweep takes a few minutes; a single compile is "
                 "quicker.", bg=WHITE, fg=SUBTLE, font=font(9), wraplength=S(560),
                 justify="left").pack(anchor="w", pady=(S(10), 0))

    # -- Wizard navigation ----------------------------------------------------
    # -- Goal-based auto search ---------------------------------------------
    def _build_goal_card(self, parent):
        """A 'state your goal, get the best model' card at the top of Home.

        The user types plain-English constraints (task, target chip, latency
        budget, quality-vs-speed) and we translate them into an architecture
        sweep that returns the best-fitting model. Only denoising is wired up
        today; the card is built to grow to other tasks later.
        """
        card = tk.Frame(parent, bg=FIELD, highlightthickness=1,
                        highlightbackground=LINE, highlightcolor=LINE)
        card.pack(fill="x", pady=(S(4), S(16)))
        inner = tk.Frame(card, bg=FIELD)
        inner.pack(fill="x", padx=S(16), pady=S(14))

        head = tk.Frame(inner, bg=FIELD)
        head.pack(fill="x")
        tk.Label(head, text="AUTO", bg=RASPBERRY, fg=WHITE,
                 font=font(8, "bold"), padx=S(6), pady=S(1)).pack(side="left")
        tk.Label(head, text="  Describe your goal — get the best model",
                 bg=FIELD, fg=INK, font=font(14, "bold")).pack(side="left")
        tk.Label(
            inner,
            text=("e.g.  \"need a denoise model under 20 ms on hailo, prioritise "
                  "quality\"   ·   \"fastest denoiser for the pi 5 cpu\""),
            bg=FIELD, fg=SUBTLE, font=font(9), wraplength=S(600),
            justify="left").pack(anchor="w", pady=(S(4), S(8)))

        self.goal_var = tk.StringVar()
        entry = tk.Entry(inner, textvariable=self.goal_var, font=font(11),
                         bg=WHITE, fg=INK, relief="flat",
                         highlightthickness=1, highlightbackground=LINE,
                         highlightcolor=RASPBERRY, insertbackground=INK)
        entry.pack(fill="x", ipady=S(7))
        entry.bind("<KeyRelease>", lambda _e: self._goal_preview())
        entry.bind("<Return>", lambda _e: self._goal_search())

        self._goal_hint = tk.Label(
            inner, text="Task supported today: denoising.", bg=FIELD, fg=SUBTLE,
            font=font(9), wraplength=S(600), justify="left")
        self._goal_hint.pack(anchor="w", pady=(S(8), 0))

        btnrow = tk.Frame(inner, bg=FIELD)
        btnrow.pack(fill="x", pady=(S(10), 0))
        RoundButton(btnrow, "FIND BEST MODEL", self._goal_search, kind="primary",
                    width=200, height=42).pack(side="right")

    def _build_noise_dataset_card(self, parent):
        """Home card: 5-phase IMX662 noise calibration → PI_RAW dataset builder."""
        card = tk.Frame(parent, bg=FIELD, highlightthickness=1,
                        highlightbackground=LINE, highlightcolor=LINE)
        card.pack(fill="x", pady=(S(4), S(16)))
        inner = tk.Frame(card, bg=FIELD)
        inner.pack(fill="x", padx=S(16), pady=S(14))

        head = tk.Frame(inner, bg=FIELD)
        head.pack(fill="x")
        tk.Label(head, text="DATA", bg=RASPBERRY, fg=WHITE,
                 font=font(8, "bold"), padx=S(6), pady=S(1)).pack(side="left")
        tk.Label(head, text="  Build training dataset from clean images",
                 bg=FIELD, fg=INK, font=font(14, "bold")).pack(side="left")
        tk.Label(
            inner,
            text=("Manager PI_RAW (cabinet_*, colour_stripes, imx219_ag*) is shown in "
                  "Dataset Studio. Add calibration shoots + synthesize imx662_ag24/48 "
                  "night pairs — existing captures are never overwritten."),
            bg=FIELD, fg=SUBTLE, font=font(9), wraplength=S(600),
            justify="left").pack(anchor="w", pady=(S(4), S(10)))

        btn_row = tk.Frame(inner, bg=FIELD)
        btn_row.pack(fill="x")
        RoundButton(btn_row, "CAMERA CAPTURE", self._open_capture_wizard,
                    kind="primary", width=190, height=40).pack(side="left")
        RoundButton(btn_row, "DATASET STUDIO", self._open_data_studio,
                    kind="secondary", width=170, height=40).pack(side="left", padx=(S(8), 0))
        RoundButton(btn_row, "NOISE DATASET WIZARD", self._open_noise_wizard,
                    kind="secondary", width=220, height=40).pack(side="left", padx=(S(8), 0))

    def _open_capture_wizard(self):
        try:
            CttCaptureWizard(self)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera capture", str(exc))

    def _open_data_studio(self):
        try:
            Imx662DataStudio(self)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Studio", str(exc))

    def _open_noise_wizard(self):
        try:
            NoiseDatasetWizard(self)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Noise dataset wizard", str(exc))

    _GOAL_HW = {
        "hailo": "hailo8", "hailo8": "hailo8", "hailo-8": "hailo8",
        "deepx": "deepx", "dx-m1": "deepx", "dxm1": "deepx", "dx m1": "deepx",
        "rpi5_cpu": "rpi5_cpu", "cpu": "rpi5_cpu", "raspberry": "rpi5_cpu",
        "rpi": "rpi5_cpu", "pi 5": "rpi5_cpu", "pi5": "rpi5_cpu",
    }
    _GOAL_HW_NAMES = {"rpi5_cpu": "Raspberry Pi 5 (CPU)",
                      "hailo8": "Pi 5 + Hailo-8", "deepx": "DeepX DX-M1"}

    def _parse_goal(self, text: str) -> dict:
        """Translate a free-text goal into sweep constraints (best-effort)."""
        import re
        t = (text or "").lower()
        out: dict = {"task": "denoise", "hardware": None, "max_latency": None,
                     "prefer": None, "sensor": None, "unsupported": None}

        # Task (only denoise wired up today) --------------------------------
        for kw in ("detect", "segment", "classif", "super-res", "super res",
                   "upscal", "pose", "depth estim"):
            if kw in t:
                out["unsupported"] = kw
                break

        # Target chip -------------------------------------------------------
        for key, hw in self._GOAL_HW.items():
            if key in t:
                out["hardware"] = hw
                break

        # Latency budget: prefer explicit ms, else convert fps -------------
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ms|millisec)", t)
        if m:
            out["max_latency"] = float(m.group(1))
        else:
            f = re.search(r"(\d+(?:\.\d+)?)\s*fps", t)
            if f and float(f.group(1)) > 0:
                out["max_latency"] = round(1000.0 / float(f.group(1)), 1)

        # Preference (quality vs speed) ------------------------------------
        if any(k in t for k in ("quality", "accura", "psnr", "sharp",
                                "best result", "cleanest")):
            out["prefer"] = "quality"
        elif any(k in t for k in ("fast", "speed", "low latency", "low-latency",
                                  "real-time", "realtime", "quick", "lightweight")):
            out["prefer"] = "speed"
        else:
            out["prefer"] = "balanced"

        # Sensor (optional) -------------------------------------------------
        if "imx219" in t:
            out["sensor"] = "imx219"
        elif "imx662" in t or "low light" in t or "lowlight" in t or "dark" in t:
            out["sensor"] = "imx662"
        elif "imxng" in t or "next-gen" in t or "next gen" in t:
            out["sensor"] = "imxng"
        return out

    def _goal_summary(self, g: dict) -> str:
        bits = ["Denoising"]
        if g.get("hardware"):
            bits.append("on " + self._GOAL_HW_NAMES.get(g["hardware"], g["hardware"]))
        if g.get("max_latency"):
            bits.append(f"under {g['max_latency']:.0f} ms")
        pref = g.get("prefer")
        if pref and pref != "balanced":
            bits.append(f"prioritising {pref}")
        if g.get("sensor"):
            bits.append(f"sensor {g['sensor']}")
        return "  ·  ".join(bits)

    def _goal_preview(self):
        if not hasattr(self, "_goal_hint"):
            return
        text = self.goal_var.get().strip()
        if not text:
            self._goal_hint.config(text="Task supported today: denoising.",
                                   fg=SUBTLE)
            return
        g = self._parse_goal(text)
        if g["unsupported"]:
            self._goal_hint.config(
                text=(f"'{g['unsupported']}…' isn't supported yet — only denoising "
                      "for now. I'll search denoisers matching the rest."),
                fg=AMBER)
            return
        self._goal_hint.config(text="Will search:  " + self._goal_summary(g),
                               fg=GREEN)

    def _goal_search(self):
        text = self.goal_var.get().strip()
        if not text:
            self._goal_hint.config(text="Type a goal first, e.g. \"denoise under "
                                        "20 ms on hailo\".", fg=AMBER)
            return
        g = self._parse_goal(text)

        # Apply parsed constraints to the wizard state so the sweep + summary
        # reflect them, then switch to sweep mode.
        if g["hardware"]:
            try:
                self.rows["hardware"].set(g["hardware"])
            except Exception:
                pass
        if g["sensor"]:
            try:
                self.rows["sensor"].set(g["sensor"])
                self._on_sensor_change()
            except Exception:
                pass
        self.eval_var.set("sweep")
        try:
            self._on_eval_change()
        except Exception:
            pass

        cmd = self._build_sweep_command()
        if g["max_latency"]:
            cmd += ["--max-latency", str(g["max_latency"])]
        if g["prefer"]:
            cmd += ["--prefer", g["prefer"]]

        self._save_gui_state()
        note = self._goal_summary(g)
        if g["unsupported"]:
            note = f"(only denoising supported) {note}"
        self._run_command(cmd, "Auto-search…", f"Finding the best model:  {note}")

    def _show_home(self):
        """Quick-run landing: prominent Run, optional full config wizard."""
        self._wizard_mode = "home"
        for st in self._steps:
            st["holder"].pack_forget()
        self._home.pack(fill="both", expand=True)
        self._refresh_home_summary()
        self._render_wiz_header()
        self._render_nav()
        self._save_gui_state()          # remember edits when leaving the wizard
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
        if self.extended_train_var.get():
            ext = (self.extended_steps_var.get() or "1500").strip()
            rows.append(("Extended training",
                         f"full dataset · {ext} steps", GREEN))
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
            self.extended_train_var.set(bool(getattr(cfg.optimization, "extended_train", False)))
            self.extended_steps_var.set(str(getattr(cfg.optimization, "extended_steps", 1500)))
            if hasattr(self, "_on_extended_toggle"):
                self._on_extended_toggle()
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

    # -- Remembered settings (persist wizard choices across launches) ----------
    # Rows whose widgets always exist vs. those created on demand by the
    # model-family / loss selectors (which must be set after re-rendering).
    _STATE_SIMPLE_ROWS = ("hardware", "sensor", "gain", "steps", "frames")
    _STATE_MODEL_ROWS = ("base_channels", "block_depth", "conv_type", "activation")
    _STATE_LOSS_ENTRIES = ("charbonnier_eps", "huber_delta", "ssim_window", "ssim_weight")
    _STATE_OTHER_ENTRIES = ("filter", "noise_std", "batch", "burst",
                            "naf_enc", "naf_mid", "naf_dec")
    _STATE_BOOL_VARS = ("sim_noise_var", "quantize_var", "qat_var", "all_sensors_var",
                        "extended_train_var")
    _STATE_STR_VARS = ("mode_var", "source_var", "eval_var", "extended_steps_var")

    def _gui_state(self) -> dict:
        """Snapshot every user-facing wizard choice into a JSON-safe dict."""
        state: dict = {"rows": {}, "entries": {}, "vars": {}}
        for key in (self._STATE_SIMPLE_ROWS + self._STATE_MODEL_ROWS
                    + ("model_family", "loss")):
            row = self.rows.get(key)
            if row is not None:
                state["rows"][key] = row.get()
        for key in (self._STATE_LOSS_ENTRIES + self._STATE_OTHER_ENTRIES):
            var = self.entries.get(key)
            if var is not None:
                state["entries"][key] = var.get()
        for name in self._STATE_BOOL_VARS:
            var = getattr(self, name, None)
            if var is not None:
                state["vars"][name] = bool(var.get())
        for name in self._STATE_STR_VARS:
            var = getattr(self, name, None)
            if var is not None:
                state["vars"][name] = var.get()
        if getattr(self, "dataset_path", None):
            state["dataset_path"] = self.dataset_path
        return state

    def _apply_gui_state(self, state: dict):
        """Re-apply a saved snapshot, honouring the family/loss render order."""
        if not state:
            return
        try:
            rows = state.get("rows", {})
            entries = state.get("entries", {})
            variables = state.get("vars", {})

            for name in self._STATE_BOOL_VARS:
                if name in variables and getattr(self, name, None) is not None:
                    getattr(self, name).set(bool(variables[name]))
            for name in self._STATE_STR_VARS:
                if name in variables and getattr(self, name, None) is not None:
                    getattr(self, name).set(variables[name])

            for key in self._STATE_SIMPLE_ROWS:
                if key in rows and key in self.rows:
                    self.rows[key].set(rows[key])

            # Model family drives which rows exist — set it, re-render, then fill.
            if "model_family" in rows and "model_family" in self.rows:
                self.rows["model_family"].set(rows["model_family"])
                self._render_model_options()
            for key in self._STATE_MODEL_ROWS:
                if key in rows and key in self.rows:
                    self.rows[key].set(rows[key])

            # Loss name drives which parameter entries exist — same pattern.
            if "loss" in rows and "loss" in self.rows:
                self.rows["loss"].set(rows["loss"])
                self._render_loss_options()

            for key in (self._STATE_LOSS_ENTRIES + self._STATE_OTHER_ENTRIES):
                if key in entries and key in self.entries:
                    self.entries[key].set(entries[key])

            if state.get("dataset_path"):
                self.dataset_path = state["dataset_path"]
                if hasattr(self, "dataset_label"):
                    self.dataset_label.config(text=str(self.dataset_path), fg=SUBTLE)

            # Refresh dependent UI (enabled/disabled fields, hints, summary).
            self._on_source_change()
            self._on_mode_change()
            self._on_sensor_change()
            self._on_eval_change()
            if hasattr(self, "_home_summary"):
                self._refresh_home_summary()
        except Exception:
            pass

    def _load_gui_state(self) -> dict | None:
        try:
            if GUI_STATE_PATH.is_file():
                return json.loads(GUI_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def _save_gui_state(self):
        try:
            GUI_STATE_PATH.write_text(
                json.dumps(self._gui_state(), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        self._save_gui_state()
        try:
            self.destroy()
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
               [16, 32, 64], _choice_int(prev.get("base_channels"), 32))
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
               depth_vals,
               max(_choice_int(prev.get("block_depth"), 4),
                   2 if fam == "rednet" else 0))

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
        self._save_gui_state()          # persist the exact settings being run
        if self.eval_var.get() == "sweep":
            self._run_command(self._build_sweep_command(), "Searching…",
                              "Training & ranking model variants for this target")
        else:
            self._run_command(self._build_command(), "Compiling…",
                              "Running the 6-level optimization stack")

    def _run_again(self):
        """Re-run the exact command from the last run (same parameters)."""
        cmd = getattr(self, "_run_cmd", None)
        if not cmd:
            self._back()                # nothing to repeat — go back to the form
            return
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        sweep = "search.py" in " ".join(str(c) for c in cmd)
        if sweep:
            self._run_command(cmd, "Searching…",
                              "Re-running the same sweep parameters")
        else:
            self._run_command(cmd, "Compiling…",
                              "Re-running with the same parameters")

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
        if self.extended_train_var.get():
            cmd += ["--extended-train"]
            ext = (self.extended_steps_var.get() or "1500").strip()
            cmd += ["--extended-steps", ext if ext.isdigit() else "1500"]
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
        place_window(dlg, 740, 640, master=self, min_w=520, min_h=440)
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

    def _stop_live_sessions(self):
        """Close any in-app live window and kill a previous Pi live.py session."""
        try:
            from nsa.pi_remote import (load_pi_live_settings, should_use_pi_remote,
                                       stop_live_on_pi, stop_local_ssh_session)
            stop_local_ssh_session()
            if should_use_pi_remote(ROOT):
                s = load_pi_live_settings(ROOT)
                stop_live_on_pi(str(s["ssh_host"]), str(s["repo"]))
        except Exception:  # noqa: BLE001
            pass
        lv = getattr(self, "_live_view", None)
        if lv is not None:
            try:
                if lv.winfo_exists():
                    lv._on_close()
            except Exception:  # noqa: BLE001
                pass
        self._live_view = None

    def _live_test(self):
        """Live camera: Pi over SSH from AI server, local LiveView elsewhere."""
        if not (ROOT / "outputs" / "model.pt").exists():
            if not messagebox.askyesno(
                "Live testing",
                "No compiled model checkpoint (outputs/model.pt) was found yet.\n\n"
                "Live testing will rebuild and quick-calibrate a model first "
                "(a few seconds). Continue?"):
                return
        self._stop_live_sessions()
        try:
            from nsa.pi_remote import run_live_on_pi, should_use_pi_remote
            if should_use_pi_remote(ROOT):
                err = run_live_on_pi(ROOT)
                if err is None:
                    messagebox.showinfo(
                        "Live testing",
                        "Started live.py on the Pi's CSI camera over SSH.\n\n"
                        "The RAW | DENOISED window opens on the MONITOR "
                        "ATTACHED TO THE PI. Press q or ESC there to stop.\n\n"
                        "If you click LIVE TEST again, the previous window is "
                        "closed automatically.\n\n"
                        "AI-server SSH log: outputs/pi_live.log")
                    return
                messagebox.showerror("Pi live testing", err)
                return
            src = "opencv" if sys.platform.startswith("win") else "auto"
            self._live_view = LiveView(self, source=src)
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
        RoundButton(foot_top, "RUN AGAIN", self._run_again, kind="secondary",
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
        RoundButton(foot_bot, "NEW RUN", self._back, kind="secondary",
                    width=110, height=40).pack(side="left")
        RoundButton(foot_bot, "HISTORY", self._show_history, kind="secondary",
                    width=100, height=40).pack(side="left", padx=(S(6), 0))
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

            # -- Perceptual metrics (SSIM / LPIPS) --------------------------
            if s.get("ssim_out") is not None or s.get("lpips_out") is not None:
                rowp = tk.Frame(body, bg=WHITE); rowp.pack(fill="x", pady=(0, S(10)))
                ssim_out = s.get("ssim_out")
                ssim_gain = s.get("ssim_gain")
                self._metric_card(
                    rowp, "Structure (SSIM)",
                    f"{ssim_out:.3f}" if isinstance(ssim_out, (int, float)) else "—",
                    accent=GREEN,
                    sub=(f"+{ssim_gain:.3f} vs input" if isinstance(ssim_gain, (int, float))
                         else "1.0 = perfect structure"))
                lpips_out = s.get("lpips_out")
                lpips_gain = s.get("lpips_gain")
                self._metric_card(
                    rowp, "Perceptual (LPIPS)",
                    f"{lpips_out:.3f}" if isinstance(lpips_out, (int, float)) else "—",
                    accent=GREEN,
                    sub=(f"-{lpips_gain:.3f} vs input" if isinstance(lpips_gain, (int, float))
                         else "lower = looks better"))
                self._metric_card(
                    rowp, "Why it matters",
                    "anti-blur",
                    sub="LPIPS/SSIM penalise over-smoothing PSNR misses")

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
            if s.get("extended_train"):
                ext_steps = s.get("extended_steps") or 0
                details.append(("Extended training",
                                f"full dataset  ·  {ext_steps} steps"))
            if m.get("custom_nafnet"):
                details.append(("NAFNet topology",
                                f"enc {m.get('nafnet_enc')} · mid {m.get('nafnet_middle')} "
                                f"· dec {m.get('nafnet_dec')}"))
            for k, v in details:
                r = tk.Frame(body, bg=WHITE); r.pack(fill="x", pady=S(2))
                tk.Label(r, text=k, bg=WHITE, fg=SUBTLE, font=font(10),
                         width=14, anchor="w").pack(side="left")
                vcol = GREEN if k == "Extended training" else INK
                tk.Label(r, text=v, bg=WHITE, fg=vcol, font=font(10, "bold")).pack(side="left")

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
        if PANEL_PATH.exists():
            self._result_img = _load_scaled_photo(PANEL_PATH, S(640))
            if self._result_img is not None:
                tk.Label(body, image=self._result_img, bg=WHITE).pack(anchor="w", pady=(S(4), S(12)))
            else:
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
        if scale_path.exists():
            self._scaling_img = _load_scaled_photo(scale_path, S(640))
            if self._scaling_img is not None:
                tk.Label(body, image=self._scaling_img, bg=WHITE).pack(
                    anchor="w", pady=(S(4), S(12)))
            else:
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
            _wperc = ""
            if winner.get("ssim") is not None:
                _wperc += f"SSIM {winner.get('ssim')}   ·   "
            if winner.get("lpips") is not None:
                _wperc += f"LPIPS {winner.get('lpips')}   ·   "
            tk.Label(body, text=f"PSNR {winner.get('psnr')} dB   ·   " + _wperc +
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
        place_window(dlg, 500, 500, master=self, min_w=420, min_h=400)
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
        ]
        if r.get("ssim") is not None:
            metrics.append(("Sweep SSIM", f"{r.get('ssim')}", INK))
        if r.get("lpips") is not None:
            metrics.append(("Sweep LPIPS", f"{r.get('lpips')}", INK))
        metrics += [
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
        place_window(win, 720, 520, master=self, min_w=520, min_h=400)
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
        place_window(win, 820, 560, master=self, min_w=520, min_h=380)
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
        if rec.get("ssim_out") is not None:
            bits.append(f"SSIM {rec['ssim_out']:.3f}")
        if rec.get("lpips_out") is not None:
            bits.append(f"LPIPS {rec['lpips_out']:.3f}")
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
