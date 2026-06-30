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
            return max(1.0, dpi / 96.0)
        except Exception:
            return 1.0
    # Linux / macOS: enlarge by default so the UI is readable out of the box.
    return 1.35 if sys.platform.startswith("linux") else 1.2


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
        if self.kind == "primary":
            fill = RASPBERRY_DK if hover else RASPBERRY
            if not self._enabled:
                fill = "#E2A9B8"
            return fill, fill, "white"
        return (HOVER if hover else WHITE), RASPBERRY, RASPBERRY

    def _draw(self, hover=False):
        self.delete("all")
        fill, border, fg = self._palette(hover)
        r = self.h // 2
        self.create_polygon(_round_points(2, 2, self.w - 2, self.h - 2, r),
                            smooth=True, fill=fill, outline=border, width=1.5)
        self.create_text(self.w / 2, self.h / 2, text=self.text, fill=fg,
                        font=font(11, "bold"))

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
        self.create_text(tx, S(36), text="NSA", anchor="w", fill=RASPBERRY,
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

    def __init__(self, parent, title, desc, values, default):
        super().__init__(parent, bg=WHITE)
        self.columnconfigure(0, weight=1)
        left = tk.Frame(self, bg=WHITE)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text=title, bg=WHITE, fg=INK, font=font(11, "bold")).pack(anchor="w")
        tk.Label(left, text=desc, bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")

        self.var = tk.StringVar(value=str(default))
        self.combo = ttk.Combobox(self, textvariable=self.var,
                                  values=[str(v) for v in values],
                                  state="readonly", width=14, font=font(10),
                                  style="Rpi.TCombobox")
        self.combo.grid(row=0, column=1, sticky="e", padx=(S(8), 0))
        tk.Frame(self, bg=LINE, height=1).grid(row=1, column=0, columnspan=2,
                                               sticky="ew", pady=(S(12), 0))

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(str(value))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        _resolve_font_family()
        self.title("NSA — Neural Sensor Architecture")
        self.configure(bg=WHITE)
        self.geometry(f"{S(980)}x{S(680)}")
        self.minsize(S(900), S(620))
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
                     font=font(9)).pack(anchor="w")

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
                     font=font(9)).pack(anchor="w")
        return cb

    def _entry_row(self, parent, key, title, desc, default=""):
        row = tk.Frame(parent, bg=WHITE)
        row.pack(fill="x", pady=S(6))
        row.columnconfigure(0, weight=1)
        left = tk.Frame(row, bg=WHITE); left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text=title, bg=WHITE, fg=INK, font=font(11, "bold")).pack(anchor="w")
        tk.Label(left, text=desc, bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w")
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(row, textvariable=var, width=18, font=font(10))
        ent.grid(row=0, column=1, sticky="e", padx=(S(8), 0))
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))
        self.entries[key] = var
        return var

    # -- Form view ------------------------------------------------------------
    def _build_form(self):
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()

        pad = S(34)
        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(24), S(4)))
        tk.Label(header, text="Compilation Profile", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(header, text="Configure the 6-level stack, then compile a "
                 "hardware-ready denoiser.",
                 bg=WHITE, fg=SUBTLE, font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(12), 0))

        # footer first so it stays pinned; body fills the space above it.
        self._build_footer()

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(4), 0))
        body = self._make_scrollable(outer)

        self.rows = {}
        self.entries = {}
        self.mode_var = tk.StringVar(value="single")
        self.source_var = tk.StringVar(value="sim")
        self.sim_noise_var = tk.BooleanVar(value=False)
        self.quantize_var = tk.BooleanVar(value=True)
        self.qat_var = tk.BooleanVar(value=False)
        self.eval_var = tk.StringVar(value="single")   # single | sweep
        self.export_var = tk.BooleanVar(value=True)    # build transferable package

        def add_rows(specs):
            for key, title, desc, values, default in specs:
                row = ConfigRow(body, title, desc, values, default)
                row.pack(fill="x", pady=S(6))
                self.rows[key] = row

        # -- MODE -------------------------------------------------------------
        self._section(body, "MODE")
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

        # -- LEVEL 1: CAPTURE SOURCE -----------------------------------------
        self._section(body, "LEVEL 1 · CAPTURE SOURCE")
        self._radio(body, "Simulated capture", "sim", enabled=True,
                    variable=self.source_var,
                    desc="Synthesise a noisy frame from the sensor's noise physics.",
                    command=self._on_source_change)
        self._radio(body, "Real captures", "real", enabled=True,
                    variable=self.source_var, badge="REAL DATA",
                    desc="Load real frames; paired noisy/gt folders are auto-detected.",
                    command=self._on_source_change)

        # Dataset / upload controls (real mode).
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
            "Keyword filter for folders (e.g. imx219 ag12)", "")
        self._check(body, "Simulate sensor noise on loaded frames",
                    "Inject the selected sensor's physics on top of the real frames.",
                    self.sim_noise_var)

        self._section(body, "LEVEL 1 · IMAGE SENSOR PROFILE")
        add_rows([
            ("sensor", "Sensor Profile",
             "imx219 legacy · imx662 Starvis 2 · imxng unreleased",
             ["imx219", "imx662", "imxng"], "imx662"),
            ("gain", "Sensor Gain", "Challenge-frame analog gain", [256, 512], 512),
        ])

        # -- LEVEL 2: GROUND TRUTH -------------------------------------------
        self._section(body, "LEVEL 2 · GROUND TRUTH / DATA")
        add_rows([
            ("frames", "Temporal Frames",
             "Reads averaged for synthetic ground truth", [64, 128, 256], 256),
        ])
        tk.Label(body, text="     Paired noisy/gt folders auto-detected · "
                 "detail-scored patch selection (denoise-hw logic).",
                 bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w", pady=(0, S(4)))

        # -- LEVEL 3: MODEL --------------------------------------------------
        self._section(body, "LEVEL 3 · MODEL ARCHITECTURE")
        tk.Label(body, text="     These are the STARTING parameters for a single "
                 "compile. Turn on the Pareto sweep (below) to train many "
                 "variations and rank them.", bg=WHITE, fg=RASPBERRY,
                 font=font(9, "bold"), wraplength=S(560), justify="left").pack(
                     anchor="w", pady=(0, S(4)))
        add_rows([
            ("model_family", "Model Family", "CNN · U-Net · NAFNet",
             ["cnn", "unet", "nafnet"], "nafnet"),
            ("base_channels", "Base Channels", "Network width", [16, 32, 64], 32),
            ("block_depth", "Block Depth", "Network depth", [2, 4, 8], 4),
            ("conv_type", "Convolution", "Standard or depthwise-separable",
             ["standard", "depthwise"], "depthwise"),
            ("activation", "Activation", "gelu on DeepX forces QAT injection",
             ["relu", "gelu", "silu"], "relu"),
        ])
        tk.Label(body, text="     Custom NAFNet topology (NAFNet family only) — "
                 "leave blank for a flat NAFNet.", bg=WHITE, fg=SUBTLE,
                 font=font(9)).pack(anchor="w", pady=(S(6), 0))
        self.naf_enc_var = self._entry_row(
            body, "naf_enc", "NAFNet Encoders",
            "Per-level encoder block counts, e.g. 1 2 2", "")
        self.naf_mid_var = self._entry_row(
            body, "naf_mid", "NAFNet Middle", "Bottleneck block count, e.g. 4", "")
        self.naf_dec_var = self._entry_row(
            body, "naf_dec", "NAFNet Decoders",
            "Per-level decoder block counts, e.g. 2 2 1", "")

        # -- LEVELS 4 & 6: HARDWARE ------------------------------------------
        self._section(body, "LEVELS 4 & 6 · HARDWARE COMPILER TARGET")
        add_rows([
            ("hardware", "Target Hardware", "Pi 5 CPU · Hailo-8 (est.) · DeepX (est.)",
             ["rpi5_cpu", "hailo8", "deepx"], "hailo8"),
        ])
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
        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(S(8), 0))
        self._check(body, "Compile & export transferable package (.zip)",
                    "End-to-end: after compiling, bundle the device binary + ONNX "
                    "+ manifest + flash steps into a transferable .zip for the target.",
                    self.export_var, command=self._on_eval_change)

        # -- LEVEL 5: CALIBRATION / QUANTIZATION -----------------------------
        self._section(body, "LEVEL 5 · CALIBRATION & QUANTIZATION")
        add_rows([
            ("steps", "Calibration", "Live fit iterations (lower = faster)",
             [70, 160, 220], 160),
        ])
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

        # -- EVALUATION ------------------------------------------------------
        self._section(body, "EVALUATION")
        self._radio(body, "Compile this single model", "single", enabled=True,
                    variable=self.eval_var, badge="TEST ONE",
                    desc="Run the exact config above and produce artifacts + report.",
                    command=self._on_eval_change)
        self._radio(body, "Sweep & rank models (find the best)", "sweep",
                    enabled=True, variable=self.eval_var, badge="COMPARE",
                    desc="Train many variants (families × widths × depths) and show "
                         "a ranked leaderboard + Pareto front. Takes a few minutes; "
                         "your starting conv/activation are kept fixed.",
                    command=self._on_eval_change)
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

        self._on_mode_change()
        self._on_source_change()
        self._on_eval_change()

    def _build_footer(self):
        pad = S(34)
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(16))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        RoundButton(footer, "APP OPTIONS", self._app_options, kind="secondary",
                    width=150, height=44).pack(side="left")
        self.run_btn = RoundButton(footer, "RUN COMPILE", self._run, kind="primary",
                                   width=180, height=44)
        self.run_btn.pack(side="right")

    def _on_source_change(self):
        real = self.source_var.get() == "real"
        for attr in ("dataset_btn", "upload_btn"):
            if hasattr(self, attr):
                getattr(self, attr).set_enabled(real)
        if hasattr(self, "dataset_label"):
            self.dataset_label.config(fg=(SUBTLE if real else "#C4C4C4"))

    def _on_mode_change(self):
        # Batch size only matters in batch mode (kept editable, just a hint).
        pass

    def _on_eval_change(self):
        if not hasattr(self, "run_btn"):
            return
        if self.eval_var.get() == "sweep":
            self.run_btn.set_text("RUN SWEEP")
        elif self.export_var.get():
            self.run_btn.set_text("COMPILE & EXPORT")
        else:
            self.run_btn.set_text("RUN COMPILE")

    def _choose_dataset(self):
        if self.source_var.get() != "real":
            return
        path = filedialog.askdirectory(title="Select a dataset folder (real captures / paired noisy-gt)")
        if path:
            self.dataset_path = path
            self.upload_files = []
            self.dataset_label.config(text=Path(path).name)

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
        self.geometry(f"{S(980)}x{S(680)}")
        self._build_chrome()

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
        win.grab_set()

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
                    "block_depth", "conv_type", "activation", "gain", "steps",
                    "frames"):
            flag = "--" + key.replace("_", "-")
            cmd += [flag, self.rows[key].get()]

        if not self.quantize_var.get():
            cmd += ["--no-quantize"]
        if self.qat_var.get():
            cmd += ["--qat"]
        if self.export_var.get():
            cmd += ["--export"]

        mode = self.mode_var.get()
        if mode == "batch":
            bs = (self.batch_var.get() or "6").strip()
            cmd += ["--batch", bs if bs.isdigit() else "6"]
        elif mode == "temporal":
            cmd += ["--temporal"]
            bu = (self.burst_var.get() or "8").strip()
            cmd += ["--burst", bu if bu.isdigit() else "8"]

        # Custom NAFNet topology (nafnet family only).
        if self.rows["model_family"].get() == "nafnet":
            enc = (self.naf_enc_var.get() or "").split()
            mid = (self.naf_mid_var.get() or "").strip()
            dec = (self.naf_dec_var.get() or "").split()
            if enc and all(t.isdigit() for t in enc):
                cmd += ["--nafnet-enc", *enc]
                if mid.isdigit():
                    cmd += ["--nafnet-middle", mid]
                if dec and all(t.isdigit() for t in dec):
                    cmd += ["--nafnet-dec", *dec]

        if self.source_var.get() == "real":
            cmd += ["--real"]
            dataset = self.dataset_path or self._materialise_uploads()
            if dataset:
                cmd += ["--dataset", dataset]
            if self.sim_noise_var.get():
                cmd += ["--simulate-noise"]
            tokens = (self.filter_var.get() or "").split()
            if tokens:
                cmd += ["--filter", *tokens]

        if self.input_raw:
            cmd += ["--input-raw", self.input_raw]
        return cmd

    def _build_sweep_command(self):
        # Search families × widths × depths; keep the chosen conv/activation fixed
        # (keeps the space bounded so a GUI sweep finishes in a couple of minutes).
        cmd = [sys.executable, str(ROOT / "search.py"),
               "--hardware", self.rows["hardware"].get(),
               "--sensor", self.rows["sensor"].get(),
               "--gain", self.rows["gain"].get(),
               "--conv-type", self.rows["conv_type"].get(),
               "--activation", self.rows["activation"].get(),
               "--search-steps", "30",
               "--patch-size", "128",
               "--top", "8",
               "--no-final-run"]
        if self.source_var.get() == "real":
            dataset = self.dataset_path or self._materialise_uploads()
            if dataset:
                cmd += ["--real", "--dataset", dataset]
            if self.sim_noise_var.get():
                cmd += ["--simulate-noise"]
            tokens = (self.filter_var.get() or "").split()
            if tokens:
                cmd += ["--filter", *tokens]
        return cmd

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
        path = path.parent if path.suffix else path
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            return
        except Exception:
            pass
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
            subtitle = (f"{s.get('hardware_name','')}   ·   "
                        f"{m.get('family','').upper()} "
                        f"{m.get('base_channels','')}ch × {m.get('block_depth','')} · "
                        f"{m.get('conv_type','')} · {m.get('activation','')} · "
                        f"{s.get('precision','')}")
        else:
            subtitle = "Results written to outputs/"
        tk.Label(header, text=subtitle, bg=WHITE, fg=SUBTLE,
                 font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        # -- Footer (pinned) -------------------------------------------------
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(14))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        RoundButton(footer, "RUN AGAIN", self._back, kind="secondary",
                    width=150, height=44).pack(side="left")
        RoundButton(footer, "OPEN OUTPUTS", self._open_outputs, kind="secondary",
                    width=170, height=44).pack(side="left", padx=(S(8), 0))
        RoundButton(footer, "FULL LOG", self._show_log, kind="primary",
                    width=140, height=44).pack(side="right")
        if s and s.get("package_zip"):
            RoundButton(footer, "OPEN PACKAGE",
                        lambda: self._reveal(s.get("package_zip")), kind="primary",
                        width=170, height=44).pack(side="right", padx=(0, S(8)))
        else:
            RoundButton(footer, "EXPORT PACKAGE", self._export_package, kind="primary",
                        width=180, height=44).pack(side="right", padx=(0, S(8)))

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(6), 0))
        body = self._make_scrollable(outer)

        if not s:
            tk.Label(body, text="No summary found. See the full log for details.",
                     bg=WHITE, fg=SUBTLE, font=font(11)).pack(anchor="w", pady=S(10))

        # -- Fitness banner --------------------------------------------------
        if s:
            grade = s.get("grade", "")
            gcol = {"OPTIMAL": GREEN, "BALANCED": "#C98A1B",
                    "SUBOPTIMAL": "#C98A1B", "INFEASIBLE": RASPBERRY}.get(grade, INK)
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

    def _show_ranking(self):
        """Leaderboard of every model the sweep trained (from outputs/pareto.json)."""
        try:
            data = json.loads((ROOT / "outputs" / "pareto.json").read_text(encoding="utf-8"))
        except Exception:
            data = {}
        rows = data.get("all_results", [])
        winner = data.get("winner", {})
        pad = S(34)
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for w in self.main.winfo_children():
            w.destroy()

        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(22), S(2)))
        tk.Label(header, text="Model Ranking", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(header, text=f"{data.get('target_label','')}  ·  "
                 f"{len(rows)} models trained & ranked by Pareto fitness  ·  "
                 f"✦ = Pareto-optimal", bg=WHITE, fg=SUBTLE,
                 font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(10), 0))

        # Footer (pinned).
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(side="bottom", fill="x", padx=pad, pady=S(14))
        tk.Frame(self.main, bg=LINE, height=1).pack(side="bottom", fill="x", padx=pad)
        RoundButton(footer, "BACK", self._back, kind="secondary",
                    width=120, height=44).pack(side="left")
        RoundButton(footer, "OPEN OUTPUTS", self._open_outputs, kind="secondary",
                    width=170, height=44).pack(side="left", padx=(S(8), 0))
        RoundButton(footer, "FULL LOG", self._show_log, kind="secondary",
                    width=130, height=44).pack(side="left", padx=(S(8), 0))
        if winner:
            RoundButton(footer, "USE WINNER", lambda: self._use_winner(winner),
                        kind="primary", width=170, height=44).pack(side="right")

        outer = tk.Frame(self.main, bg=WHITE)
        outer.pack(fill="both", expand=True, padx=pad, pady=(S(6), 0))
        body = self._make_scrollable(outer)

        if not rows:
            tk.Label(body, text="No ranking found. See the full log for details.",
                     bg=WHITE, fg=SUBTLE, font=font(11)).pack(anchor="w", pady=S(10))
            return

        # Column header.
        cols = [("#", 3), ("MODEL", 20), ("PARAMS", 9), ("PSNR", 9),
                ("LATENCY", 10), ("FITNESS", 9), ("GRADE", 12)]
        hrow = tk.Frame(body, bg=WHITE); hrow.pack(fill="x", pady=(S(6), S(2)))
        for label, w in cols:
            tk.Label(hrow, text=label, bg=WHITE, fg=SUBTLE, font=font(8, "bold"),
                     width=w, anchor="w").pack(side="left")
        tk.Frame(body, bg=LINE, height=1).pack(fill="x", pady=(0, S(2)))

        gcol = {"OPTIMAL": GREEN, "BALANCED": "#C98A1B",
                "SUBOPTIMAL": "#C98A1B", "INFEASIBLE": RASPBERRY}
        for i, r in enumerate(rows, 1):
            best = (i == 1)
            bg = FIELD if best else WHITE
            row = tk.Frame(body, bg=bg); row.pack(fill="x", pady=1)
            star = "✦ " if r.get("pareto") else "  "
            model = (f"{r.get('family','').upper()} {r.get('base_channels')}ch×"
                     f"{r.get('block_depth')} {r.get('conv_type','')[:2]}")
            g = r.get("grade", "")
            cells = [
                (f"{i}", 3, INK),
                (star + model, 20, INK if not best else RASPBERRY),
                (f"{r.get('params',0)/1000:.1f}K", 9, SUBTLE),
                (f"{r.get('psnr','—')} dB", 9, INK),
                (f"{r.get('latency_ms','—')} ms", 10, SUBTLE),
                (f"{r.get('fitness','—')}", 9, gcol.get(g, INK)),
                (g, 12, gcol.get(g, INK)),
            ]
            for text, w, fg in cells:
                tk.Label(row, text=text, bg=bg, fg=fg,
                         font=font(9, "bold" if best else "normal"),
                         width=w, anchor="w").pack(side="left")

        # Winner call-out.
        if winner:
            self._section(body, "RECOMMENDED (rank 1)")
            tk.Label(body, text=f"{winner.get('family','').upper()}  "
                     f"{winner.get('base_channels')}ch × depth {winner.get('block_depth')}  ·  "
                     f"{winner.get('conv_type')}  ·  {winner.get('activation')}",
                     bg=WHITE, fg=GREEN, font=font(13, "bold")).pack(anchor="w")
            tk.Label(body, text=f"PSNR {winner.get('psnr')} dB   ·   "
                     f"{winner.get('latency_ms')} ms   ·   "
                     f"fitness {winner.get('fitness')} / 100 ({winner.get('grade')})",
                     bg=WHITE, fg=INK, font=font(10)).pack(anchor="w", pady=(S(2), 0))
            tk.Label(body, text="Click USE WINNER to load this config into the form, "
                     "then RUN COMPILE for the full pipeline + artifacts.",
                     bg=WHITE, fg=SUBTLE, font=font(9)).pack(anchor="w", pady=(S(4), S(10)))

    def _use_winner(self, winner: dict):
        self.sidebar.reset()
        self._build_form()                     # rebuilds self.rows with defaults
        self.eval_var.set("single")
        self._on_eval_change()
        for key in ("model_family", "base_channels", "block_depth",
                    "conv_type", "activation"):
            jkey = "family" if key == "model_family" else key
            val = winner.get(jkey)
            if val is not None and key in self.rows:
                self.rows[key].set(val)

    def _show_log(self):
        win = tk.Toplevel(self)
        win.title("NSA — Full compilation log")
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
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            return
        except Exception:
            pass
        # No file manager (e.g. headless / SSH): at least tell the user where it is.
        try:
            messagebox.showinfo("Outputs folder", f"Results are saved in:\n{path}")
        except Exception:
            pass


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
