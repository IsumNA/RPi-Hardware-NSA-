#!/usr/bin/env python3
"""NSA Compiler - desktop UI.

A Raspberry Pi Imager-styled front-end for the 6-Level Optimization Stack:
clean white surface, raspberry-red accents, the official Raspberry Pi logo,
a rounded minimal sans typeface, and a live pipeline progress sidebar.

The UI shells out to ``run_demo.py`` so it always runs the exact same pipeline
the CLI does, streaming the live compilation log into the window.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk

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
        self._radio(body, "Temporal Video Denoise", "temporal", enabled=False,
                    badge="COMING SOON",
                    desc="Multi-frame temporal denoising of video sequences.")
        self.batch_var = self._entry_row(
            body, "batch", "Batch Size", "Frames to load in batch mode", "6")

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
        self._coming_soon(body, "Custom NAFNet topology (scaling / encoders)",
                          "Per-level downscale + NAFBlock counts, e.g. scaling "
                          "2 2 1 · encoders 1 2 2 (denoise-hw style).")

        # -- LEVELS 4 & 6: HARDWARE ------------------------------------------
        self._section(body, "LEVELS 4 & 6 · HARDWARE COMPILER TARGET")
        add_rows([
            ("hardware", "Target Hardware", "Pi 5 CPU · Hailo-8 (est.) · DeepX (est.)",
             ["rpi5_cpu", "hailo8", "deepx"], "hailo8"),
        ])
        self._coming_soon(body, "Live Silicon Deployment",
                          "Flash the compiled .hef/.bin to real hardware "
                          "(requires Hailo / DeepX vendor SDK).")

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
        self._coming_soon(body, "Quantization-Aware Training (QAT)",
                          "Fine-tune with simulated INT8 to recover quantization loss.")

        # -- EVALUATION (roadmap) --------------------------------------------
        self._section(body, "EVALUATION")
        self._coming_soon(body, "Automated Pareto Space Sweep / Optuna Loop",
                          "Auto-search the 6-level design space for the optimal "
                          "accuracy / latency trade-off.")
        self._coming_soon(body, "Patch-cache training set builder",
                          "Pre-scan a dataset into detail-scored patches for full "
                          "training runs (denoise-hw dataset.py).")

        self._on_mode_change()
        self._on_source_change()

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
        tk.Label(header, text="Compiling…", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        self.status = tk.Label(header, text="Running the 6-level optimization stack",
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

        if self.mode_var.get() == "batch":
            bs = (self.batch_var.get() or "6").strip()
            cmd += ["--batch", bs if bs.isdigit() else "6"]

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

    def _start_process(self):
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"

        def worker():
            try:
                self.proc = subprocess.Popen(
                    self._build_command(), stdout=subprocess.PIPE,
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
        self.status.config(text="Compilation complete — artifacts written to outputs/")
        self.open_btn.set_enabled(True)
        self._show_result()

    def _show_result(self):
        if not (ImageTk and PANEL_PATH.exists()):
            return
        try:
            im = Image.open(PANEL_PATH)
            target_w = S(900)
            ratio = target_w / im.width
            im = im.resize((int(im.width * ratio), int(im.height * ratio)), Image.LANCZOS)
            self._result_img = ImageTk.PhotoImage(im)
            top = tk.Toplevel(self)
            top.title("NSA — Validation Matrix")
            top.configure(bg=WHITE)
            tk.Label(top, image=self._result_img, bg=WHITE).pack(padx=S(10), pady=S(10))
        except Exception:
            pass

    def _back(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.sidebar.reset()
        self._build_form()

    def _open_outputs(self):
        try:
            os.startfile(str(ROOT / "outputs"))  # type: ignore[attr-defined]
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
