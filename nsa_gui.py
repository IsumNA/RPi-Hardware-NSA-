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


# -- High-DPI awareness (crisp text on scaled Windows displays) ----------------
def _detect_scale() -> float:
    """Make the process DPI-aware and return the display scale factor."""
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
        return max(1.0, dpi / 96.0)
    except Exception:
        return 1.0


SCALE = _detect_scale()


def S(x: float) -> int:
    """Scale a pixel dimension for the current display."""
    return int(round(x * SCALE))


def FT(size: float) -> int:
    """Scale a font point size for the current display."""
    return int(round(size * SCALE))


def font(size: float, weight: str = "normal", family: str = "Segoe UI"):
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
        self.title("NSA — Neural Sensor Architecture")
        self.configure(bg=WHITE)
        self.geometry(f"{S(960)}x{S(660)}")
        self.minsize(S(900), S(620))
        try:
            self.tk.call("tk", "scaling", SCALE * 1.0)
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

    # -- Form view ------------------------------------------------------------
    def _build_form(self):
        for w in self.main.winfo_children():
            w.destroy()

        pad = S(34)
        header = tk.Frame(self.main, bg=WHITE)
        header.pack(fill="x", padx=pad, pady=(S(28), S(4)))
        tk.Label(header, text="Compilation Profile", bg=WHITE, fg=INK,
                 font=font(19, "bold")).pack(anchor="w")
        tk.Label(header, text="Select the target hardware and model architecture, "
                 "then compile a hardware-ready denoiser.",
                 bg=WHITE, fg=SUBTLE, font=font(10)).pack(anchor="w", pady=(S(2), 0))
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad, pady=(S(14), S(6)))

        body = tk.Frame(self.main, bg=WHITE)
        body.pack(fill="both", expand=True, padx=pad, pady=(S(6), 0))

        self.rows = {}
        specs = [
            ("hardware", "Target Hardware", "Level 6 export profile",
             ["rpi5_cpu", "hailo8", "deepx"], "hailo8"),
            ("model_family", "Model Family", "Denoiser architecture",
             ["cnn", "unet", "nafnet"], "nafnet"),
            ("base_channels", "Base Channels", "Network width", [16, 32, 64], 32),
            ("block_depth", "Block Depth", "Network depth", [2, 4, 8], 4),
            ("conv_type", "Convolution", "Standard or depthwise-separable",
             ["standard", "depthwise"], "depthwise"),
            ("activation", "Activation", "gelu on DeepX forces QAT injection",
             ["relu", "gelu", "silu"], "relu"),
            ("gain", "Sensor Gain", "IMX662 challenge-frame analog gain",
             [256, 512], 512),
            ("steps", "Calibration", "Live fit iterations (lower = faster)",
             [70, 160, 220], 160),
        ]
        for key, title, desc, values, default in specs:
            row = ConfigRow(body, title, desc, values, default)
            row.pack(fill="x", pady=S(7))
            self.rows[key] = row

        raw_row = tk.Frame(body, bg=WHITE)
        raw_row.pack(fill="x", pady=S(7))
        raw_row.columnconfigure(0, weight=1)
        left = tk.Frame(raw_row, bg=WHITE); left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Input RAW", bg=WHITE, fg=INK, font=font(11, "bold")).pack(anchor="w")
        self.raw_label = tk.Label(left, text="synthetic (auto-generated)", bg=WHITE,
                                  fg=SUBTLE, font=font(9))
        self.raw_label.pack(anchor="w")
        RoundButton(raw_row, "CHOOSE RAW", self._choose_raw, kind="secondary",
                    width=140, height=36).grid(row=0, column=1, sticky="e")

        self._build_footer()

    def _build_footer(self):
        pad = S(34)
        tk.Frame(self.main, bg=LINE, height=1).pack(fill="x", padx=pad)
        footer = tk.Frame(self.main, bg=WHITE)
        footer.pack(fill="x", padx=pad, pady=S(18))
        RoundButton(footer, "APP OPTIONS", self._noop, kind="secondary",
                    width=150, height=44).pack(side="left")
        self.run_btn = RoundButton(footer, "RUN COMPILE", self._run, kind="primary",
                                   width=180, height=44)
        self.run_btn.pack(side="right")

    def _choose_raw(self):
        path = filedialog.askopenfilename(
            title="Select IMX662 Bayer RAW frame",
            filetypes=[("RAW / image", "*.npy *.png *.tif *.tiff *.dng *.raw *.jpg"),
                       ("All files", "*.*")])
        if path:
            self.input_raw = path
            self.raw_label.config(text=Path(path).name)

    def _noop(self):
        pass

    # -- Run view -------------------------------------------------------------
    def _run(self):
        pad = S(34)
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
        for key in ("hardware", "model_family", "base_channels", "block_depth",
                    "conv_type", "activation", "gain", "steps"):
            flag = "--" + key.replace("_", "-")
            cmd += [flag, self.rows[key].get()]
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


if __name__ == "__main__":
    App().mainloop()
