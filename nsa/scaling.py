"""Resolution (pixels) vs hardware throughput (TOPS) scaling chart.

Uses the same GFLOP + latency model as ``inference.estimate_device_latency_ms``
to show how effective on-device throughput rises with input resolution until it
plateaus at each chip's peak TOPS rating.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .compiler import CAPS
from .inference import estimate_device_latency_ms, model_gflops

# Raspberry Pi Imager palette (match visualize.py).
WHITE = "#FFFFFF"
INK = "#2B2B2B"
SUBTLE = "#8C8C8C"
RASPBERRY = "#C51A4A"
GREEN = "#6CC04A"
LINE = "#E4E4E4"
AMBER = "#C98A1B"

CHIP_STYLE = {
    "hailo8": {"color": RASPBERRY, "marker": "o"},
    "deepx": {"color": GREEN, "marker": "s"},
    "rpi5_cpu": {"color": "#4A7FC8", "marker": "^"},
}

# Side lengths to sweep (patch = square side in pixels).
DEFAULT_PATCHES = (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)


def effective_tops(gflops: float, latency_ms: float) -> float:
    """Achieved TOPS = GFLOPs/frame ÷ seconds/frame."""
    return gflops / max(latency_ms, 1e-6)


def scaling_curves(model, patch_sizes=DEFAULT_PATCHES,
                   hardwares=None) -> dict:
    """Return per-chip curves: pixels[], effective_tops[], peak_tops."""
    hardwares = hardwares or list(CAPS.keys())
    out: dict = {}
    for key in hardwares:
        caps = CAPS[key]
        peak = float(caps.get("tops_peak", 0))
        quant = bool(caps.get("needs_quant", False))
        pixels, tops, ms, gflops_list = [], [], [], []
        for patch in patch_sizes:
            px = patch * patch
            g = model_gflops(model, patch)
            lat = estimate_device_latency_ms(model, patch, key, quant)
            pixels.append(px)
            gflops_list.append(round(g, 4))
            ms.append(round(lat, 2))
            tops.append(round(min(effective_tops(g, lat), peak * 1.02), 3))
        out[key] = {
            "label": caps["label"],
            "peak_tops": peak,
            "pixels": pixels,
            "effective_tops": tops,
            "latency_ms": ms,
            "gflops": gflops_list,
        }
    return out


def render_scaling_chart(
    model,
    save_path: Path,
    *,
    current_patch: int | None = None,
    selected_hardware: str | None = None,
    patch_sizes=DEFAULT_PATCHES,
    show: bool = False,
) -> Path:
    """Plot resolution (pixels) vs effective TOPS for every Pi-class target."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    curves = scaling_curves(model, patch_sizes)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    mp_fmt = FuncFormatter(lambda x, _p: f"{x/1e6:.2f}" if x >= 1e6 else f"{x/1e3:.0f}K")

    for key, c in curves.items():
        sty = CHIP_STYLE.get(key, {"color": INK, "marker": "o"})
        px = np.array(c["pixels"], dtype=float)
        ty = np.array(c["effective_tops"], dtype=float)
        lw = 2.8 if key == selected_hardware else 1.8
        alpha = 1.0 if key == selected_hardware else 0.82
        ax.plot(px, ty, color=sty["color"], marker=sty["marker"], lw=lw,
                ms=5, alpha=alpha, label=c["label"])
        if c["peak_tops"] > 0:
            ax.axhline(c["peak_tops"], color=sty["color"], ls="--", lw=1.0,
                       alpha=0.35)
            ax.text(px[-1] * 1.002, c["peak_tops"],
                    f" {c['peak_tops']:.2f} TOPS peak", color=sty["color"],
                    fontsize=8, va="bottom", alpha=0.75)

    if current_patch:
        cpx = current_patch * current_patch
        ax.axvline(cpx, color=SUBTLE, ls=":", lw=1.4, alpha=0.9)
        ax.text(cpx, ax.get_ylim()[1] * 0.97, f"  compile @ {current_patch}² px",
                color=SUBTLE, fontsize=8, va="top", rotation=90)

    ax.set_xlabel("Input resolution (pixels = width × height)", color=INK, fontsize=11)
    ax.set_ylabel("Effective throughput (TOPS)", color=INK, fontsize=11)
    ax.xaxis.set_major_formatter(mp_fmt)
    ax.set_title("Resolution scaling vs hardware performance",
                 color=INK, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, color=LINE, linewidth=0.8, alpha=0.8)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=LINE,
              fontsize=9)
    fig.text(0.12, 0.02,
             "Effective TOPS = model GFLOPs ÷ frame latency  ·  "
             "dashed = chip peak  ·  curves use the compiler latency model",
             color=SUBTLE, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=WHITE)
    if show:
        plt.show()
    plt.close(fig)
    return save_path
