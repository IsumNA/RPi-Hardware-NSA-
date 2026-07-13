"""Resolution (pixels) vs hardware throughput (TOPS) scaling chart.

Models frame latency as fixed I/O overhead plus compute time at each chip's
peak TOPS, so effective throughput rises with resolution until it plateaus at
the rated peak (small models may never fully saturate the accelerator).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .compiler import CAPS
from .inference import LATENCY_MODEL, model_gflops

# Raspberry Pi Imager palette (match visualize.py).
WHITE = "#FFFFFF"
INK = "#2B2B2B"
SUBTLE = "#8C8C8C"
RASPBERRY = "#C51A4A"
GREEN = "#6CC04A"
LINE = "#E4E4E4"

CHIP_STYLE = {
    "hailo8": {"color": RASPBERRY, "marker": "o"},
    "deepx": {"color": GREEN, "marker": "s"},
    "rpi5_cpu": {"color": "#4A7FC8", "marker": "^"},
    "intel_npu": {"color": "#0071C5", "marker": "D"},
}

# Side lengths to sweep (patch = square side in pixels).
DEFAULT_PATCHES = (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
                   640, 768, 896, 1024, 1280)


def effective_tops(gflops: float, latency_ms: float) -> float:
    """Achieved TOPS = GFLOPs/frame ÷ seconds/frame (1 GFLOP/ms ≡ 1 TOPS)."""
    return gflops / max(latency_ms, 1e-6)


def scaling_latency_ms(gflops: float, hardware: str, peak_tops: float) -> float:
    """Overhead + compute-bound time at peak rated throughput."""
    base_ms, _ = LATENCY_MODEL.get(hardware, (10.0, 1.0))
    if peak_tops <= 0:
        return base_ms + gflops * 20.0
    # peak_tops (TOPS) ≡ GFLOP/ms for FLOP-counted MACs in this chart.
    return base_ms + gflops / peak_tops


def scaling_curves(model, patch_sizes=DEFAULT_PATCHES,
                   hardwares=None) -> dict:
    """Return per-chip curves: pixels[], effective_tops[], peak_tops, util%."""
    hardwares = hardwares or list(CAPS.keys())
    out: dict = {}
    for key in hardwares:
        caps = CAPS[key]
        peak = float(caps.get("tops_peak", 0))
        pixels, tops, ms, gflops_list, util = [], [], [], [], []
        for patch in patch_sizes:
            px = patch * patch
            g = model_gflops(model, patch)
            lat = scaling_latency_ms(g, key, peak)
            et = min(effective_tops(g, lat), peak)
            pixels.append(px)
            gflops_list.append(round(g, 4))
            ms.append(round(lat, 2))
            tops.append(round(et, 3))
            util.append(round(100.0 * et / peak, 2) if peak > 0 else 0.0)
        out[key] = {
            "label": caps["label"],
            "peak_tops": peak,
            "pixels": pixels,
            "effective_tops": tops,
            "utilization_pct": util,
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
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    fig.patch.set_facecolor(WHITE)

    mp_fmt = FuncFormatter(lambda x, _p: f"{x/1e6:.2f}" if x >= 1e6 else f"{x/1e3:.0f}K")

    all_tops: list[float] = []
    for key, c in curves.items():
        sty = CHIP_STYLE.get(key, {"color": INK, "marker": "o"})
        px = np.array(c["pixels"], dtype=float)
        ty = np.array(c["effective_tops"], dtype=float)
        uy = np.array(c["utilization_pct"], dtype=float)
        all_tops.extend(ty.tolist())
        lw = 2.8 if key == selected_hardware else 1.8
        alpha = 1.0 if key == selected_hardware else 0.85
        ax_top.plot(px, ty, color=sty["color"], marker=sty["marker"], lw=lw,
                    ms=4, alpha=alpha, label=c["label"])
        ax_bot.plot(px, uy, color=sty["color"], marker=sty["marker"], lw=lw,
                    ms=4, alpha=alpha)

        peak = c["peak_tops"]
        if peak > 0:
            ax_top.axhline(peak, color=sty["color"], ls="--", lw=1.0, alpha=0.4)
            ax_top.text(px[-1] * 1.002, peak, f" {peak:g} TOPS peak",
                        color=sty["color"], fontsize=8, va="bottom", alpha=0.8)
            ax_bot.axhline(100.0, color=sty["color"], ls="--", lw=1.0, alpha=0.35)

    # Y-axis: fit the data curves, not the peak ratings (avoids flat lines at zero).
    ymax = max(all_tops) * 1.18 if all_tops else 1.0
    peaks = [c["peak_tops"] for c in curves.values() if c["peak_tops"] > 0]
    if peaks and min(peaks) <= ymax:
        ymax = max(ymax, max(peaks) * 1.05)
    ax_top.set_ylim(0, ymax)
    ax_bot.set_ylim(0, min(105, max(
        max(u for c in curves.values() for u in c["utilization_pct"]) * 1.12, 10)))

    if current_patch:
        cpx = current_patch * current_patch
        for ax in (ax_top, ax_bot):
            ax.axvline(cpx, color=SUBTLE, ls=":", lw=1.4, alpha=0.9)
        ax_top.text(cpx, ymax * 0.97, f"  compile @ {current_patch}² px",
                    color=SUBTLE, fontsize=8, va="top", rotation=90)

    ax_top.set_ylabel("Effective throughput (TOPS)", color=INK, fontsize=11)
    ax_bot.set_ylabel("% of peak TOPS", color=INK, fontsize=10)
    ax_bot.set_xlabel("Input resolution (pixels = width × height)", color=INK, fontsize=11)
    ax_bot.xaxis.set_major_formatter(mp_fmt)
    ax_top.set_title("Resolution scaling vs hardware performance",
                     color=INK, fontsize=14, fontweight="bold", pad=12)
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(WHITE)
        ax.grid(True, color=LINE, linewidth=0.8, alpha=0.8)
        ax.tick_params(colors=INK)
        for spine in ax.spines.values():
            spine.set_color(LINE)
    ax_top.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=LINE,
                  fontsize=9)
    fig.text(0.12, 0.01,
             "Latency = I/O overhead + GFLOPs ÷ peak TOPS  ·  "
             "small models may not reach 100% — top panel is absolute TOPS, "
             "bottom is % of chip peak",
             color=SUBTLE, fontsize=8.5)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.93, bottom=0.1, hspace=0.12)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=WHITE)
    if show:
        plt.show()
    plt.close(fig)
    return save_path
