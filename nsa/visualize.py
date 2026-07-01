"""3-panel before / ground-truth / after validation matrix.

Styled to match the Raspberry Pi Imager: clean white surface, raspberry-red
accents, soft grey dividers, the official Raspberry Pi logo and a rounded,
minimal sans typeface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

# -- Raspberry Pi Imager palette ----------------------------------------------
WHITE = "#FFFFFF"
SURFACE = "#FFFFFF"
INK = "#2B2B2B"
SUBTLE = "#8C8C8C"
RASPBERRY = "#C51A4A"
GREEN = "#6CC04A"
LINE = "#E4E4E4"
FIELD = "#F4F4F4"

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "rpi_logo.png"


def _gamma(img: np.ndarray, g: float = 2.2) -> np.ndarray:
    return np.clip(np.clip(img, 0, 1) ** (1.0 / g), 0, 1)


def _pick_font(plt) -> None:
    from matplotlib import font_manager
    for fam in ("Nunito Sans", "Nunito", "Segoe UI", "Verdana", "DejaVu Sans"):
        if any(fam.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break


def render_panel(
    noisy: np.ndarray,
    ground_truth: np.ndarray,
    denoised: np.ndarray,
    meta: dict,
    save_path: Path,
    show: bool = True,
) -> Path:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    _pick_font(plt)
    plt.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": LINE,
    })

    fig = plt.figure(figsize=(14.5, 6.2))
    fig.patch.set_facecolor(WHITE)

    # -- Header band ----------------------------------------------------------
    try:
        logo = mpimg.imread(str(LOGO_PATH))
        ax_logo = fig.add_axes([0.018, 0.875, 0.075, 0.105])
        ax_logo.imshow(logo)
        ax_logo.axis("off")
        tx = 0.085
    except Exception:
        tx = 0.02

    fig.text(tx, 0.94, "NSA", fontsize=21, fontweight="bold", color=RASPBERRY,
             ha="left", va="center")
    fig.text(tx + 0.052, 0.94, "Neural Architecture Search",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(tx, 0.895, "Visual Validation Matrix  ·  6-Level Optimization Stack",
             fontsize=10.5, color=SUBTLE, ha="left", va="center")

    fig.text(0.982, 0.945, meta["hardware_name"], fontsize=12, fontweight="bold",
             color=INK, ha="right", va="center")
    fig.text(0.982, 0.905,
             f"gain Δ  +{meta['psnr_out'] - meta['psnr_in']:.1f} dB   ·   "
             f"{meta['precision']}",
             fontsize=10, color=SUBTLE, ha="right", va="center")

    # thin divider under the header
    fig.add_artist(plt.Line2D([0.018, 0.982], [0.85, 0.85],
                              color=LINE, linewidth=1.2, transform=fig.transFigure))

    # -- Panels ---------------------------------------------------------------
    gs = fig.add_gridspec(1, 3, left=0.018, right=0.982, top=0.78, bottom=0.06,
                          wspace=0.05)
    real = meta.get("real_capture", False)
    kind = meta.get("gt_kind", "temporal")
    src = meta.get("frame_source", "")
    if real and kind in ("paired", "paired+sim") and src:
        raw_sub = f"{meta.get('sensor', 'sensor')}  ·  {src}  (denoise-hw paired)"
    elif real:
        raw_sub = f"{meta.get('sensor', 'sensor')}  ·  real capture"
    else:
        raw_sub = f"{meta.get('sensor', 'sensor')}  ·  {meta['gain']}× gain  ·  synthetic"
    if kind in ("paired", "paired+sim"):
        gt_title, gt_sub = "GROUND TRUTH", "paired gt frame"
    elif kind == "reference":
        gt_title, gt_sub = "REFERENCE", "NL-means reference"
    else:
        gt_title, gt_sub = "GROUND TRUTH", f"temporal avg  ·  {meta['frames']} frames"
    panels = [
        ("RAW INPUT", noisy, raw_sub,
         RASPBERRY, f"PSNR {meta['psnr_in']:.1f} dB", RASPBERRY),
        (gt_title, ground_truth, gt_sub,
         INK, None, None),
        ("MODEL OUTPUT", denoised, f"{meta['family'].upper()}  ·  {meta['precision']}",
         GREEN, f"PSNR {meta['psnr_out']:.1f} dB", GREEN),
    ]
    for i, (title, img, sub, accent, badge, badge_c) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(_gamma(img))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE); spine.set_linewidth(1.2)

        letter = chr(ord("A") + i)
        ax.set_title(f"  {letter}    {title}", color=accent, fontsize=13,
                     fontweight="bold", pad=9, loc="left")
        ax.text(0.5, -0.055, sub, transform=ax.transAxes, fontsize=10,
                color=SUBTLE, ha="center", va="top")
        if badge:
            ax.text(0.035, 0.95, f" {badge} ", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color="white",
                    ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.35", fc=badge_c, ec="none"))

    fig.savefig(save_path, dpi=170, facecolor=WHITE)
    if show:
        try:
            mng = plt.get_current_fig_manager()
            try:
                mng.set_window_title("NSA · Validation Matrix")
            except Exception:
                pass
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return save_path
