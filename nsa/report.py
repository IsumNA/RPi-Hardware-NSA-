"""Pareto fitness scorecard (final output of the stack).

Combines the three competing objectives - image quality, latency, and
quantization robustness - into a single 0-100 Pareto fitness score so the
manager can see at a glance how well a given configuration balanced the
trade-offs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import RPI_GREEN, RPI_RASPBERRY, console

# Fallback memory band (total footprint, KB) used only when no SRAM budget is
# known: lean (<= ~256 KB) -> 1.0, heavy (>= ~8 MB) -> 0.0, log-scaled.
_MEM_LEAN_KB = 256.0
_MEM_HEAVY_KB = 8192.0

# Reference flash budget (KB) for normalising the weight/storage footprint.
_FLASH_REF_KB = 2048.0

# Tie-breaker weight (gamma): memory only moves the score by a few points, so it
# stays quiet while quality/latency dominate and decides otherwise-equal configs.
_W_QUALITY = 0.48
_W_LATENCY = 0.30
_W_ROBUST = 0.18
_W_MEMORY = 0.04


@dataclass
class Fitness:
    psnr: float
    latency_ms: float
    fps: float
    quant_drop_db: float       # negative = quality lost going to INT8
    score: float
    grade: str
    quality_score: float
    latency_score: float
    robust_score: float
    weight_kb: float = 0.0     # storage / flash footprint
    act_kb: float = 0.0        # peak runtime activation (SRAM) footprint
    total_kb: float = 0.0      # weight + activation
    mem_score: float = 1.0     # 1 = lean, 0 = heavy (the tie-breaker)
    sram_budget_kb: float = 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mem_score(total_kb: float) -> float:
    """Map total memory footprint to a [0,1] leanness score (log-scaled)."""
    total_kb = max(total_kb, 1.0)
    hi, lo = math.log2(_MEM_HEAVY_KB), math.log2(_MEM_LEAN_KB)
    return _clamp01((hi - math.log2(total_kb)) / (hi - lo))


def _headroom_score(act_kb: float, weight_kb: float, sram_budget_kb: float) -> float:
    """Leanness as on-chip breathing room: how much SRAM is left for the ISP.

    Dominated by the peak activation footprint vs the SRAM budget (that is what
    competes with the rest of the camera pipeline), with a light weight-footprint
    term so storage size still breaks ties between equal-activation models.
    """
    act_frac = _clamp01(act_kb / max(sram_budget_kb, 1.0))
    weight_frac = _clamp01(weight_kb / _FLASH_REF_KB)
    return _clamp01(1.0 - 0.85 * act_frac - 0.15 * weight_frac)


def compute_fitness(psnr: float, latency_ms: float, quant_drop_db: float,
                    weight_kb: float | None = None, act_kb: float | None = None,
                    sram_budget_kb: float | None = None,
                    target_fps: float = 30.0) -> Fitness:
    fps = 1000.0 / max(latency_ms, 1e-6)

    # Quality band for extreme-gain RAW denoising: 20 dB -> 0, 31 dB -> 1.
    # (Recovering a usable ~28-31 dB frame from a ~14 dB capture is excellent.)
    quality = _clamp01((psnr - 20.0) / (31.0 - 20.0))
    # Latency: at/above target fps -> full marks, degrades below.
    latency_s = _clamp01(fps / target_fps)
    latency_s = latency_s ** 0.5  # diminishing returns past real-time
    # Robustness: 0 dB drop -> 1, -1.5 dB -> 0.
    robust = _clamp01(1.0 - (abs(quant_drop_db) / 1.5))

    if weight_kb is None or act_kb is None:
        # Backward-compatible path: no memory data -> original three-way blend.
        wk = ak = total = 0.0
        mem = 1.0
        score = 100.0 * (0.50 * quality + 0.30 * latency_s + 0.20 * robust)
    else:
        wk, ak = max(weight_kb, 0.0), max(act_kb, 0.0)
        total = wk + ak
        mem = (_headroom_score(ak, wk, sram_budget_kb) if sram_budget_kb
               else _mem_score(total))
        # Fitness = a*PSNR + b*FPS + c*robust + gamma*leanness  (gamma small).
        score = 100.0 * (_W_QUALITY * quality + _W_LATENCY * latency_s
                         + _W_ROBUST * robust + _W_MEMORY * mem)

    score = round(score, 1)
    grade = ("OPTIMAL" if score >= 85 else
             "BALANCED" if score >= 70 else
             "SUBOPTIMAL" if score >= 50 else "INFEASIBLE")
    return Fitness(psnr, latency_ms, fps, quant_drop_db, score,
                   grade, quality, latency_s, robust,
                   weight_kb=wk, act_kb=ak, total_kb=total, mem_score=mem,
                   sram_budget_kb=(sram_budget_kb or 0.0))


def _bar(frac: float, width: int = 22) -> Text:
    filled = int(round(_clamp01(frac) * width))
    colour = RPI_GREEN if frac >= 0.7 else "#E8A33D" if frac >= 0.45 else RPI_RASPBERRY
    t = Text()
    t.append("█" * filled, style=colour)
    t.append("░" * (width - filled), style="#444444")
    return t


def print_report(fit: Fitness, hardware_name: str, profile: str) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="muted", no_wrap=True)
    table.add_column(justify="left")

    table.add_row("Target Optimization Profile",
                  Text(hardware_name, style=f"bold {RPI_GREEN}"))
    table.add_row("Model Profile", Text(profile, style="white"))
    table.add_row("", "")
    table.add_row("Image Quality (PSNR)",
                  Text(f"{fit.psnr:.1f} dB", style="bold white"))
    table.add_row("Target Latency / Frame Rate",
                  Text(f"{fit.latency_ms:.1f} ms  ({fit.fps:.0f} FPS)", style="bold white"))
    table.add_row("Quantization Accuracy Drop",
                  Text(f"{fit.quant_drop_db:+.1f} dB  (FP32 vs INT8)", style="bold white"))
    if fit.total_kb > 0:
        table.add_row("Weight Memory (storage)",
                      Text(f"{fit.weight_kb:,.0f} KB", style="bold white"))
        if fit.sram_budget_kb and fit.sram_budget_kb < 500_000:
            used = 100.0 * fit.act_kb / fit.sram_budget_kb
            act_txt = (f"{fit.act_kb:,.0f} KB  "
                       f"({used:.0f}% of {fit.sram_budget_kb:,.0f} KB SRAM)")
        else:
            act_txt = f"{fit.act_kb:,.0f} KB"
        table.add_row("Activation Memory (peak SRAM)",
                      Text(act_txt, style="bold white"))
    table.add_row("", "")
    table.add_row("Quality", _bar(fit.quality_score))
    table.add_row("Latency", _bar(fit.latency_score))
    table.add_row("INT8 Robustness", _bar(fit.robust_score))
    if fit.total_kb > 0:
        table.add_row("Memory Efficiency", _bar(fit.mem_score))

    grade_colour = {
        "OPTIMAL": RPI_GREEN, "BALANCED": "#E8A33D",
        "SUBOPTIMAL": "#E8A33D", "INFEASIBLE": RPI_RASPBERRY,
    }[fit.grade]
    score_line = Text()
    score_line.append("FINAL PARETO FITNESS SCORE   ", style="bold white")
    score_line.append(f"{fit.score:.1f} / 100", style=f"bold {grade_colour}")
    score_line.append(f"   [{fit.grade}]", style=f"bold {grade_colour}")

    body = Table.grid()
    body.add_column()
    body.add_row(table)
    body.add_row("")
    body.add_row(Align.center(score_line))

    console.print()
    console.print(
        Panel(
            body,
            title="[rpi]PROTOTYPE PERFORMANCE REPORT[/rpi]",
            title_align="left",
            border_style=grade_colour,
            padding=(1, 3),
        )
    )
