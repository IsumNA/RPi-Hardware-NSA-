"""Pareto fitness scorecard (final output of the stack).

Combines the three competing objectives - image quality, latency, and
quantization robustness - into a single 0-100 Pareto fitness score so the
manager can see at a glance how well a given configuration balanced the
trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import RPI_GREEN, RPI_RASPBERRY, console


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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_fitness(psnr: float, latency_ms: float, quant_drop_db: float,
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

    score = 100.0 * (0.50 * quality + 0.30 * latency_s + 0.20 * robust)
    score = round(score, 1)
    grade = ("OPTIMAL" if score >= 85 else
             "BALANCED" if score >= 70 else
             "SUBOPTIMAL" if score >= 50 else "INFEASIBLE")
    return Fitness(psnr, latency_ms, fps, quant_drop_db, score,
                   grade, quality, latency_s, robust)


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
    table.add_row("", "")
    table.add_row("Quality", _bar(fit.quality_score))
    table.add_row("Latency", _bar(fit.latency_score))
    table.add_row("INT8 Robustness", _bar(fit.robust_score))

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
