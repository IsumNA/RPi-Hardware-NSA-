"""Raspberry Pi branded terminal theme.

Centralises all colours, the logo, and the helper widgets so the whole CLI has
a single, clean, minimalist Raspberry Pi look-and-feel.
"""

from __future__ import annotations

import sys
import time

# Ensure UTF-8 output so the Raspberry Pi glyphs render on legacy Windows
# consoles (cp1252) instead of raising UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

# Make the process DPI-aware on Windows so the matplotlib validation window
# renders crisp (not bitmap-stretched) on high-DPI displays.
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# -- Official Raspberry Pi brand palette --------------------------------------
RPI_RASPBERRY = "#C51A4A"   # primary raspberry red
RPI_GREEN = "#6CC04A"       # leaf green
RPI_DARK = "#2B2B2B"
RPI_GREY = "#8C8C8C"
RPI_LIGHT = "#F3F3F3"

NSA_THEME = Theme(
    {
        "rpi": f"bold {RPI_RASPBERRY}",
        "leaf": f"bold {RPI_GREEN}",
        "muted": RPI_GREY,
        "ok": f"bold {RPI_GREEN}",
        "warn": "bold #E8A33D",
        "err": f"bold {RPI_RASPBERRY}",
        "key": "bold white",
        "val": RPI_GREEN,
        "level": f"bold {RPI_RASPBERRY}",
        "headline": "bold white",
    }
)

console = Console(theme=NSA_THEME, highlight=False)


# -- Raspberry Pi logo (clean two-leaf raspberry) -----------------------------
_LOGO = r"""
            [leaf]..    ..[/leaf]
          [leaf].:'      ':.[/leaf]
          [leaf]:          :[/leaf]
           [leaf]':.    .:'[/leaf]
        [rpi].:::.[/rpi]  [leaf]''[/leaf]  [rpi].:::.[/rpi]
      [rpi]:::::::::[/rpi]  [rpi]:::::::::[/rpi]
      [rpi]':::::::::::::::::::'[/rpi]
        [rpi]':::::::::::::::'[/rpi]
          [rpi]':::::::::::'[/rpi]
            [rpi]':::::::'[/rpi]
               [rpi]'::'[/rpi]
"""


def banner(subtitle: str = "Neural Architecture Search") -> None:
    """Render the top-of-screen Raspberry Pi branded banner."""
    title = Text()
    title.append("NSA", style=f"bold {RPI_RASPBERRY}")
    title.append("  ::  ", style="muted")
    title.append("6-Level Optimization Stack", style="bold white")

    sub = Text(subtitle, style="muted")

    body = Group(
        Align.center(Text.from_markup(_LOGO.strip("\n"))),
        Align.center(title),
        Align.center(sub),
    )
    console.print()
    console.print(
        Panel(
            body,
            border_style=RPI_RASPBERRY,
            padding=(1, 6),
            title="[muted]raspberrypi ~ nsa-compiler[/muted]",
            title_align="left",
            subtitle="[muted]press ^C to abort[/muted]",
            subtitle_align="right",
        )
    )


def level_rule(level: int, name: str) -> None:
    """Print a clean section divider for one stack level."""
    label = Text()
    label.append(f"  LEVEL {level}  ", style=f"on {RPI_RASPBERRY} white")
    label.append(f"  {name}", style="bold white")
    console.print()
    console.print(Rule(label, style=RPI_GREEN, align="left"))


def kv_table(rows: list[tuple[str, str]], title: str | None = None) -> Panel:
    """Render a tidy key/value panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="muted", no_wrap=True)
    table.add_column(style="val")
    for k, v in rows:
        table.add_row(k, v)
    return Panel(
        table,
        border_style=RPI_GREEN,
        title=f"[rpi]{title}[/rpi]" if title else None,
        title_align="left",
        padding=(1, 2),
    )


def log(msg: str, kind: str = "info") -> None:
    """Single styled compiler log line with a glyph + timestamp."""
    glyphs = {
        "info": ("[leaf]•[/leaf]", ""),
        "ok": ("[ok]✓[/ok]", "ok"),
        "warn": ("[warn]▲[/warn]", "warn"),
        "err": ("[err]✗[/err]", "err"),
        "step": ("[rpi]»[/rpi]", "white"),
    }
    glyph, style = glyphs.get(kind, glyphs["info"])
    ts = time.strftime("%H:%M:%S")
    text = f"[muted]{ts}[/muted] {glyph} "
    text += f"[{style}]{msg}[/{style}]" if style else msg
    console.print(text)


def pause(seconds: float) -> None:
    """Small theatrical pause so the live log reads naturally in a demo."""
    time.sleep(max(0.0, seconds))
