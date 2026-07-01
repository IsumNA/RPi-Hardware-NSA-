#!/usr/bin/env python3
"""NSA Architecture Search
==========================
Exhaustively searches the model configuration space for a given target chip
and returns the best-performing framework (model family + architecture flags).

Usage examples
--------------
  # Simulated IMX662 frames, search for best DeepX config:
  python search.py --hardware deepx

  # Real captures from a dataset folder, Hailo-8 target:
  python search.py --hardware hailo8 --real --dataset ./datasets/imx219_raws

  # Constrain to NAFNet variants only on the Pi 5 CPU:
  python search.py --hardware rpi5_cpu --model-family nafnet

  # Faster (fewer calibration steps per candidate):
  python search.py --hardware deepx --search-steps 30

  # Full search then a complete final run on the winner:
  python search.py --hardware hailo8 --sensor imx219 --gain 512
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nsa.config import (
    ACTIVATIONS, BASE_CHANNELS, BLOCK_DEPTHS, CONV_TYPES,
    MODEL_FAMILIES, Config, DataConfig, ModelConfig, OptimizationConfig,
    OutputConfig, RunConfig, SensorConfig,
)
from nsa.compiler import CAPS, assess_targets, compile_stack
from nsa.model_opts import search_combo_valid
from nsa.inference import (
    calibrate_multi, estimate_device_latency_ms, fake_quantize_int8, psnr, run,
)
from nsa.models import build_model, count_params
from nsa.raw_io import build_frame, build_frame_from_source, list_frames
from nsa.report import compute_fitness, print_report
from nsa.sensors import SENSOR_KEYS, get_sensor
from nsa.theme import RPI_GREEN, RPI_RASPBERRY, banner, console as nsa_console
from nsa.visualize import render_panel

console = Console()

# ---------------------------------------------------------------------------
# Colours (match nsa/theme.py palette)
# ---------------------------------------------------------------------------
_GREEN     = RPI_GREEN        # "#6CC04A"
_RED       = RPI_RASPBERRY    # "#C51A4A"
_AMBER     = "#E8A33D"
_MUTED     = "#6B7A9A"
_BRIGHT    = "#DDE3F0"


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    model_family: str
    sensor: str
    base_channels: int
    block_depth: int
    conv_type: str
    activation: str
    psnr_out: float
    latency_ms: float
    quant_drop: float
    fitness: float
    grade: str
    warnings: list[str]
    n_params: int
    duration_s: float
    chips: dict = None      # per-chip suitability {key: {verdict, fps, ...}}


def _build_search_cfg(
    hardware: str,
    sensor_key: str,
    gain: int,
    patch_size: int,
    temporal_frames: int,
    calibration_steps: int,
    real_capture: bool,
    dataset_path: Optional[str],
    simulate_noise: bool,
    filter_tokens: list[str],
    seed: int,
    model_family: str,
    base_channels: int,
    block_depth: int,
    conv_type: str,
    activation: str,
) -> Config:
    cfg = Config(
        hardware=hardware,
        model=ModelConfig(
            model_family=model_family,
            base_channels=base_channels,
            block_depth=block_depth,
            conv_type=conv_type,
            activation=activation,
        ),
        sensor=SensorConfig(
            sensor=sensor_key,
            gain=gain,
            real_capture=real_capture,
            dataset_path=dataset_path,
            simulate_noise=simulate_noise,
            filter=filter_tokens,
        ),
        data=DataConfig(temporal_frames=temporal_frames),
        optimization=OptimizationConfig(
            quantize=True,
            calibration_steps=calibration_steps,
            patch_size=patch_size,
        ),
        output=OutputConfig(dir="outputs", show_window=False, seed=seed),
        run=RunConfig(mode="single"),
    )
    return cfg


def _is_feasible(hardware: str, activation: str, base_channels: int,
                 block_depth: int, patch_size: int) -> tuple[bool, str]:
    """Return (feasible, reason) — skip truly infeasible combos early."""
    caps = CAPS[hardware]
    # GELU on DeepX forces QAT (allowed but noted as a warning, not skipped).
    # We only skip combinations that genuinely cannot produce a useful result.

    # Estimate SRAM — if it's > 3× the budget even with tiling, skip.
    from nsa.compiler import _estimate_sram_kb
    cfg_stub = _build_search_cfg(
        hardware, "imx662", 512, patch_size, 64, 10, False, None, False, [], 1,
        "cnn", base_channels, block_depth, "standard", activation,
    )
    sram = _estimate_sram_kb(cfg_stub)
    if hardware in ("hailo8", "deepx") and sram > caps["sram_kb"] * 3:
        return False, f"SRAM {sram:,.0f} KB >> budget {caps['sram_kb']:,} KB (even with tiling)"
    return True, ""


def _run_candidate(cfg: Config, frames: list) -> SearchResult:
    t0 = time.perf_counter()

    torch.manual_seed(cfg.output.seed)
    np.random.seed(cfg.output.seed)

    model = build_model(cfg.model)
    n_params = count_params(model)

    # Suppress per-candidate compiler terminal chatter — capture silently.
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        result = compile_stack(cfg, n_params)

    pairs = [(f.noisy_rgb, f.clean_rgb) for f in frames]
    calibrate_multi(model, pairs, cfg.optimization.calibration_steps,
                    cfg.output.seed, progress=None)

    psnr_fp32 = float(np.mean(
        [psnr(run(model, f.noisy_rgb)[0], f.clean_rgb) for f in frames]
    ))

    quantized = cfg.uses_accelerator and cfg.optimization.quantize
    if quantized:
        qmodel = fake_quantize_int8(model)
        psnr_int8 = float(np.mean(
            [psnr(run(qmodel, f.noisy_rgb)[0], f.clean_rgb) for f in frames]
        ))
        quant_drop = psnr_int8 - psnr_fp32
        final_psnr = psnr_int8
        export_model = qmodel
    else:
        quant_drop = 0.0
        final_psnr = psnr_fp32
        export_model = model

    latency_ms = estimate_device_latency_ms(
        export_model, cfg.optimization.patch_size, cfg.hardware, quantized
    )
    weight_kb = n_params / 1024.0 if quantized else n_params * 4 / 1024.0
    fit = compute_fitness(final_psnr, latency_ms, quant_drop,
                          weight_kb=weight_kb, act_kb=result.est_sram_kb,
                          sram_budget_kb=result.sram_budget_kb)

    # Cross-chip suitability for this exact trained model (RPi5/Hailo-8/DeepX).
    chips: dict = {}
    try:
        for a in assess_targets(cfg, export_model, cfg.optimization.quantize):
            chips[a.key] = {
                "verdict": a.verdict,
                "fps": round(a.fps, 1),
                "latency_ms": round(a.latency_ms, 1),
                "fits": bool(a.fits),
                "tiled": bool(a.tiled),
                "native": bool(a.act_native),
                "notes": list(a.notes),
            }
    except Exception:
        chips = {}

    return SearchResult(
        model_family=cfg.model.model_family,
        sensor=cfg.sensor.sensor,
        base_channels=cfg.model.base_channels,
        block_depth=cfg.model.block_depth,
        conv_type=cfg.model.conv_type,
        activation=cfg.model.activation,
        psnr_out=final_psnr,
        latency_ms=latency_ms,
        quant_drop=quant_drop,
        fitness=fit.score,
        grade=fit.grade,
        warnings=result.warnings,
        n_params=n_params,
        duration_s=time.perf_counter() - t0,
        chips=chips,
    )


def _pareto_front(results: list[SearchResult]) -> list[SearchResult]:
    """Non-dominated set maximizing PSNR while minimizing latency & params."""
    front = []
    for r in results:
        dominated = False
        for o in results:
            if o is r:
                continue
            better_eq = (o.psnr_out >= r.psnr_out and o.latency_ms <= r.latency_ms
                         and o.n_params <= r.n_params)
            strictly = (o.psnr_out > r.psnr_out or o.latency_ms < r.latency_ms
                        or o.n_params < r.n_params)
            if better_eq and strictly:
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: r.latency_ms)
    return front


def _save_pareto(results: list[SearchResult], front: list[SearchResult],
                 winner: SearchResult, caps: dict, args) -> Path:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    def _row(r: SearchResult) -> dict:
        return {
            "family": r.model_family, "sensor": r.sensor,
            "base_channels": r.base_channels,
            "block_depth": r.block_depth, "conv_type": r.conv_type,
            "activation": r.activation, "params": r.n_params,
            "psnr": round(r.psnr_out, 2), "latency_ms": round(r.latency_ms, 1),
            "fitness": r.fitness, "grade": r.grade,
            "pareto": r in front,
            "chips": r.chips or {},
        }

    payload = {
        "target": args.hardware, "target_label": caps["label"],
        "sensor": "all" if args.all_sensors else args.sensor,
        "all_sensors": bool(args.all_sensors), "gain": args.gain,
        "data_source": "real" if args.real_capture else "simulated",
        "search_steps": args.search_steps,
        "n_evaluated": len(results),
        "winner": _row(winner),
        "pareto_front": [_row(r) for r in front],
        "all_results": [_row(r) for r in sorted(results, key=lambda r: r.fitness, reverse=True)],
    }
    path = out / "pareto.json"
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return path


def _grade_colour(grade: str) -> str:
    return {
        "OPTIMAL": _GREEN, "STRONG": _GREEN,
        "FAIR": _AMBER, "WEAK": _RED,
    }.get(grade, _MUTED)


def _results_table(results: list[SearchResult], title: str = "Search Results",
                   show_sensor: bool = False) -> Table:
    tbl = Table(
        title=title, title_style=f"bold {_BRIGHT}",
        border_style=_MUTED, header_style=f"bold {_MUTED}",
        show_lines=False, pad_edge=True,
    )
    tbl.add_column("#",       style=_MUTED,   width=3,  justify="right")
    if show_sensor:
        tbl.add_column("Sensor", style=_MUTED, width=7)
    tbl.add_column("Family",  style=_BRIGHT,  width=8)
    tbl.add_column("Ch×Dep",  style=_MUTED,   width=7,  justify="right")
    tbl.add_column("Conv",    style=_MUTED,   width=10)
    tbl.add_column("Act",     style=_MUTED,   width=6)
    tbl.add_column("Params",  style=_MUTED,   width=8,  justify="right")
    tbl.add_column("PSNR",    style="bold",   width=8,  justify="right")
    tbl.add_column("Latency", style=_MUTED,   width=10, justify="right")
    tbl.add_column("Fitness", style="bold",   width=9,  justify="right")
    tbl.add_column("Grade",   width=11)

    for i, r in enumerate(results, 1):
        gc = _grade_colour(r.grade)
        warn_marker = " ▲" if r.warnings else ""
        tbl.add_row(
            str(i),
            *(([r.sensor]) if show_sensor else []),
            r.model_family.upper(),
            f"{r.base_channels}×{r.block_depth}",
            r.conv_type,
            r.activation,
            f"{r.n_params/1000:.1f}K",
            f"{r.psnr_out:.1f} dB",
            f"{r.latency_ms:.1f} ms",
            Text(f"{r.fitness:.1f}", style=f"bold {gc}"),
            Text(f"{r.grade}{warn_marker}", style=gc),
        )
    return tbl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search.py",
        description="NSA Architecture Search — finds the best model config for a target chip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Target
    p.add_argument("--hardware", choices=["rpi5_cpu", "hailo8", "deepx"],
                   default="hailo8", help="target accelerator (default: hailo8)")
    # Data source
    p.add_argument("--sensor", choices=["imx219", "imx662", "imxng"],
                   default="imx219", help="sensor noise profile (default: imx219)")
    p.add_argument("--gain", type=int, choices=[256, 512], default=512,
                   help="analog gain of the test frame (default: 512)")
    p.add_argument("--real", dest="real_capture", action="store_true",
                   help="use real captures from --dataset as the noisy input")
    p.add_argument("--simulated", dest="simulated", action="store_true",
                   help="synthesise sensor physics instead of real captures")
    p.add_argument("--dataset", dest="dataset_path", default=None,
                   help="folder or file of real captures (default: datasets/PI_RAW)")
    p.add_argument("--all-sensors", dest="all_sensors", action="store_true",
                   help="sweep across every sensor profile (imx219, imx662, imxng) too")
    p.add_argument("--simulate-noise", dest="simulate_noise", action="store_true",
                   help="inject sensor noise on top of loaded frames (real capture mode)")
    p.add_argument("--filter", nargs="*", default=[],
                   help="keyword filter for dataset folders (e.g. imx219 ag12)")
    # Search space constraints (pin parts of the space to search within)
    p.add_argument("--model-family", dest="model_family", choices=list(MODEL_FAMILIES),
                   default=None, help="restrict search to one model family")
    p.add_argument("--base-channels", dest="base_channels", type=int,
                   choices=[16, 32, 64], default=None,
                   help="restrict to a specific channel width")
    p.add_argument("--block-depth", dest="block_depth", type=int,
                   choices=[2, 4, 8], default=None,
                   help="restrict to a specific block depth")
    p.add_argument("--conv-type", dest="conv_type",
                   choices=["standard", "depthwise"], default=None,
                   help="restrict to a specific conv type")
    p.add_argument("--activation", choices=["relu", "gelu", "silu"], default=None,
                   help="restrict to a specific activation")
    # Search tuning
    p.add_argument("--search-steps", dest="search_steps", type=int, default=60,
                   help="calibration steps per candidate during search (default: 60)")
    p.add_argument("--final-steps", dest="final_steps", type=int, default=220,
                   help="calibration steps for the full final run (default: 220)")
    p.add_argument("--patch-size", dest="patch_size", type=int, default=192,
                   help="working resolution square crop (default: 192)")
    p.add_argument("--temporal-frames", dest="temporal_frames", type=int, default=64,
                   help="frames averaged for synthetic ground truth (default: 64)")
    p.add_argument("--top", type=int, default=5,
                   help="how many top results to show in the summary (default: 5)")
    p.add_argument("--optuna", dest="optuna", type=int, default=0, metavar="TRIALS",
                   help="use an Optuna TPE search of N trials instead of full grid "
                        "(falls back to grid if optuna isn't installed)")
    p.add_argument("--no-final-run", dest="no_final_run", action="store_true",
                   help="skip the full pipeline run on the winner")
    p.add_argument("--seed", type=int, default=662)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    if not args.simulated:
        from nsa.config import Config
        from nsa.denoise_hw_data import apply_auto_dataset
        _cfg = Config()
        if apply_auto_dataset(_cfg, Path(__file__).resolve().parent):
            args.real_capture = True
            args.dataset_path = args.dataset_path or _cfg.sensor.dataset_path
            if not args.filter:
                args.filter = list(_cfg.sensor.filter or [])

    banner("NSA Architecture Search")

    # -- Build search space --------------------------------------------------
    families    = [args.model_family]  if args.model_family  else list(MODEL_FAMILIES)
    channels    = [args.base_channels] if args.base_channels  else list(BASE_CHANNELS)
    depths      = [args.block_depth]   if args.block_depth    else list(BLOCK_DEPTHS)
    conv_types  = [args.conv_type]     if args.conv_type      else list(CONV_TYPES)
    activations = [args.activation]    if args.activation     else list(ACTIVATIONS)

    candidates = list(itertools.product(families, channels, depths, conv_types, activations))

    # Filter infeasible combinations
    feasible = []
    skipped  = []
    for (fam, ch, dep, ct, act) in candidates:
        if not search_combo_valid(fam, ct, act):
            skipped.append(((fam, ch, dep, ct, act), "irrelevant conv/activation for family"))
            continue
        ok, reason = _is_feasible(args.hardware, act, ch, dep, args.patch_size)
        if ok:
            feasible.append((fam, ch, dep, ct, act))
        else:
            skipped.append(((fam, ch, dep, ct, act), reason))

    sensors = list(SENSOR_KEYS) if args.all_sensors else [args.sensor]

    caps = CAPS[args.hardware]
    console.print()
    console.print(f"  [bold {_BRIGHT}]Target chip   :[/] {caps['label']}")
    console.print(f"  [bold {_BRIGHT}]Sensor        :[/] "
                  f"{'ALL profiles (' + ', '.join(sensors) + ')' if args.all_sensors else args.sensor}  @{args.gain}×")
    console.print(f"  [bold {_BRIGHT}]Data source   :[/] {'real captures' if args.real_capture else 'simulated physics'}")
    console.print(f"  [bold {_BRIGHT}]Search space  :[/] {len(feasible)} candidates × {len(sensors)} sensor(s) = {len(feasible) * len(sensors)} runs  ({len(skipped)} infeasible skipped)")
    console.print(f"  [bold {_BRIGHT}]Steps/candidate:[/] {args.search_steps}  (full run: {args.final_steps})")
    console.print()

    if not feasible:
        console.print(f"[bold {_RED}]No feasible configurations for this hardware. Try a smaller patch or fewer channels.[/]")
        return 1

    # -- Frame loader (cached per sensor) -----------------------------------
    _frames_cache: dict = {}

    def load_frames(sensor_key: str):
        if sensor_key in _frames_cache:
            return _frames_cache[sensor_key]
        sensor = get_sensor(sensor_key)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        frames = []
        if args.real_capture:
            sources = list_frames(args.dataset_path, args.filter or [], limit=1)
            for src in sources:
                try:
                    frames.append(build_frame_from_source(
                        src, args.gain, args.temporal_frames, args.patch_size,
                        sensor, args.seed, simulate_noise=args.simulate_noise,
                    ))
                except Exception as exc:
                    console.print(f"  [{_AMBER}]Skipped {src.get('name','?')}: {exc}[/]")
        if not frames:
            if args.real_capture:
                console.print(f"  [{_AMBER}]No usable real frames — falling back to synthetic {sensor_key}[/]")
            frames.append(build_frame(
                input_raw=None, gain=args.gain,
                temporal_frames=args.temporal_frames, patch=args.patch_size,
                sensor=sensor, seed=args.seed,
            ))
        psnr_in = float(np.mean([psnr(f.noisy_rgb, f.clean_rgb) for f in frames]))
        console.print(f"  [{_MUTED}]{sensor_key}: input PSNR {psnr_in:.1f} dB  ·  "
                      f"{len(frames)} frame(s)  ·  {frames[0].width}×{frames[0].height}px[/]")
        _frames_cache[sensor_key] = frames
        return frames

    # -- Search loop ---------------------------------------------------------
    results: list[SearchResult] = []
    n = len(feasible)
    multi_sensor = len(sensors) > 1

    def _evaluate(fam, ch, dep, ct, act, tag, sensor_key, frames):
        cfg = _build_search_cfg(
            hardware=args.hardware, sensor_key=sensor_key, gain=args.gain,
            patch_size=args.patch_size, temporal_frames=args.temporal_frames,
            calibration_steps=args.search_steps, real_capture=args.real_capture,
            dataset_path=args.dataset_path, simulate_noise=args.simulate_noise,
            filter_tokens=args.filter or [], seed=args.seed,
            model_family=fam, base_channels=ch, block_depth=dep,
            conv_type=ct, activation=act,
        )
        spref = f"{sensor_key:<7} " if multi_sensor else ""
        label = f"{spref}{fam.upper()} {ch}ch×{dep} {ct[:3]} {act}"
        console.print(f"  [{_MUTED}]{tag}[/]  {label:<42}", end="")
        try:
            sr = _run_candidate(cfg, frames)
            results.append(sr)
            gc = _grade_colour(sr.grade)
            warn = " ▲" if sr.warnings else "  "
            console.print(
                f"  PSNR [{_BRIGHT}]{sr.psnr_out:>5.1f} dB[/]"
                f"  {sr.latency_ms:>6.1f} ms"
                f"  [{gc}]fit {sr.fitness:>5.1f}  {sr.grade}{warn}[/]"
            )
            return sr
        except Exception as exc:
            console.print(f"  [{_RED}]ERROR: {exc}[/]")
            return None

    # -- Optuna TPE search (optional) ----------------------------------------
    used_optuna = False
    if args.optuna > 0:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            used_optuna = True
        except Exception:
            console.print(f"  [{_AMBER}]optuna not installed — falling back to grid "
                          f"(pip install optuna)[/]\n")

    if used_optuna:
        fam_opts = sorted({c[0] for c in feasible})
        ch_opts = sorted({c[1] for c in feasible})
        dep_opts = sorted({c[2] for c in feasible})
        ct_opts = sorted({c[3] for c in feasible})
        act_opts = sorted({c[4] for c in feasible})
        console.print(f"  [{_MUTED}]Optuna TPE search: {args.optuna} trials...[/]\n")

        def _objective(trial):
            fam = trial.suggest_categorical("family", fam_opts)
            ch = trial.suggest_categorical("base_channels", ch_opts)
            dep = trial.suggest_categorical("block_depth", dep_opts)
            ct = trial.suggest_categorical("conv_type", ct_opts)
            act = trial.suggest_categorical("activation", act_opts)
            sk = trial.suggest_categorical("sensor", sensors) if multi_sensor else sensors[0]
            ok, _ = _is_feasible(args.hardware, act, ch, dep, args.patch_size)
            if not ok:
                raise optuna.TrialPruned()
            sr = _evaluate(fam, ch, dep, ct, act, f"[t{trial.number+1:>3}]",
                           sk, load_frames(sk))
            if sr is None:
                raise optuna.TrialPruned()
            return sr.fitness

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(_objective, n_trials=args.optuna)
    else:
        total = n * len(sensors)
        console.print(f"  [{_MUTED}]Running {total} candidates...[/]\n")
        idx = 0
        for sensor_key in sensors:
            frames = load_frames(sensor_key)
            for (fam, ch, dep, ct, act) in feasible:
                idx += 1
                _evaluate(fam, ch, dep, ct, act, f"[{idx:>3}/{total}]",
                          sensor_key, frames)

    if not results:
        console.print(f"\n[bold {_RED}]All candidates failed.[/]")
        return 1

    # -- Sort and report -----------------------------------------------------
    results.sort(key=lambda r: r.fitness, reverse=True)
    top_n = results[: args.top]

    console.print()
    console.print(_results_table(top_n, f"Top {min(args.top, len(results))} Configurations — {caps['label']}", show_sensor=args.all_sensors))

    winner = results[0]
    console.print()
    console.print(Panel(
        f"  [bold {_BRIGHT}]Best framework for[/] [bold {_GREEN}]{caps['label']}[/]\n\n"
        f"  [bold {_GREEN}]{winner.model_family.upper()}[/]"
        f"  {winner.base_channels}ch × depth {winner.block_depth}"
        f"  ·  {winner.conv_type}  ·  {winner.activation}"
        + (f"  ·  sensor [bold]{winner.sensor}[/]" if args.all_sensors else "")
        + "\n\n"
        f"  PSNR  [bold]{winner.psnr_out:.1f} dB[/]   "
        f"Latency  [bold]{winner.latency_ms:.1f} ms[/]   "
        f"Fitness  [bold {_grade_colour(winner.grade)}]{winner.fitness:.1f} / 100  [{winner.grade}][/]\n"
        + (f"\n  [{_AMBER}]Compiler notes: {'; '.join(winner.warnings)}[/]" if winner.warnings else ""),
        title=f"[bold {_GREEN}]RECOMMENDED CONFIGURATION[/]",
        border_style=_GREEN,
        padding=(0, 2),
    ))

    # -- Pareto front + JSON -------------------------------------------------
    front = _pareto_front(results)
    if len(front) > 1:
        console.print()
        console.print(_results_table(
            sorted(front, key=lambda r: r.latency_ms),
            f"Pareto Front — {len(front)} non-dominated configs (PSNR ↑ / latency ↓ / params ↓)",
            show_sensor=args.all_sensors))
    pareto_path = _save_pareto(results, front, winner, caps, args)
    console.print(f"\n  [{_MUTED}]Saved sweep results -> {pareto_path}[/]")
    try:
        import json as _json
        from nsa.history import record_sweep
        _rec = record_sweep(_json.loads(pareto_path.read_text(encoding="utf-8")),
                            Path("outputs"))
        console.print(f"  [{_MUTED}]Saved to run history -> "
                      f"{Path(_rec['dir']).name}[/]")
    except Exception as _exc:  # noqa: BLE001
        console.print(f"  [{_AMBER}]Could not write run history: {_exc}[/]")

    if skipped:
        console.print(f"\n  [{_MUTED}]{len(skipped)} combinations skipped (infeasible SRAM):[/]")
        for (combo, reason) in skipped[:4]:
            console.print(f"    [{_MUTED}]· {combo[0].upper()} {combo[1]}ch×{combo[2]} — {reason}[/]")
        if len(skipped) > 4:
            console.print(f"    [{_MUTED}]  … and {len(skipped)-4} more[/]")

    # -- Full run on the winner ----------------------------------------------
    if args.no_final_run:
        console.print(f"\n  [{_MUTED}](--no-final-run set — skipping full pipeline run)[/]")
        return 0

    console.print()
    console.print(f"  [{_MUTED}]Running full pipeline on the winner ({args.final_steps} steps)...[/]\n")

    import subprocess
    cmd = [
        sys.executable, "run_demo.py",
        "--hardware",     args.hardware,
        "--model-family", winner.model_family,
        "--base-channels", str(winner.base_channels),
        "--block-depth",   str(winner.block_depth),
        "--conv-type",     winner.conv_type,
        "--activation",    winner.activation,
        "--sensor",        winner.sensor,
        "--gain",          str(args.gain),
        "--steps",         str(args.final_steps),
        "--seed",          str(args.seed),
    ]
    if args.real_capture and args.dataset_path:
        cmd += ["--real", "--dataset", args.dataset_path]
    if args.simulate_noise:
        cmd += ["--simulate-noise"]
    if args.filter:
        cmd += ["--filter"] + args.filter

    # Remove None entries that sneak in from the hardware double-assignment above
    cmd = [str(c) for c in cmd if c is not None]

    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print(f"\n[bold {_RED}]Search aborted.[/]")
        sys.exit(130)
