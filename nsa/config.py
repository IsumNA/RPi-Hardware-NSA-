"""Configuration model for the NSA optimization stack.

Loads ``config.yaml``, applies command-line overrides, and validates every
option against the allowed choices for the 6-level stack.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sensors import SENSOR_KEYS

# -- Allowed choices (the "compiler" front-end vocabulary) --------------------
HARDWARE = {
    "rpi5_cpu": "Raspberry Pi 5 (CPU)",
    "hailo8": "Raspberry Pi 5 + Hailo-8",
    "deepx": "DeepX DX-M1",
}
MODEL_FAMILIES = ("cnn", "dncnn", "unet", "rednet", "ridnet", "nafnet",
                  "ffdnet", "drunet", "restormer")
BASE_CHANNELS = (16, 32, 64)
BLOCK_DEPTHS = (2, 4, 8)
CONV_TYPES = ("standard", "depthwise")
ACTIVATIONS = ("relu", "gelu", "silu")
GAINS = (256, 512)


class ConfigError(ValueError):
    """Raised when a configuration value is outside the allowed set."""


@dataclass
class ModelConfig:
    model_family: str = "nafnet"
    base_channels: int = 32
    block_depth: int = 4
    conv_type: str = "depthwise"
    activation: str = "relu"
    # Custom multi-scale NAFNet topology (empty => flat NAFNet of block_depth).
    nafnet_enc_blocks: list = field(default_factory=list)   # e.g. [1, 2, 2]
    nafnet_middle_blocks: int = 1
    nafnet_dec_blocks: list = field(default_factory=list)   # e.g. [2, 2, 1]


@dataclass
class SensorConfig:
    sensor: str = "imx219"          # profile key from nsa.sensors (matches PI_RAW samples)
    input_raw: str | None = None
    real_capture: bool = True       # load real frames from dataset_path by default
    dataset_path: str | None = "datasets/PI_RAW"
    simulate_noise: bool = False    # inject sensor noise on top of loaded frames
    filter: list = field(default_factory=lambda: ["imx219", "ag12"])
    gain: int = 512


@dataclass
class DataConfig:
    temporal_frames: int = 64


@dataclass
class OptimizationConfig:
    quantize: bool = True
    qat: bool = False               # true fake-quant-in-the-loop training
    calibration_steps: int = 220
    patch_size: int = 256


@dataclass
class OutputConfig:
    dir: str = "outputs"
    show_window: bool = True
    seed: int = 662
    export: bool = False        # build a transferable hardware package at the end


@dataclass
class RunConfig:
    mode: str = "single"            # single | batch | temporal
    batch_size: int = 6             # frames processed in batch mode
    burst: int = 8                  # frames in a temporal-denoise burst
    temporal_alpha: float = 0.6     # IIR blend weight for temporal denoise


@dataclass
class Config:
    hardware: str = "hailo8"
    model: ModelConfig = field(default_factory=ModelConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # -- convenience --------------------------------------------------------
    @property
    def hardware_name(self) -> str:
        return HARDWARE[self.hardware]

    @property
    def uses_accelerator(self) -> bool:
        return self.hardware in ("hailo8", "deepx")

    @property
    def artifact_ext(self) -> str:
        return {"hailo8": ".hef", "deepx": ".bin", "rpi5_cpu": ".ort"}[self.hardware]

    def validate(self) -> None:
        m, s = self.model, self.sensor
        checks = [
            (self.hardware in HARDWARE, "hardware", self.hardware, list(HARDWARE)),
            (m.model_family in MODEL_FAMILIES, "model_family", m.model_family, MODEL_FAMILIES),
            (m.base_channels in BASE_CHANNELS, "base_channels", m.base_channels, BASE_CHANNELS),
            (m.block_depth in BLOCK_DEPTHS, "block_depth", m.block_depth, BLOCK_DEPTHS),
            (m.conv_type in CONV_TYPES, "conv_type", m.conv_type, CONV_TYPES),
            (m.activation in ACTIVATIONS, "activation", m.activation, ACTIVATIONS),
            (s.gain in GAINS, "gain", s.gain, GAINS),
            (s.sensor in SENSOR_KEYS, "sensor", s.sensor, SENSOR_KEYS),
        ]
        for ok, name, got, allowed in checks:
            if not ok:
                raise ConfigError(
                    f"Invalid value for '{name}': {got!r}. Allowed: {list(allowed)}"
                )
        # Custom NAFNet topology is only honoured for the nafnet family.
        if m.nafnet_enc_blocks and m.model_family != "nafnet":
            raise ConfigError(
                "Custom NAFNet topology (nafnet_enc_blocks) requires model_family='nafnet'."
            )
        if m.nafnet_enc_blocks:
            if any(int(n) < 1 for n in m.nafnet_enc_blocks):
                raise ConfigError("nafnet_enc_blocks must be positive integers.")
            if m.nafnet_dec_blocks and len(m.nafnet_dec_blocks) != len(m.nafnet_enc_blocks):
                raise ConfigError("nafnet_dec_blocks must match the length of nafnet_enc_blocks.")


def _merge(dc, data: dict) -> None:
    """Assign known dict keys onto a dataclass instance."""
    for key, val in (data or {}).items():
        if hasattr(dc, key):
            setattr(dc, key, val)


def load_config(path: str | Path) -> Config:
    raw = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cfg = Config()
    cfg.hardware = raw.get("hardware", cfg.hardware)
    _merge(cfg.model, raw.get("model", {}))
    _merge(cfg.sensor, raw.get("sensor", {}))
    _merge(cfg.data, raw.get("data", {}))
    _merge(cfg.optimization, raw.get("optimization", {}))
    _merge(cfg.output, raw.get("output", {}))
    _merge(cfg.run, raw.get("run", {}))
    try:
        from nsa.denoise_hw_data import apply_auto_dataset
        project = Path(__file__).resolve().parents[1]
        if cfg.sensor.real_capture:
            apply_auto_dataset(cfg, project)
    except Exception:
        pass
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_demo.py",
        description="NSA 6-Level Optimization Stack - hardware-aware RAW denoiser compiler",
    )
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--hardware", choices=list(HARDWARE))
    p.add_argument("--model-family", dest="model_family", choices=MODEL_FAMILIES)
    p.add_argument("--base-channels", dest="base_channels", type=int, choices=BASE_CHANNELS)
    p.add_argument("--block-depth", dest="block_depth", type=int, choices=BLOCK_DEPTHS)
    p.add_argument("--conv-type", dest="conv_type", choices=CONV_TYPES)
    p.add_argument("--activation", choices=ACTIVATIONS)
    p.add_argument("--nafnet-enc", dest="nafnet_enc", nargs="*", type=int,
                   help="custom NAFNet encoder block counts, e.g. 1 2 2")
    p.add_argument("--nafnet-middle", dest="nafnet_middle", type=int,
                   help="custom NAFNet middle block count")
    p.add_argument("--nafnet-dec", dest="nafnet_dec", nargs="*", type=int,
                   help="custom NAFNet decoder block counts, e.g. 2 2 1")
    p.add_argument("--sensor", choices=list(SENSOR_KEYS),
                   help="image sensor profile (Level 1)")
    p.add_argument("--input-raw", dest="input_raw", help="path to a Bayer RAW frame")
    p.add_argument("--dataset", dest="dataset_path",
                   help="folder/file of real captures (real-capture mode)")
    p.add_argument("--real", dest="real_capture", action="store_true",
                   help="use real captures from --dataset/dataset_path as the noisy input")
    p.add_argument("--simulated", dest="simulated", action="store_true",
                   help="synthesise sensor physics instead of loading real captures")
    p.add_argument("--simulate-noise", dest="simulate_noise", action="store_true",
                   help="inject the selected sensor's noise on top of loaded frames")
    p.add_argument("--filter", dest="filter", nargs="*",
                   help="keyword filter for dataset folders (denoise-hw style, e.g. imx219 ag12)")
    p.add_argument("--batch", dest="batch", type=int,
                   help="batch mode: process up to N frames and average the metrics")
    p.add_argument("--temporal", dest="temporal", action="store_true",
                   help="temporal video-denoise mode (recursive burst denoising)")
    p.add_argument("--burst", dest="burst", type=int,
                   help="frames in a temporal-denoise burst (default 8)")
    p.add_argument("--qat", dest="qat", action="store_true",
                   help="quantization-aware training (fake-quant in the loop)")
    p.add_argument("--gain", type=int, choices=GAINS, help="analog gain of the test frame")
    p.add_argument("--steps", dest="steps", type=int,
                   help="override calibration steps (lower = faster demo)")
    p.add_argument("--frames", dest="frames", type=int,
                   help="temporal frames averaged for the synthetic ground truth")
    p.add_argument("--no-quantize", action="store_true", help="disable the INT8 path")
    p.add_argument("--export", dest="export", action="store_true",
                   help="build a transferable hardware deployment package (.zip) at the end")
    p.add_argument("--no-window", action="store_true", help="do not open the validation window")
    p.add_argument("--seed", type=int)
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.hardware:
        cfg.hardware = args.hardware
    if args.model_family:
        cfg.model.model_family = args.model_family
    if args.base_channels:
        cfg.model.base_channels = args.base_channels
    if args.block_depth:
        cfg.model.block_depth = args.block_depth
    if args.conv_type:
        cfg.model.conv_type = args.conv_type
    if args.activation:
        cfg.model.activation = args.activation
    if getattr(args, "nafnet_enc", None):
        cfg.model.nafnet_enc_blocks = list(args.nafnet_enc)
    if getattr(args, "nafnet_middle", None):
        cfg.model.nafnet_middle_blocks = int(args.nafnet_middle)
    if getattr(args, "nafnet_dec", None):
        cfg.model.nafnet_dec_blocks = list(args.nafnet_dec)
    if args.sensor:
        cfg.sensor.sensor = args.sensor
    if args.input_raw:
        cfg.sensor.input_raw = args.input_raw
    if args.dataset_path:
        cfg.sensor.dataset_path = args.dataset_path
    if getattr(args, "real_capture", False):
        cfg.sensor.real_capture = True
    if getattr(args, "simulated", False):
        cfg.sensor.real_capture = False
    if getattr(args, "simulate_noise", False):
        cfg.sensor.simulate_noise = True
    if getattr(args, "filter", None):
        cfg.sensor.filter = list(args.filter)
    if getattr(args, "batch", None):
        cfg.run.mode = "batch"
        cfg.run.batch_size = max(1, int(args.batch))
    if getattr(args, "temporal", False):
        cfg.run.mode = "temporal"
    if getattr(args, "burst", None):
        cfg.run.burst = max(2, int(args.burst))
    if getattr(args, "qat", False):
        cfg.optimization.qat = True
    if args.gain:
        cfg.sensor.gain = args.gain
    if args.steps:
        cfg.optimization.calibration_steps = args.steps
    if getattr(args, "frames", None):
        cfg.data.temporal_frames = args.frames
    if args.no_quantize:
        cfg.optimization.quantize = False
    if getattr(args, "export", False):
        cfg.output.export = True
    if args.no_window:
        cfg.output.show_window = False
    if args.seed is not None:
        cfg.output.seed = args.seed
    return cfg
