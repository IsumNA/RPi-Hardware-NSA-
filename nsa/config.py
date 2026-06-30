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
MODEL_FAMILIES = ("cnn", "unet", "nafnet")
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


@dataclass
class SensorConfig:
    sensor: str = "imx662"          # profile key from nsa.sensors
    input_raw: str | None = None
    real_capture: bool = False      # load real frames instead of synthesising
    dataset_path: str | None = None # folder/file of real captures (e.g. IMX219 repo)
    gain: int = 512


@dataclass
class DataConfig:
    temporal_frames: int = 64


@dataclass
class OptimizationConfig:
    quantize: bool = True
    calibration_steps: int = 220
    patch_size: int = 256


@dataclass
class OutputConfig:
    dir: str = "outputs"
    show_window: bool = True
    seed: int = 662


@dataclass
class Config:
    hardware: str = "hailo8"
    model: ModelConfig = field(default_factory=ModelConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
    p.add_argument("--sensor", choices=list(SENSOR_KEYS),
                   help="image sensor profile (Level 1)")
    p.add_argument("--input-raw", dest="input_raw", help="path to a Bayer RAW frame")
    p.add_argument("--dataset", dest="dataset_path",
                   help="folder/file of real captures (real-capture mode)")
    p.add_argument("--real", dest="real_capture", action="store_true",
                   help="use real captures from --dataset/dataset_path as the noisy input")
    p.add_argument("--gain", type=int, choices=GAINS, help="analog gain of the test frame")
    p.add_argument("--steps", dest="steps", type=int,
                   help="override calibration steps (lower = faster demo)")
    p.add_argument("--no-quantize", action="store_true", help="disable the INT8 path")
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
    if args.sensor:
        cfg.sensor.sensor = args.sensor
    if args.input_raw:
        cfg.sensor.input_raw = args.input_raw
    if args.dataset_path:
        cfg.sensor.dataset_path = args.dataset_path
    if getattr(args, "real_capture", False):
        cfg.sensor.real_capture = True
    if args.gain:
        cfg.sensor.gain = args.gain
    if args.steps:
        cfg.optimization.calibration_steps = args.steps
    if args.no_quantize:
        cfg.optimization.quantize = False
    if args.no_window:
        cfg.output.show_window = False
    if args.seed is not None:
        cfg.output.seed = args.seed
    return cfg
