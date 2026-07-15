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
    "intel_npu": "Intel AI Boost (NPU / OpenVINO)",
}
MODEL_FAMILIES = ("cnn", "dncnn", "unet", "rednet", "ridnet", "nafnet",
                  "ffdnet", "drunet", "restormer",
                  "attn_unet2", "eamamba", "unifyformer",
                  "remonet", "emvd", "mstmn",
                  "raw_denoiser")
BASE_CHANNELS = (16, 32, 64)
BLOCK_DEPTHS = (2, 4, 8)
CONV_TYPES = ("standard", "depthwise")
ACTIVATIONS = ("relu", "gelu", "silu")
GAINS = (256, 512)

# -- Loss vocabulary ----------------------------------------------------------
# Individual loss "terms" that can be combined into a composite objective by
# joining them with '+', e.g. "l1+perceptual+edge". Each term's default weight
# is what it contributes when several are summed (a lone term is used unscaled).
LOSS_TERMS = ("l1", "l2", "charbonnier", "huber", "ssim", "perceptual", "edge",
              "swt", "swtrel")
DEFAULT_LOSS_WEIGHTS = {
    "l1": 1.0, "l2": 1.0, "charbonnier": 1.0, "huber": 1.0,
    "ssim": 0.2, "perceptual": 0.1, "edge": 0.05,
    # Stationary-wavelet loss: multi-scale subband matching, the anti-blur
    # upgrade over the single-band 'edge' term.
    "swt": 0.1,
    # Normalized SWT: subband error / GT subband energy — zero at identity,
    # maximal under blur, proportional even in dark frames (see _swt_rel_loss).
    "swtrel": 0.5,
}
# Named single-selection presets kept for backwards compatibility / convenience.
# ``charbonnier_ssim`` is a special (1-w)·charbonnier + w·(1-SSIM) blend; the
# rest expand to a term list via LOSS_ALIASES (or are a bare term name).
LOSSES = ("charbonnier", "l1", "l2", "mse", "huber", "ssim", "charbonnier_ssim",
          "l1_perceptual_edge")
LOSS_ALIASES = {"mse": "l2", "l1_perceptual_edge": "l1+perceptual+edge"}


def parse_loss_terms(name: str) -> list:
    """Split a loss ``name`` into its individual terms (expanding aliases)."""
    key = (name or "charbonnier").strip().lower()
    key = LOSS_ALIASES.get(key, key)
    return [t for t in (s.strip() for s in key.split("+")) if t]


def valid_loss(name: str) -> bool:
    """True if ``name`` is a known preset or a '+'-composite of known terms."""
    key = (name or "").strip().lower()
    if key == "charbonnier_ssim":
        return True
    terms = parse_loss_terms(key)
    return bool(terms) and all(t in LOSS_TERMS for t in terms)


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
    # Frozen Hugging Face Hub model to run (see nsa.hf_runner).
    hf_model: str | None = None
    hf_weight: str | None = None   # optional filename inside the snapshot


@dataclass
class SensorConfig:
    sensor: str = "imx219"          # profile key from nsa.sensors (matches PI_RAW samples)
    input_raw: str | None = None
    real_capture: bool = True       # load real frames from dataset_path by default
    dataset_path: str | None = "datasets/PI_RAW"
    simulate_noise: bool = False    # inject sensor noise on top of loaded frames
    filter: list = field(default_factory=lambda: ["imx219", "ag12"])
    gain: int = 256
    noise_std: float | None = None  # override Gaussian read-noise std (electrons RMS); None = sensor default


@dataclass
class DataConfig:
    temporal_frames: int = 64


@dataclass
class LossConfig:
    # Single term (charbonnier | l1 | l2 | huber | ssim | perceptual | edge), a
    # '+'-composite of terms (e.g. "l1+perceptual+edge"), or the charbonnier_ssim
    # preset. Composite terms are summed using ``weights`` (see DEFAULT_LOSS_WEIGHTS).
    name: str = "l1_perceptual_edge"
    charbonnier_eps: float = 1e-3   # Charbonnier L2->L1 transition (smaller = sharper)
    huber_delta: float = 1.0        # Huber/smooth-L1 crossover threshold
    ssim_window: int = 11           # Gaussian window for the SSIM term (odd)
    ssim_weight: float = 0.2        # blend weight for the charbonnier_ssim preset (0..1)
    # Per-term weights for '+'-composite losses; empty means DEFAULT_LOSS_WEIGHTS.
    weights: dict = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    quantize: bool = True
    qat: bool = False               # true fake-quant-in-the-loop training
    calibration_steps: int = 300
    patch_size: int = 256
    loss: LossConfig = field(default_factory=LossConfig)
    # Optional extended training: after the quick calibration, keep training on
    # EVERY paired image in the dataset (PI_RAW) for a much stronger denoiser.
    extended_train: bool = False    # enable the extra full-dataset training pass
    extended_steps: int = 1500      # optimizer steps for the extended pass
    extended_max_side: int = 1024   # legacy resize cap, used only when extended_tile == 0
    # Native-resolution training tiles: cut N random tile×tile squares from each
    # capture instead of resizing (downscaling averages the grain away, so the
    # model under-estimates real noise). 0 = legacy resize-to-max_side.
    extended_tile: int = 512
    extended_tiles_per_image: int = 4
    # Training-sample emphasis: w = gain^gain_emphasis · (1 + dark_emphasis·darkness).
    # Oversamples the hard high-analogue-gain, low-intensity captures instead of
    # sampling every folder uniformly. 0 / 0 restores uniform sampling.
    gain_emphasis: float = 0.5      # exponent on the folder's ag<N> analogue gain
    dark_emphasis: float = 2.0      # extra weight for dark scenes (mean < 0.35)


@dataclass
class OutputConfig:
    dir: str = "outputs"
    show_window: bool = True
    seed: int = 662
    export: bool = False        # build a transferable hardware package at the end
    # Which capture to show in the validation matrix when the dataset spans an
    # analogue-gain sweep (imx662_ag1..ag512). "high" = noisiest (default, the
    # meaningful low-light stress test), "low" = cleanest, "first" = dataset
    # order, or a number (e.g. "512") to prefer the capture closest to that gain.
    validate_gain: str = "high"
    # Detail-restore unsharp mask applied to the DENOISED output (0 = off).
    # Counters the conditional-mean softness of regression denoisers; measured
    # to improve LPIPS at ~zero PSNR cost on held-out ag512 frames.
    sharpen: float = 0.5
    # Auto-copy result images (validation panel, denoised outputs, summary) to a
    # destination you can view — e.g. "you@laptop:~/nsa_results". Empty = off.
    # Must be SSH-reachable FROM this machine; password via NSA_RESULTS_PASS env
    # (needs sshpass) or key-based auth. See nsa/results_sync.py.
    results_dest: str = ""


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
        return self.hardware in ("hailo8", "deepx", "intel_npu")

    @property
    def requires_int8(self) -> bool:
        """Hailo/DeepX are INT8-only; Intel NPU runs FP16 OpenVINO graphs natively."""
        return self.hardware in ("hailo8", "deepx")

    @property
    def artifact_ext(self) -> str:
        return {
            "hailo8": ".hef",
            "deepx": ".bin",
            "rpi5_cpu": ".ort",
            "intel_npu": ".xml",
        }[self.hardware]

    def validate(self) -> None:
        from .model_opts import MIN_BLOCK_DEPTH, normalize_model_config, uses_activation, uses_conv_type

        m, s = self.model, self.sensor
        normalize_model_config(m)
        checks = [
            (self.hardware in HARDWARE, "hardware", self.hardware, list(HARDWARE)),
            (m.model_family in MODEL_FAMILIES, "model_family", m.model_family, MODEL_FAMILIES),
            (m.base_channels in BASE_CHANNELS, "base_channels", m.base_channels, BASE_CHANNELS),
            (m.block_depth in BLOCK_DEPTHS, "block_depth", m.block_depth, BLOCK_DEPTHS),
            (s.gain in GAINS, "gain", s.gain, GAINS),
            (s.sensor in SENSOR_KEYS, "sensor", s.sensor, SENSOR_KEYS),
            (valid_loss(self.optimization.loss.name), "loss", self.optimization.loss.name,
             tuple(LOSSES) + ("<term>+<term>...",)),
        ]
        if uses_conv_type(m.model_family):
            checks.append((m.conv_type in CONV_TYPES, "conv_type", m.conv_type, CONV_TYPES))
        if uses_activation(m.model_family):
            checks.append((m.activation in ACTIVATIONS, "activation", m.activation, ACTIVATIONS))
        min_d = MIN_BLOCK_DEPTH.get(m.model_family)
        if min_d and m.block_depth < min_d:
            raise ConfigError(
                f"block_depth for {m.model_family!r} must be >= {min_d} (got {m.block_depth})."
            )
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


def project_root() -> Path:
    """Repository root (parent of the ``nsa`` package)."""
    return Path(__file__).resolve().parents[1]


def resolve_config_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve ``config.yaml`` against the project root, not the process CWD."""
    root = root or project_root()
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    in_project = root / p
    if in_project.is_file():
        return in_project.resolve()
    if p.is_file():
        return p.resolve()
    return in_project


def resolve_data_path(path: str | Path | None, root: Path | None = None) -> Path | None:
    """Anchor relative data paths to the project root."""
    if not path:
        return None
    root = root or project_root()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _merge(dc, data: dict) -> None:
    """Assign known dict keys onto a dataclass instance."""
    for key, val in (data or {}).items():
        if hasattr(dc, key):
            setattr(dc, key, val)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load ``config.yaml`` only — no dataset auto-detection (see ``finalize_dataset_config``)."""
    root = project_root()
    cfg_path = resolve_config_path(path, root)
    raw = {}
    if cfg_path.is_file():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    cfg = Config()
    cfg.hardware = raw.get("hardware", cfg.hardware)
    _merge(cfg.model, raw.get("model", {}))
    _merge(cfg.sensor, raw.get("sensor", {}))
    _merge(cfg.data, raw.get("data", {}))
    raw_opt = dict(raw.get("optimization", {}) or {})
    raw_loss = raw_opt.pop("loss", None)
    _merge(cfg.optimization, raw_opt)
    if isinstance(raw_loss, dict):
        _merge(cfg.optimization.loss, raw_loss)
    _merge(cfg.output, raw.get("output", {}))
    _merge(cfg.run, raw.get("run", {}))

    if cfg.sensor.dataset_path:
        resolved = resolve_data_path(cfg.sensor.dataset_path, root)
        if resolved is not None:
            cfg.sensor.dataset_path = str(resolved)
    if cfg.sensor.input_raw:
        resolved = resolve_data_path(cfg.sensor.input_raw, root)
        if resolved is not None:
            cfg.sensor.input_raw = str(resolved)
    return cfg


def finalize_dataset_config(cfg: Config, root: Path | None = None) -> bool:
    """After YAML + CLI overrides: resolve dataset path without overriding user intent."""
    from nsa.denoise_hw_data import finalize_dataset_config as _finalize
    return _finalize(cfg, root or project_root())


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
    p.add_argument("--noise-std", dest="noise_std", type=float,
                   help="override the injected Gaussian read-noise std (electrons RMS, denoise-hw style); "
                        "negative restores the sensor default")
    p.add_argument("--steps", dest="steps", type=int,
                   help="override calibration steps (lower = faster demo)")
    p.add_argument("--extended-train", dest="extended_train", action="store_true",
                   help="after quick calibration, keep training on EVERY paired "
                        "image in the dataset for a stronger denoiser")
    p.add_argument("--extended-steps", dest="extended_steps", type=int,
                   help="optimizer steps for the extended full-dataset pass "
                        "(default 1500)")
    p.add_argument("--extended-max-side", dest="extended_max_side", type=int,
                   help="cap image long-side when loading full frames for extended "
                        "training (default 1024)")
    p.add_argument("--extended-tile", dest="extended_tile", type=int,
                   help="native-res training tile size (0 = legacy resize; "
                        "default 512 — preserves real noise statistics)")
    p.add_argument("--extended-tiles", dest="extended_tiles_per_image", type=int,
                   help="random native-res tiles cut per capture (default 4)")
    p.add_argument("--gain-emphasis", dest="gain_emphasis", type=float,
                   help="training-sample weight exponent on analogue gain — "
                        "oversamples high-gain (grainy) captures (default 0.5; "
                        "0 = uniform)")
    p.add_argument("--dark-emphasis", dest="dark_emphasis", type=float,
                   help="extra training-sample weight for low-intensity scenes "
                        "(default 2.0; 0 = off)")
    p.add_argument("--loss", dest="loss",
                   help="training loss: a preset (%s), a single term (%s), or a "
                        "'+'-composite e.g. l1+perceptual+edge (default: "
                        "l1_perceptual_edge)" % (", ".join(LOSSES), ", ".join(LOSS_TERMS)))
    p.add_argument("--charbonnier-eps", dest="charbonnier_eps", type=float,
                   help="Charbonnier epsilon (L2->L1 transition; smaller = sharper)")
    p.add_argument("--huber-delta", dest="huber_delta", type=float,
                   help="Huber/smooth-L1 crossover threshold")
    p.add_argument("--ssim-window", dest="ssim_window", type=int,
                   help="Gaussian window size for the SSIM loss term (odd)")
    p.add_argument("--ssim-weight", dest="ssim_weight", type=float,
                   help="blend weight for the charbonnier_ssim preset (0..1)")
    p.add_argument("--loss-weight", dest="loss_weights", action="append",
                   metavar="TERM=VALUE",
                   help="weight for a term in a '+'-composite loss, e.g. "
                        "--loss-weight perceptual=0.1 (repeatable)")
    p.add_argument("--frames", dest="frames", type=int,
                   help="temporal frames averaged for the synthetic ground truth")
    p.add_argument("--no-quantize", action="store_true", help="disable the INT8 path")
    p.add_argument("--export", dest="export", action="store_true",
                   help="build a transferable hardware deployment package (.zip) at the end")
    p.add_argument("--no-window", action="store_true", help="do not open the validation window")
    p.add_argument("--validate-gain", dest="validate_gain", metavar="HIGH|LOW|FIRST|<int>",
                   help="which analogue-gain capture the validation matrix uses when the "
                        "dataset spans a gain sweep: high=noisiest (default), low=cleanest, "
                        "first=dataset order, or a gain number (e.g. 512)")
    p.add_argument("--sharpen", dest="sharpen", type=float,
                   help="detail-restore unsharp mask on the denoised output "
                        "(default 0.5; 0 = off)")
    p.add_argument("--results-dest", dest="results_dest",
                   help="auto-copy result images to this SSH target after the run, "
                        "e.g. you@laptop:~/nsa_results (password via NSA_RESULTS_PASS)")
    p.add_argument("--seed", type=int)
    p.add_argument("--hf-model", dest="hf_model",
                   help="frozen Hugging Face model id to run (downloads snapshot if needed)")
    p.add_argument("--hf-weight", dest="hf_weight",
                   help="specific weight file inside the Hub snapshot (e.g. NAFNet-SIDD-width64.onnx)")
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
    if getattr(args, "hf_model", None):
        cfg.model.hf_model = args.hf_model
    if getattr(args, "hf_weight", None):
        cfg.model.hf_weight = args.hf_weight
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
    if getattr(args, "noise_std", None) is not None:
        cfg.sensor.noise_std = float(args.noise_std)
    if args.steps:
        cfg.optimization.calibration_steps = args.steps
    if getattr(args, "extended_train", False):
        cfg.optimization.extended_train = True
    if getattr(args, "extended_steps", None):
        cfg.optimization.extended_steps = max(1, int(args.extended_steps))
    if getattr(args, "extended_max_side", None):
        cfg.optimization.extended_max_side = max(64, int(args.extended_max_side))
    if getattr(args, "extended_tile", None) is not None:
        cfg.optimization.extended_tile = max(0, int(args.extended_tile))
    if getattr(args, "extended_tiles_per_image", None):
        cfg.optimization.extended_tiles_per_image = max(1, int(args.extended_tiles_per_image))
    if getattr(args, "gain_emphasis", None) is not None:
        cfg.optimization.gain_emphasis = max(0.0, float(args.gain_emphasis))
    if getattr(args, "dark_emphasis", None) is not None:
        cfg.optimization.dark_emphasis = max(0.0, float(args.dark_emphasis))
    if getattr(args, "loss", None):
        cfg.optimization.loss.name = args.loss
    if getattr(args, "charbonnier_eps", None) is not None:
        cfg.optimization.loss.charbonnier_eps = float(args.charbonnier_eps)
    if getattr(args, "huber_delta", None) is not None:
        cfg.optimization.loss.huber_delta = float(args.huber_delta)
    if getattr(args, "ssim_window", None) is not None:
        cfg.optimization.loss.ssim_window = int(args.ssim_window)
    if getattr(args, "ssim_weight", None) is not None:
        cfg.optimization.loss.ssim_weight = float(args.ssim_weight)
    for spec in (getattr(args, "loss_weights", None) or []):
        term, _, val = str(spec).partition("=")
        term = term.strip().lower()
        if term and val.strip():
            cfg.optimization.loss.weights[term] = float(val)
    if getattr(args, "frames", None):
        cfg.data.temporal_frames = args.frames
    if args.no_quantize:
        cfg.optimization.quantize = False
    if getattr(args, "export", False):
        cfg.output.export = True
    if args.no_window:
        cfg.output.show_window = False
    if getattr(args, "validate_gain", None):
        cfg.output.validate_gain = str(args.validate_gain).strip().lower()
    if getattr(args, "sharpen", None) is not None:
        cfg.output.sharpen = max(0.0, float(args.sharpen))
    if getattr(args, "results_dest", None) is not None:
        cfg.output.results_dest = str(args.results_dest).strip()
    if args.seed is not None:
        cfg.output.seed = args.seed
    return cfg
