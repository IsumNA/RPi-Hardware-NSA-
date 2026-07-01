"""Hardware-aware compiler front-end (Levels 4-6).

Evaluates the chosen architecture against the constraints of the target
accelerator and emits a live compilation log: operator legalization, memory
budgeting, quantization scheme selection (PTQ vs forced QAT), and the final
export format. This is what makes the demo feel like a real toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .theme import log, pause

# -- Per-target capability table ----------------------------------------------
CAPS = {
    "rpi5_cpu": {
        "label": "Raspberry Pi 5 (Cortex-A76 NEON)",
        "precision": "fp16",
        "native_acts": {"relu", "gelu", "silu"},
        "needs_quant": False,
        "sram_kb": 1_000_000,   # main memory, effectively unlimited
        "format": "ONNX Runtime (.ort)",
        "tops_peak": 0.04,      # ~40 GFLOPS FP16 class (shown on TOPS chart)
    },
    "hailo8": {
        "label": "Hailo-8 (26 TOPS, Dataflow)",
        "precision": "int8",
        "native_acts": {"relu"},
        "needs_quant": True,
        "sram_kb": 20_000,      # on-chip SRAM budget
        "format": "Hailo Executable Format (.hef)",
        "tops_peak": 26.0,
    },
    "deepx": {
        "label": "DeepX DX-M1 (25 TOPS NPU)",
        "precision": "int8",
        "native_acts": {"relu", "silu"},
        "needs_quant": True,
        "sram_kb": 16_000,
        "format": "DeepX Runtime Binary (.bin)",
        "tops_peak": 25.0,
    },
}


# Families whose graph contains ConvTranspose (needs resize+conv on NPUs).
_CONVT_FAMILIES = {"unet", "rednet", "drunet"}

# Families with NPU-awkward ops (LayerNorm / softmax attention) that really want
# floating-point execution — used to add caveats in the suitability matrix.
_TRANSFORMER_FAMILIES = {"restormer"}


@dataclass
class CompileResult:
    precision: str = "fp16"
    quantize: bool = False
    quant_scheme: str = "none"          # none | PTQ | QAT
    est_sram_kb: float = 0.0
    sram_budget_kb: float = 0.0
    tiled: bool = False
    warnings: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    target_label: str = ""
    export_format: str = ""

    @property
    def fits(self) -> bool:
        return self.tiled or self.est_sram_kb <= self.sram_budget_kb


def _estimate_sram_kb(cfg: Config, bytes_per: int | None = None) -> float:
    """Rough activation+weight footprint for the working patch, in KB.

    ``bytes_per`` overrides the per-element size (1 = INT8, 2 = FP16); when None
    it follows the configured target. This lets us re-cost the same model for a
    different chip's precision when assessing cross-target suitability.
    """
    m = cfg.optimization
    mc = cfg.model
    px = m.patch_size * m.patch_size
    act = px * mc.base_channels * (mc.block_depth + 2)        # activation tensors
    weights = (mc.base_channels ** 2) * 9 * mc.block_depth     # 3x3 conv weights
    depthwise_factor = 0.45 if mc.conv_type == "depthwise" else 1.0
    if bytes_per is None:
        bytes_per = 1 if cfg.uses_accelerator else 2          # int8 vs fp16
    return (act * bytes_per + weights * depthwise_factor * bytes_per) / 1024.0


def compile_stack(cfg: Config, n_params: int) -> CompileResult:
    """Walk the compiler passes for the selected target, logging as we go."""
    caps = CAPS[cfg.hardware]
    res = CompileResult(
        precision=caps["precision"],
        sram_budget_kb=caps["sram_kb"],
        target_label=caps["label"],
        export_format=caps["format"],
    )

    act = cfg.model.activation
    fam = cfg.model.model_family

    # -- Pass 1: operator legalization ----------------------------------------
    log(f"Parsing graph: {fam.upper()} | {n_params/1e3:.1f}K params | act={act}", "step")
    pause(0.25)
    log(f"Target backend resolved -> {caps['label']} [{caps['precision'].upper()}]", "info")
    pause(0.2)

    if act in caps["native_acts"]:
        log(f"Operator legalization: '{act}' supported natively", "ok")
        res.passes.append(f"legalize:{act}->native")
    else:
        if cfg.hardware == "deepx" and act == "gelu":
            log("GELU activation detected for DeepX target. "
                "Forcing QAT layer injection to prevent compilation failure...", "warn")
            res.warnings.append("DeepX cannot fuse FP GELU -> QAT injection forced.")
            res.quant_scheme = "QAT"
            res.passes.append("inject:QAT(gelu)")
        elif cfg.hardware == "hailo8":
            log(f"'{act}' not in Hailo-8 native set -> substituting "
                f"piecewise-linear approximation + QAT", "warn")
            res.warnings.append(f"Hailo-8: '{act}' approximated (PWL) under QAT.")
            res.quant_scheme = "QAT"
            res.passes.append(f"approx:{act}->pwl")
        else:
            log(f"'{act}' lowered to supported primitive set", "info")
            res.passes.append(f"lower:{act}")
    pause(0.25)

    # -- Pass 2: conv / structure legalization --------------------------------
    if cfg.model.conv_type == "depthwise":
        log("Depthwise-separable convs mapped to grouped-conv engine", "ok")
        res.passes.append("map:depthwise->grouped")
    else:
        log("Standard convs scheduled on MAC array", "info")
        res.passes.append("schedule:conv->mac")
    if fam in _CONVT_FAMILIES:
        if cfg.hardware in ("hailo8", "deepx"):
            log(f"ConvTranspose ({fam.upper()} upsample) rewritten as resize+conv "
                "for NPU compatibility", "warn")
            res.warnings.append(f"{fam.upper()} ConvTranspose rewritten to resize+conv.")
            res.passes.append("rewrite:convT->resize+conv")
        else:
            res.passes.append("keep:convT")
    pause(0.25)

    # -- Pass 3: memory budgeting ---------------------------------------------
    res.est_sram_kb = _estimate_sram_kb(cfg)
    log(f"Activation memory estimate: {res.est_sram_kb:,.0f} KB "
        f"(budget {res.sram_budget_kb:,.0f} KB)", "step")
    pause(0.2)
    if cfg.uses_accelerator and res.est_sram_kb > res.sram_budget_kb:
        res.tiled = True
        log("Working set exceeds on-chip SRAM -> enabling spatial tiling "
            "(2x2) to fit", "warn")
        res.warnings.append("Spatial tiling enabled to fit on-chip SRAM.")
        res.passes.append("tile:2x2")
    else:
        log("Working set fits target memory budget", "ok")
    pause(0.25)

    # -- Pass 4: quantization scheme ------------------------------------------
    if cfg.uses_accelerator and cfg.optimization.quantize:
        res.quantize = True
        if cfg.optimization.qat and res.quant_scheme != "QAT":
            res.quant_scheme = "QAT"
            log("QAT requested -> inserting fake-quant nodes for "
                "quantization-aware training", "info")
            res.passes.append("quant:QAT-int8(user)")
        if res.quant_scheme != "QAT":
            res.quant_scheme = "PTQ"
            log("Selecting INT8 post-training quantization (PTQ) with "
                "per-channel scales", "info")
            res.passes.append("quant:PTQ-int8")
        else:
            log("QAT path active -> fake-quant nodes inserted, calibrating "
                "INT8 scales", "info")
            res.passes.append("quant:QAT-int8")
        log(f"Calibrating on the live IMX662 frame "
            f"({cfg.optimization.calibration_steps} steps)...", "step")
    elif cfg.uses_accelerator and not cfg.optimization.quantize:
        log("Quantization disabled by flag -> target requires INT8; "
            "results may not fit/run on device", "warn")
        res.warnings.append("INT8 disabled though target is an INT8-only NPU.")
    else:
        log(f"CPU target keeps {caps['precision'].upper()} precision "
            "(no quantization required)", "ok")
    pause(0.25)

    # -- Pass 5: export format ------------------------------------------------
    log(f"Export profile locked -> {caps['format']}", "ok")
    res.passes.append(f"export:{caps['format']}")
    pause(0.2)
    return res


# -- Cross-target suitability --------------------------------------------------
@dataclass
class TargetAssessment:
    """Whether the chosen model is suitable to run on a given chip."""
    key: str
    label: str
    precision: str
    format: str
    act_kb: float
    budget_kb: float
    fits: bool                # fits on-chip memory (after tiling, if applicable)
    tiled: bool
    mem_frac: float           # peak activation / budget (pre-tiling)
    latency_ms: float
    fps: float
    act_native: bool
    verdict: str              # "SUITABLE" | "CAVEATS" | "UNSUITABLE"
    notes: list[str] = field(default_factory=list)


_ACCELERATORS = ("hailo8", "deepx")


def assess_targets(cfg: Config, model, quantize_enabled: bool,
                   chosen: str | None = None) -> list[TargetAssessment]:
    """Score the trained model against every Raspberry Pi-class target chip.

    Looks at each chip's spec (precision, native ops, on-chip SRAM budget, the
    compute/latency model) and returns a per-chip verdict so the operator can
    see which silicon this exact network is suitable to deploy on.
    """
    # Local import keeps the compiler importable without torch present.
    from .inference import estimate_device_latency_ms

    patch = cfg.optimization.patch_size
    out: list[TargetAssessment] = []
    for key, caps in CAPS.items():
        accel = key in _ACCELERATORS
        bytes_per = 1 if accel else 2
        act_kb = _estimate_sram_kb(cfg, bytes_per=bytes_per)
        budget = float(caps["sram_kb"])
        mem_frac = act_kb / budget if budget else 0.0
        notes: list[str] = []

        tiled = False
        fits = act_kb <= budget
        if not fits and accel:
            tiled = True           # accelerators can spill to 2x2 spatial tiles
            fits = True
            notes.append("needs 2×2 spatial tiling to fit on-chip SRAM")
        elif not fits:
            notes.append("working set exceeds memory budget")

        act_native = cfg.model.activation in caps["native_acts"]
        if not act_native:
            notes.append(f"'{cfg.model.activation}' not native → "
                         f"PWL/QAT approximation")

        quantized = accel and quantize_enabled
        if accel and not quantize_enabled:
            notes.append("INT8-only NPU but quantization is disabled")

        if cfg.model.model_family in _CONVT_FAMILIES and accel:
            notes.append(f"{cfg.model.model_family.upper()} ConvTranspose "
                         "rewritten to resize+conv")

        transformer_on_npu = cfg.model.model_family in _TRANSFORMER_FAMILIES and accel
        if transformer_on_npu:
            notes.append(f"{cfg.model.model_family.upper()} LayerNorm/softmax "
                         "attention runs in FP fallback (limited NPU offload)")

        latency = estimate_device_latency_ms(model, patch, key, quantized)
        fps = 1000.0 / max(latency, 1e-6)

        # -- verdict ---------------------------------------------------------
        verdict = "SUITABLE"
        if not fits or (accel and not quantize_enabled):
            verdict = "UNSUITABLE"
        elif (tiled or not act_native or transformer_on_npu
              or (cfg.model.model_family in _CONVT_FAMILIES and accel)):
            verdict = "CAVEATS"
        if fps < 5.0:
            notes.append(f"~{fps:.0f} FPS — well below real-time")
            verdict = "UNSUITABLE" if verdict == "UNSUITABLE" else "CAVEATS"
        elif fps < 15.0:
            notes.append(f"~{fps:.0f} FPS — sub-real-time")
            if verdict == "SUITABLE":
                verdict = "CAVEATS"

        out.append(TargetAssessment(
            key=key, label=caps["label"], precision=caps["precision"].upper(),
            format=caps["format"], act_kb=act_kb, budget_kb=budget, fits=fits,
            tiled=tiled, mem_frac=mem_frac, latency_ms=latency, fps=fps,
            act_native=act_native, verdict=verdict, notes=notes,
        ))

    # Put the chosen target first so the report reads naturally.
    if chosen:
        out.sort(key=lambda a: (a.key != chosen, a.key))
    return out
