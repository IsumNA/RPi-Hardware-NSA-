"""Per-model-family option applicability (Level 3).

Single source of truth for which architecture flags actually change the built
graph. NAFNet uses SimpleGate (not relu/gelu/silu); Restormer hard-codes GELU
in its FFN; NAFNet ignores ``conv_type`` in the block implementation.
"""

from __future__ import annotations

from .config import ModelConfig

NO_ACTIVATION = frozenset({"nafnet", "restormer"})
NO_CONV_TYPE = frozenset({"nafnet", "restormer"})
CONVT_FAMILIES = frozenset({"unet", "rednet", "drunet"})
TRANSFORMER_FAMILIES = frozenset({"restormer"})

FIXED_NONLINEARITY = {
    "nafnet": "SimpleGate (split × product)",
    "restormer": "GELU gated FFN (fixed)",
}

# block_depth is internally halved per stage for these U-Nets.
HALVED_DEPTH_FAMILIES = frozenset({"unet", "drunet"})

# REDNet requires at least two encoder/decoder stages.
MIN_BLOCK_DEPTH = {"rednet": 2}


def uses_activation(family: str) -> bool:
    return family not in NO_ACTIVATION


def uses_conv_type(family: str) -> bool:
    return family not in NO_CONV_TYPE


def uses_nafnet_topology(family: str) -> bool:
    return family == "nafnet"


def uses_flat_block_depth(family: str, cfg: ModelConfig) -> bool:
    """``block_depth`` applies to flat stacks; U-shaped NAFNet uses topology fields."""
    if family == "nafnet" and list(cfg.nafnet_enc_blocks or []):
        return False
    return True


def effective_activation(cfg: ModelConfig) -> str:
    if cfg.model_family == "nafnet":
        return "simplegate"
    if cfg.model_family == "restormer":
        return "gelu"
    return cfg.activation


def effective_conv_type(cfg: ModelConfig) -> str:
    if cfg.model_family in NO_CONV_TYPE:
        return "depthwise"
    return cfg.conv_type


def normalize_model_config(cfg: ModelConfig) -> ModelConfig:
    """Enforce family-specific constraints (e.g. REDNet min depth)."""
    fam = cfg.model_family
    min_d = MIN_BLOCK_DEPTH.get(fam)
    if min_d and cfg.block_depth < min_d:
        cfg.block_depth = min_d
    return cfg


def search_combo_valid(family: str, conv_type: str, activation: str) -> bool:
    """Skip redundant grid points (e.g. NAFNet × {gelu, silu})."""
    if not uses_conv_type(family) and conv_type != "depthwise":
        return False
    if not uses_activation(family) and activation != "relu":
        return False
    return True


def profile_rows(cfg: ModelConfig) -> list[tuple[str, str]]:
    """Compilation-profile key/value rows for terminal + GUI."""
    fam = cfg.model_family
    rows: list[tuple[str, str]] = [
        ("model_family", fam),
        ("base_channels", str(cfg.base_channels)),
    ]
    if uses_flat_block_depth(fam, cfg):
        label = "block_depth"
        if fam in HALVED_DEPTH_FAMILIES:
            eff = max(1, cfg.block_depth // 2)
            rows.append((label, f"{cfg.block_depth} ({eff} blocks / encoder stage)"))
        else:
            rows.append((label, str(cfg.block_depth)))
    elif fam == "nafnet":
        enc = list(cfg.nafnet_enc_blocks)
        dec = list(cfg.nafnet_dec_blocks or enc[::-1])
        rows.append(("topology", f"enc {enc} · mid {cfg.nafnet_middle_blocks} · dec {dec}"))
    if uses_conv_type(fam):
        rows.append(("conv_type", cfg.conv_type))
    else:
        rows.append(("conv_type", f"{effective_conv_type(cfg)} (fixed)"))
    if uses_activation(fam):
        rows.append(("activation", cfg.activation))
    else:
        rows.append(("nonlinearity", FIXED_NONLINEARITY[fam]))
    return rows


def model_display_line(cfg: ModelConfig) -> str:
    """Short subtitle for GUI results (family, width, depth, conv, act)."""
    rows = {k: v for k, v in profile_rows(cfg)}
    fam = cfg.model_family.upper()
    width = rows.get("base_channels", str(cfg.base_channels))
    if "topology" in rows:
        depth = rows["topology"]
    else:
        depth = rows.get("block_depth", str(cfg.block_depth))
    conv = rows.get("conv_type", cfg.conv_type)
    if uses_activation(cfg.model_family):
        extra = rows.get("activation", cfg.activation)
    else:
        extra = rows.get("nonlinearity", "")
    return f"{fam} {width}ch · {depth} · {conv} · {extra}"


def instantiate_summary(cfg: ModelConfig) -> str:
    """One-line architecture description for compile logs."""
    fam = cfg.model_family.upper()
    c = cfg.base_channels
    if fam == "NAFNET" and list(cfg.nafnet_enc_blocks or []):
        dec = cfg.nafnet_dec_blocks or cfg.nafnet_enc_blocks[::-1]
        return (f"multi-scale NAFNet ({c}ch, enc {cfg.nafnet_enc_blocks} · "
                f"middle {cfg.nafnet_middle_blocks} · dec {dec})")
    if fam == "NAFNET":
        return f"flat NAFNet ({c}ch × {cfg.block_depth} NAFBlocks, SimpleGate)"
    if fam == "RESTORMER":
        return f"Restormer ({c}ch × {cfg.block_depth} transformer blocks, GELU FFN)"
    parts = [f"{c}ch × depth {cfg.block_depth}"]
    if uses_conv_type(cfg.model_family):
        parts.append(f"{cfg.conv_type} conv")
    if uses_activation(cfg.model_family):
        parts.append(cfg.activation)
    return f"{fam} ({', '.join(parts)})"
