"""Image-sensor library (Level 1).

Each entry is a physical noise profile for a camera module. Because the noise is
generated from datasheet-style parameters (quantum efficiency, read-noise floor,
full-well capacity, PRNU, chroma cross-talk), the framework can build a faithful
simulated training frame for *any* sensor — including one that has not shipped
yet — straight from its specification.

This is the key that turns NSA from "a model tuned for one camera" into a
framework that auto-optimizes a denoiser for whatever sensor the product uses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SensorProfile:
    key: str
    label: str
    family: str
    full_well: float      # electrons at unity gain
    read_noise: float     # read-noise floor, electrons RMS
    qe: float             # quantum efficiency (0-1)
    chroma_noise: float   # low-frequency chroma splotch std (post-capture)
    prnu: float           # photo-response non-uniformity (fixed pattern), fraction
    bit_depth: int
    bayer: str
    note: str


# -- The library ---------------------------------------------------------------
# Numbers are representative datasheet-style values, not exact silicon figures.
SENSORS: dict[str, SensorProfile] = {
    "imx219": SensorProfile(
        key="imx219", label="IMX219", family="Legacy CMOS",
        full_well=6000, read_noise=4.0, qe=0.55,
        chroma_noise=0.050, prnu=0.012, bit_depth=10, bayer="RGGB",
        note="Legacy Camera Module v2 — high read noise, messy chroma splotches",
    ),
    "imx662": SensorProfile(
        key="imx662", label="IMX662", family="Starvis 2",
        full_well=9000, read_noise=2.0, qe=0.80,
        chroma_noise=0.012, prnu=0.005, bit_depth=12, bayer="RGGB",
        note="Current Starvis 2 — low read noise, mostly photon-shot limited",
    ),
    "imxng": SensorProfile(
        key="imxng", label="IMX-NG", family="Starvis 2 (unreleased)",
        full_well=18000, read_noise=0.8, qe=0.92,
        chroma_noise=0.004, prnu=0.002, bit_depth=12, bayer="RGGB",
        note="Unreleased next-gen low-light — shot-noise dominated, very uniform",
    ),
}

DEFAULT = "imx662"
SENSOR_KEYS = tuple(SENSORS.keys())


def get_sensor(key: str) -> SensorProfile:
    return SENSORS[key]


def with_noise_std(sensor: SensorProfile, read_noise: float | None) -> SensorProfile:
    """Return a copy of ``sensor`` with the Gaussian read-noise std overridden.

    ``read_noise`` is the standard deviation of the additive Gaussian read noise,
    in electrons RMS — the same knob David Plowman's denoise-hw exposes for
    dialling the synthetic noise floor. ``None`` (or a negative value) leaves the
    sensor's datasheet default untouched. The photon shot-noise term is physical
    and always scales with the signal on top of this floor.
    """
    if read_noise is None or read_noise < 0:
        return sensor
    return replace(sensor, read_noise=float(read_noise))
