"""Bayer-domain synthetic noise for IMX662.

The pipeline builds ``(noisy, clean)`` pairs in the packed-Bayer float domain
that ``train_stream_to_gt.py`` consumes directly, no lossy PNG round-trip:

* ``noise.py``   — noise formation (Poisson shot + Gaussian read + row) with
  per-channel system gain ``K`` and per-gain parameters.
* ``fit.py``     — per-gain, per-channel PTC fit **with intercept** from real
  data (flat pairs where available, burst pairs elsewhere).
* ``sources.py`` — clean-image sources: burst-averaged own scenes, unprocessed
  sRGB packs (DIV2K/Flickr2K → RGGB 12-bit).
* ``dataset.py`` — ``SynthPairDataset`` for training loops.

See ``NOISE_SYNTHESIS_RESEARCH.md`` for the design rationale.
"""

from __future__ import annotations

from .noise import GainNoiseModel, synthesize_noisy_packed, load_gain_model, save_gain_model

__all__ = [
    "GainNoiseModel",
    "synthesize_noisy_packed",
    "load_gain_model",
    "save_gain_model",
]
