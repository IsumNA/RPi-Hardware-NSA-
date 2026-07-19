"""On-the-fly synthetic pair dataset for packed-Bayer training.

The tensor contract matches ``train_stream_to_gt.build_pairs`` exactly:

    noisy : (H, W, 4 * T)   float32   T independent noise realizations
                                       stacked on the channel axis, current
                                       frame is channels [0:4].
    gt    : (H, W, 4)       float32   the clean packed frame.

Random crops from the cached ``.npy`` clean images give infinite scene
diversity; each training step draws a fresh noise realization, so a dataset
of N clean images effectively yields N × epochs × steps distinct noisy pairs.

Usage
-----
::

    from nsa.synth.dataset import SynthPairDataset
    from nsa.synth.noise import load_gain_model

    models = {
        (s, g): load_gain_model(f"models/noise/{s}_ag{g}.json")
        for s in ("imx662", "imx662h") for g in (128, 256, 512)
    }
    ds = SynthPairDataset(
        manifest="datasets/synth/clean_manifest.json",
        models=models,
        crop=256,
        temporal=4,
        seed=662,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np

from .noise import GainNoiseModel, synthesize_temporal_stack


class SynthPairDataset:
    """Deterministic-per-seed, infinite (noisy, gt) generator."""

    def __init__(
        self,
        manifest: str | Path | list[dict],
        models: dict[tuple[str, int], GainNoiseModel],
        *,
        crop: int = 256,
        temporal: int = 4,
        sensor_weights: dict[str, float] | None = None,
        gain_weights: dict[int, float] | None = None,
        length: int = 4096,
        seed: int = 662,
        min_intensity: float = 0.005,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            d = json.loads(Path(manifest).read_text(encoding="utf-8"))
            entries = list(d.get("entries", []))
        else:
            entries = list(manifest)
        if not entries:
            raise ValueError("Empty manifest — build the clean cache first")
        if not models:
            raise ValueError("Provide at least one GainNoiseModel")

        self.entries = entries
        self.models = dict(models)
        self.crop = int(crop)
        self.temporal = max(1, int(temporal))
        self.length = int(length)
        self.min_intensity = float(min_intensity)

        sensors = sorted({s for (s, _) in self.models})
        gains = sorted({g for (_, g) in self.models})
        self._sensors = sensors
        self._gains = gains

        sw = {s: float(sensor_weights.get(s, 1.0)) if sensor_weights else 1.0
              for s in sensors}
        gw = {g: float(gain_weights.get(g, 1.0)) if gain_weights else 1.0
              for g in gains}
        self._sensor_w = np.array([sw[s] for s in sensors], dtype=np.float64)
        self._sensor_w /= self._sensor_w.sum()
        self._gain_w = np.array([gw[g] for g in gains], dtype=np.float64)
        self._gain_w /= self._gain_w.sum()

        self._seed = int(seed)
        self._rng_master = np.random.default_rng(self._seed)

    # ------------------------------------------------------------------ #
    #  torch.utils.data.Dataset interface (duck-typed)                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self._seed + int(idx))
        entry = self.entries[int(rng.integers(len(self.entries)))]
        gt = self._load_entry(entry)
        gt = self._random_crop(gt, rng)
        # Filter near-black patches for cleaner training signal
        if float(gt.mean()) < self.min_intensity:
            # try again once, else accept
            gt2 = self._random_crop(self._load_entry(entry), rng)
            if float(gt2.mean()) >= self.min_intensity:
                gt = gt2
        sensor = str(self._sensors[int(rng.choice(len(self._sensors), p=self._sensor_w))])
        gain = int(self._gains[int(rng.choice(len(self._gains), p=self._gain_w))])
        model = self.models.get((sensor, gain))
        if model is None:  # fall back to first available for this gain
            for s in self._sensors:
                if (s, gain) in self.models:
                    sensor, model = s, self.models[(s, gain)]
                    break
        if model is None:
            raise KeyError(f"no model for ({sensor}, {gain})")
        noisy = synthesize_temporal_stack(gt, model, self.temporal, rng=rng)
        return {
            "gt": gt.astype(np.float32),
            "noisy": noisy.astype(np.float32),
            "sensor": sensor,
            "gain": int(gain),
            "src": str(entry.get("path", "")),
        }

    # ------------------------------------------------------------------ #
    #  Iterable-style helper for scripts that don't need Dataset API      #
    # ------------------------------------------------------------------ #

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> list[dict]:
        rng = rng or self._rng_master
        return [self[int(rng.integers(self.length))] for _ in range(n)]

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _load_entry(self, entry: dict) -> np.ndarray:
        arr = np.load(entry["path"])
        return arr.astype(np.float32)

    def _random_crop(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        H, W, C = arr.shape
        if H < self.crop or W < self.crop:
            # up-pad with reflect if source is smaller than crop
            pad_h = max(0, self.crop - H)
            pad_w = max(0, self.crop - W)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            H, W, _ = arr.shape
        y = int(rng.integers(0, H - self.crop + 1))
        x = int(rng.integers(0, W - self.crop + 1))
        # random flips for augmentation
        crop = arr[y:y + self.crop, x:x + self.crop].copy()
        if rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        if rng.random() < 0.5:
            crop = crop[::-1, :].copy()
        return crop
