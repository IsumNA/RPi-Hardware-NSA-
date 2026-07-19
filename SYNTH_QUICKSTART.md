# Synthetic IMX662 Pairs — Quick-Start

Turn the bursts, calibration frames, and (optionally) DIV2K / Flickr2K into a
massive on-the-fly dataset of `(noisy, clean)` packed-Bayer pairs that
`train_stream_to_gt.py` already knows how to train on.

The full design and physics are documented in
[`NOISE_SYNTHESIS_RESEARCH.md`](NOISE_SYNTHESIS_RESEARCH.md).  This is the
run-book.

## 1. Fit per-gain noise models

Reads calibration + bursts → writes `models/noise/imx662{,h}_ag{128,256,512}.json`
(6 files).  Each JSON stores per-channel system gain `K`, read σ, row σ, BLE σ,
and per-channel PTC fit R² for provenance.

```bash
.venv/bin/python fit_imx662_noise.py \
    --sensors imx662 imx662h \
    --gains 128 256 512 \
    --crop 768 --burst-n-gt 128 --burst-n-use 128
```

Per gain, the CLI tries **both** sources — the classical bias/dark/flat fit
from `calibration/<sensor>_gain256/` and a burst-residual PTC — and keeps the
one with higher per-channel R².  If neither source covers a gain, it linearly
scales the nearest available fit (Sony NMIH-style, valid within a conversion-
gain regime).  The chosen source is recorded in each JSON's `fit_source`.

## 2. Build the clean-image cache

Turns burst folders into packed-Bayer `.npy` clean frames, and (optionally)
inverse-ISPs sRGB JPEG/PNG into the same domain.

```bash
# bursts only — fast, uses only what's on disk
.venv/bin/python build_synth_dataset.py --skip-srgb

# add DIV2K / Flickr2K after downloading them into a folder
.venv/bin/python build_synth_dataset.py \
    --srgb-root datasets/DIV2K_train_HR --srgb-tile 1024
```

Output layout:

```
datasets/synth/
  bursts/<scene>__ag1.npy          # burst-averaged clean packed frame
  srgb/<subpath>.npy               # unprocessed sRGB → packed RGGB
  clean_manifest.json              # index that the trainer consumes
```

`.npy` files are float16 by default — half the disk of float32 with no
detectable impact on training.  `--dark-scale` controls how dark the
unprocessed sRGB frames end up (0.2 = very dark, 0.5 = medium-bright).

## 3. Validate the noise model

Runs a held-out real burst through the noise model and compares distributions
against the burst's own residuals: per-channel PTC overlay, KL divergence,
tail probabilities (P > 3σ, P > 5σ), row σ.  Writes a summary PNG per
`(sensor, gain)`.

```bash
.venv/bin/python eval_noise_model.py --auto --n-gt 128 --n-test 32 --crop 512
```

Outputs land in `outputs/noise_eval/`.  Pass criteria are lenient because the
AIM 2025 / Sony NMIH ablations show denoiser accuracy is robust to K within
≈ 2×; the binding signals are noise magnitude (σ) and shape (KL).

## 4. Train with synthetic pairs

`train_stream_to_gt.py` grew four `--synth-*` flags.  You can train on synth
alone (`--synth-only`) or mix synth + real bursts (the default when
`--synth-manifest` is given).

```bash
# synth only, 4096 pairs at 384×384 packed patches, ~2 h on a 4090
.venv/bin/python -u train_stream_to_gt.py \
    --synth-manifest datasets/synth/clean_manifest.json \
    --synth-only --synth-n 4096 --synth-crop 384 \
    --gains 128,256,512 --temporal 4 \
    --steps 20000 --channels 128 --depth 8 --batch 4

# mix: real bursts + 2048 synth pairs (recommended for the deploy model)
.venv/bin/python -u train_stream_to_gt.py \
    --synth-manifest datasets/synth/clean_manifest.json \
    --synth-n 2048 --synth-crop 384 \
    --gains 128,256,512 --temporal 4 \
    --steps 20000 --channels 128 --depth 8 --batch 4
```

## Files added / touched by this pipeline

* `nsa/synth/noise.py`      — `GainNoiseModel`, `synthesize_noisy_packed`,
                               `synthesize_temporal_stack`
* `nsa/synth/fit.py`        — per-channel PTC fit **with intercept**
                               (`fit_from_calibration`, `fit_from_burst`)
* `nsa/synth/sources.py`    — burst → clean, sRGB → clean packed cache
* `nsa/synth/dataset.py`    — `SynthPairDataset` (torch-compatible)
* `fit_imx662_noise.py`     — CLI ①
* `build_synth_dataset.py`  — CLI ②
* `eval_noise_model.py`     — CLI ③
* `train_stream_to_gt.py`   — `--synth-*` flags for CLI ④
* `models/noise/`           — six per-gain JSONs (LCG + HCG × 128/256/512)

## Notes / caveats

* **HCG bursts don't exist on disk yet.**  The `imx662h_*` models today are
  calibration-based (fitted at ag256, linearly scaled to 128 / 512).  To
  validate them, capture a static burst on the Pi with
  `v4l2-ctl --set-ctrl=hcg_enable=1` and drop it under
  `datasets/imx662_project/bursts/<scene>_hcg/ag128/`.  Everything else Just
  Works — the CLI, fit and eval already look for `hcg` in path names.
* **DIV2K/Flickr2K aren't downloaded automatically.**  Download once
  (~7 GB HR / ~3 GB LR) and point `--srgb-root` at the folder.  You can add
  your phone photos or any linear sRGB source too.
* **Row banding validation is coarse.**  Synth row σ is fitted from short
  dark stacks; real bursts have per-frame scene-driven flicker that inflates
  measured row σ.  This is why the eval loosens `row_err_rel_mean` to 0.75 —
  auxiliary metric, not disqualifying.
