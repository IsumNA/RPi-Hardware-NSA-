# NSA — Project Status

**Component:** Neural Sensor Architecture — 6-Level Optimization Stack
**Stage:** Working prototype / demo
**Last updated:** 2026-06-30

Legend: ✅ implemented & working · ◐ implemented but simulated / partial · ⬜ not yet implemented

---

## 1. One-paragraph summary

NSA is a runnable prototype of a hardware-aware image-denoising **compiler**. It
takes a noisy IMX662 Bayer RAW frame plus a target-hardware + model
configuration, and produces a calibrated, INT8-quantized denoiser together with
export artifacts, a visual before/after validation, and a Pareto fitness score.
The full software pipeline (sensor sim → ground truth → model → compiler →
calibration/quantization → export) is real and executes end-to-end. The parts
that depend on physical vendor silicon (the Hailo/DeepX vendor compilers and
on-device timing) are realistically **stubbed/estimated**, not yet wired to real
hardware.

> Real-dataset ingestion (paired `noisy`/`gt` folders, keyword filtering, and
> detail-scored patch selection) is adapted from
> [`davidplowman/denoise-hw`](https://github.com/davidplowman/denoise-hw).

---

## 2. What it does *right now* (end to end)

Running `python run_demo.py` (or the GUI) executes all six levels live:

| Level | Stage | Status | What actually happens today |
|------:|-------|:------:|------------------------------|
| 1 | Sensor / Input | ✅ | Selects a sensor from a **sensor library** (IMX219 / IMX662 / unreleased IMX-NG), then **loads real captures** (paired `noisy`/`gt` folders, single files, uploads, or whole folders; `.npy`/image/`.dng` via rawpy; keyword-filtered; detail-scored crop) **or** synthesises a frame from that sensor's physical noise profile (QE, read-noise floor, full-well, PRNU, chroma; Poisson shot + Gaussian read, gain-scaled). Real frames can optionally have the sensor's noise simulated on top. |
| 2 | Data / Ground Truth | ✅ | Uses **real paired `gt`** when present (denoise-hw convention), else temporally averages N simulated reads, else derives an NL-means reference for a lone real frame. |
| 3 | Architecture | ✅ | Builds a real PyTorch denoiser — CNN (DnCNN-style), U-Net (2-scale), or NAFNet (simplified NAF blocks) — honouring channels / depth / conv-type / activation. |
| 4 | Compiler | ◐ | Runs hardware-aware passes: operator legalization, GELU→QAT / PWL handling, depthwise→grouped mapping, U-Net ConvTranspose rewrite, SRAM budgeting + tiling decision, PTQ-vs-QAT selection, export-format lock. Emits a live log. (Logic is real; it models the constraints rather than calling a vendor compiler.) |
| 5 | Calibration / Quantization | ✅ | **Real** on-frame training (random-crop, MSE, Adam, cosine LR) and **real** per-channel INT8 weight quant + per-tensor activation fake-quant with a measured FP32→INT8 PSNR drop. **True QAT** (fake-quant-in-the-loop with straight-through gradients) is available via `--qat` and is auto-enabled for non-native activations (e.g. gelu→DeepX). |
| 6 | Export | ✅/◐ | Writes a **real, validated** `exported_model.onnx` and a **real, self-describing** INT8 binary (`.hef`/`.bin`/`.ort`: magic header + JSON manifest + packed int8 weights + per-channel scales). The binary is a stand-in container, not a vendor-runtime-loadable file. |

### Four delivered outputs
- ✅ **Live compilation log** (rich CLI), incl. real constraint warnings (e.g. DeepX+GELU → forced QAT, Hailo SRAM tiling).
- ✅ **Artifacts on disk** in `outputs/`: `exported_model.onnx` (passes `onnx.checker`), `hardware_ready.{hef,bin,ort}`.
- ✅ **3-panel visual validation** (Raspberry Pi Imager-styled, real logo): raw input · ground truth · model output, with measured PSNR badges.
- ✅ **Pareto fitness scorecard**: quality + latency + INT8-robustness → single score with OPTIMAL/BALANCED/SUBOPTIMAL grade.

---

## 3. What it *can do* (capabilities & knobs)

| Capability | Status | Notes |
|------------|:------:|-------|
| Sensor library (Level 1) | ✅ | `imx219` (legacy), `imx662` (Starvis 2), `imxng` (unreleased low-light); add more in `nsa/sensors.py` |
| Optimise for an *unreleased* sensor | ✅ | Physics-based noise injection from datasheet params — no hardware needed |
| Target hardware selection | ✅ | `rpi5_cpu` (FP16/.ort), `hailo8` (INT8/.hef), `deepx` (INT8/.bin) |
| Model family | ✅ | `cnn`, `unet`, `nafnet` |
| Width / depth | ✅ | `base_channels` 16/32/64, `block_depth` 2/4/8 |
| Convolution type | ✅ | `standard`, `depthwise`-separable |
| Activation | ✅ | `relu`, `gelu`, `silu` (gelu drives the DeepX QAT path) |
| Sensor gain | ✅ | 256× / 512× challenge frames |
| Real RAW / image input | ✅ | `.npy`, standard images, and `.dng` (via `rawpy` if installed) |
| Real **paired** datasets | ✅ | `noisy.*`/`gt.*` folders auto-detected → real ground truth (denoise-hw convention) |
| Dataset keyword filter | ✅ | `--filter imx219 ag12` (denoise-hw semantics) |
| Detail-scored patch crop | ✅ | Laplacian-variance scoring picks the sharpest crop |
| Simulate noise on real frames | ✅ | `--simulate-noise`: inject a sensor's physics on top of loaded frames |
| Batch / multi-image calibration | ✅ | `--batch N`: calibrate across crops from many frames; averaged metrics |
| Upload images / choose folder | ✅ | GUI multi-file upload or folder picker |
| Config via YAML **or** CLI flags | ✅ | `config.yaml` + full `--flag` overrides |
| Calibration step count | ✅ | `--steps` (speed/quality trade-off) |
| Desktop GUI | ✅ | Imager-styled, DPI-aware, live progress sidebar; comprehensive 6-level options panel; rich results screen (model details + metrics + image + full log) |
| Mode: Single / Batch / Temporal video | ✅ | All three working today (radio) |
| Temporal video denoise | ✅ | `--temporal --burst N`: recursive IIR burst denoise, writes a denoised frame sequence to `outputs/video/` |
| True QAT | ✅ | `--qat`: fake-quant-in-the-loop training (per-channel weights + per-tensor acts, STE gradients) |
| Custom multi-scale NAFNet | ✅ | `--nafnet-enc 1 2 2 --nafnet-middle 4 --nafnet-dec 2 2 1`: U-shaped NAFNet with PixelShuffle up/down + skips |
| Automated Pareto sweep | ✅ | `search.py`: grid (or `--optuna N` TPE) search; writes a Pareto front + winner to `outputs/pareto.json` |
| Patch-cache training-set builder | ✅ | `cache.py`: detail-scored crops → `outputs/patch_cache/` (denoise-hw `dataset.py` idea) |
| Deployment package builder | ✅ | `deploy.py`: bundles artifacts + `FLASH_INSTRUCTIONS.md` + `manifest.json` into a `.zip` (flashing still needs the vendor SDK + device) |
| One-click compile & export | ✅ | `run_demo.py --export` (and the GUI *Compile & export* checkbox) builds the transferable hardware package automatically at the end of a compile |
| CLI | ✅ | Branded rich terminal UI |
| Per-config evaluation | ✅ | Computes one Pareto point per run |
| Reproducibility | ✅ | Seeded RNG (`output.seed`) |

### Representative demo scenarios that work today
- Default Pi 5 + Hailo-8 NAFNet → ~28 dB, ~13 ms/frame, score ~87 (OPTIMAL).
- DeepX + GELU → live "forced QAT injection" warning.
- DeepX 64ch×8 → activation memory overflows SRAM → automatic 2×2 tiling decision.
- Pi 5 CPU FP16 path → no quantization, slower latency estimate.

---

## 4. What is *real* vs *simulated* (fidelity ledger)

| Aspect | Real | Simulated / estimated |
|--------|:----:|:----------------------|
| Sensor noise physics | ◐ | Plausible photon-transfer model per sensor profile (QE, read noise, full-well, PRNU, chroma); values are representative, not measured from each specific chip |
| Ground-truth frame | ✅ | Real paired `gt` when available, else temporal averaging of simulated reads |
| Model + forward/backward | ✅ | Genuine PyTorch graphs and training |
| PSNR numbers | ✅ | Measured from the actual tensors |
| INT8 quantization | ✅ | Real per-channel weight + per-tensor activation quant; measured drop |
| QAT | ✅ | True fake-quant-in-the-loop training (STE); `--qat` or auto for non-native acts |
| ONNX export | ✅ | Real, structurally valid graph (FP32) |
| Device binary (.hef/.bin) | ◐ | Real self-describing container, **not** vendor-runtime loadable |
| On-device latency / FPS | ◐ | Estimated from FLOPs + per-target cost model (clearly labelled) |
| Tiling | ◐ | **Decided** by the compiler, **not executed** in inference (runs whole frame) |
| Generalisation | ◐ | Calibrated on one frame (single) or a small batch of frames; not a full dataset training run |

---

## 5. Architecture / file map

```
config.yaml          # single source of truth for inputs
run_demo.py          # CLI entry; orchestrates all 6 levels (single/batch/temporal)
search.py            # automated Pareto sweep (grid or Optuna TPE) over the design space
cache.py             # patch-cache training-set builder (detail-scored crops)
deploy.py            # deployment package builder (artifacts + flash instructions + zip)
nsa_gui.py           # Raspberry Pi Imager-styled desktop UI (DPI-aware)
requirements.txt
nsa/
  theme.py           # CLI branding, palette, UTF-8 + DPI setup
  config.py          # dataclasses, YAML load, CLI flags, validation
  sensors.py         # sensor library (per-sensor physical noise profiles)
  raw_io.py          # sensor sim, noise model, mosaic/demosaic, RAW loader
  models.py          # CNN / U-Net / NAFNet (configurable)
  compiler.py        # per-target capability table + legalization passes
  inference.py       # calibration, INT8 quant, latency model, PSNR
  export.py          # ONNX export + packed INT8 device binary
  visualize.py       # 3-panel light-themed validation matrix
  report.py          # Pareto fitness scorecard
assets/rpi_logo.png  # logo used by GUI + panel
outputs/             # generated artifacts (gitignored)
```

---

## 6. Known limitations

- **No vendor compilation / flashing.** Hailo Dataflow Compiler / DeepX SDK are
  not invoked. `deploy.py` packages everything up to the hand-off, but actually
  flashing/booting `.hef`/`.bin` needs the vendor SDK + the physical accelerator.
- **No on-device measurement.** Latency/FPS/power are modelled, not benchmarked
  (needs the real device).
- **Calibration, not full training.** Single-frame or small-batch on-frame fit;
  `cache.py` prepares a patch cache, but a full train/val training run over it is
  still future work.
- **Tiling is a decision only.** The inference path does not actually tile.
- **INT8 is not exported to ONNX.** The ONNX is FP32; quantized weights live only
  in the custom binary (no QDQ ONNX / `onnxruntime` path).
- **Basic ISP.** Generic OpenCV demosaic; rawpy handles DNG decode, but no
  black-level/AWB/CCM/tone pipeline or EXIF metadata parsing of our own.
- **No tests / CI.**

---

## 7. Roadmap — what still needs to be implemented

### Done since the first prototype
- ✅ Multi-config **Pareto sweep** + auto-pick (`search.py`, grid or `--optuna` TPE; writes `outputs/pareto.json`).
- ✅ **True QAT** (fake-quant-in-the-loop, STE) via `--qat` / auto for non-native acts.
- ✅ **Custom multi-scale NAFNet** topology (`--nafnet-enc/--nafnet-middle/--nafnet-dec`).
- ✅ **Temporal video denoise** (`--temporal --burst N`, recursive IIR, writes `outputs/video/`).
- ✅ **Patch-cache builder** (`cache.py`) and **deployment package builder** (`deploy.py`).

### Near term
- ◐ DNG decode works via `rawpy`; still want full metadata (black level, WB, CFA) + green-equalisation like denoise-hw's `dng.py`.
- ⬜ Export a **quantized (QDQ) ONNX** and add an `onnxruntime` inference path.
- ⬜ A full **train/val training run** over a patch cache (currently on-frame calibration).
- ⬜ Unit tests for config validation, noise model, quant, export round-trip; CI.

### Medium term (toward real quality)
- ⬜ Proper **ISP**: black-level subtraction, AWB, colour-correction matrix, tone curve.
- ⬜ Implement **spatial tiling** in the inference path (not just the decision).
- ⬜ Motion-compensated temporal denoise (current temporal mode assumes low motion).

### Hardware integration (toward silicon — needs the physical accelerator + vendor SDK)
- ⬜ Wire the **Hailo Dataflow Compiler** backend → produce a loadable `.hef` (`deploy.py` prepares the hand-off).
- ⬜ Wire the **DeepX toolchain** backend → produce a loadable `.bin`.
- ⬜ **On-device benchmarking** harness → replace estimated latency with measured ms / FPS / mW.
- ⬜ End-to-end **camera capture → denoise → display** loop on a Raspberry Pi 5.

---

## 8. How to verify the current state quickly

```bash
# full pipeline, fast
python run_demo.py --steps 70

# the DeepX + GELU compiler path
python run_demo.py --hardware deepx --activation gelu --steps 70

# true QAT (fake-quant in the loop)
python run_demo.py --hardware hailo8 --qat --steps 70

# custom multi-scale NAFNet topology
python run_demo.py --model-family nafnet --nafnet-enc 1 2 2 --nafnet-middle 2 --nafnet-dec 2 1 1 --steps 70

# temporal video denoise (writes outputs/video/)
python run_demo.py --temporal --burst 8 --steps 70

# automated Pareto sweep (writes outputs/pareto.json); add --optuna 20 for TPE
python search.py --hardware hailo8 --model-family cnn --search-steps 40 --no-final-run

# one-click: compile AND export the transferable hardware package
python run_demo.py --hardware hailo8 --export --no-window

# build a patch cache, then a deployment package
python cache.py --dataset ./datasets/imx219_raws --per-image 6
python deploy.py

# confirm the ONNX is structurally valid
python -c "import onnx; onnx.checker.check_model(onnx.load('outputs/exported_model.onnx')); print('ONNX OK')"
```
