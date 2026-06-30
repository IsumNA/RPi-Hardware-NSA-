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
| 5 | Calibration / Quantization | ✅/◐ | **Real** on-frame training (random-crop, MSE, Adam, cosine LR) and **real** per-channel INT8 weight quant + per-tensor activation fake-quant with a measured FP32→INT8 PSNR drop. QAT is *emulated* (post-hoc), not true fake-quant-in-the-loop training. |
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
| Desktop GUI | ✅ | Imager-styled, DPI-aware, live progress sidebar; comprehensive 6-level options panel |
| Mode: Single Frame / Batch Folder | ✅ | Both working today (radio) |
| Roadmap items shown greyed in UI | ✅ | Temporal Video Denoise, Pareto/Optuna sweep, Custom NAFNet topology, QAT, Live Silicon Deployment, patch-cache builder rendered as disabled "coming soon" controls |
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
| QAT | ◐ | Emulated post-hoc, not fake-quant-in-the-loop |
| ONNX export | ✅ | Real, structurally valid graph (FP32) |
| Device binary (.hef/.bin) | ◐ | Real self-describing container, **not** vendor-runtime loadable |
| On-device latency / FPS | ◐ | Estimated from FLOPs + per-target cost model (clearly labelled) |
| Tiling | ◐ | **Decided** by the compiler, **not executed** in inference (runs whole frame) |
| Generalisation | ◐ | Calibrated on one frame (single) or a small batch of frames; not a full dataset training run |

---

## 5. Architecture / file map

```
config.yaml          # single source of truth for inputs
run_demo.py          # CLI entry; orchestrates all 6 levels
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

- **No vendor compilation.** Hailo Dataflow Compiler / DeepX SDK are not invoked;
  `.hef`/`.bin` cannot yet be deployed to a device.
- **No on-device measurement.** Latency/FPS/power are modelled, not benchmarked.
- **Calibration, not full training.** Single-frame or small-batch on-frame fit;
  no train/val split or large-dataset training run, no generalisation guarantees.
- **Tiling is a decision only.** The inference path does not actually tile.
- **INT8 is not exported.** The ONNX is FP32; quantized weights live only in the
  custom binary (no QDQ ONNX, no `onnxruntime` path — not installed).
- **Basic ISP.** Generic OpenCV demosaic; rawpy handles DNG decode, but no
  black-level/AWB/CCM/tone pipeline or EXIF metadata parsing of our own.
- **No automated Pareto search.** One config → one score; nothing sweeps the
  design space automatically yet.
- **No tests / CI.**

---

## 7. Roadmap — what still needs to be implemented

### Near term (makes the demo a tool)
- ◐ DNG decode works via `rawpy`; still want full metadata (black level, WB, CFA) + green-equalisation like denoise-hw's `dng.py`.
- ⬜ Multi-config **Pareto sweep** + auto-pick of the best point (the score function exists; add the search).
- ⬜ Export a **quantized (QDQ) ONNX** and add an `onnxruntime` inference path.
- ⬜ Unit tests for config validation, noise model, quant, export round-trip; CI.

### Medium term (toward real quality)
- ◐ Real paired low-light RAW **datasets** are now ingested (denoise-hw `PI_RAW` layout: `noisy.dng`/`gt.dng`, keyword filter, detail scoring); next is a full train/val training run rather than on-frame calibration.
- ⬜ **True QAT** (fake-quant nodes during training) + calibration dataset.
- ⬜ Proper **ISP**: black-level subtraction, AWB, colour-correction matrix, tone curve.
- ⬜ Implement **spatial tiling** in the inference path (not just the decision).

### Hardware integration (toward silicon)
- ⬜ Wire the **Hailo Dataflow Compiler** backend → produce a loadable `.hef`.
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

# confirm the ONNX is structurally valid
python -c "import onnx; onnx.checker.check_model(onnx.load('outputs/exported_model.onnx')); print('ONNX OK')"
```
