# NSA — Neural Sensor Architecture
### A working 6-Level Optimization Stack for hardware-aware RAW image denoising

This prototype turns a **theoretical 6-level optimization stack** into a real,
runnable software toolchain. You give it a target accelerator and a model
configuration; it ingests a noisy **IMX662 Bayer RAW** frame and compiles a
**hardware-ready, quantized denoiser** — printing a live compilation log,
writing the export artifacts, opening a visual before/after matrix, and scoring
the result on a Pareto fitness scale.

Everything is genuine: real PyTorch models, real on-frame calibration, real
ONNX export, real per-channel INT8 quantization, and real PSNR measurements.

---

## The 6-Level Optimization Stack

| Level | Stage | What happens |
|------:|-------|--------------|
| **1** | **Sensor / Input** | Ingest an IMX662 Bayer RAW frame (or synthesise a physically-plausible one) at extreme analog gain (256× / 512×); demosaic to linear RGB. |
| **2** | **Data / Ground Truth** | Build a clean reference by temporally averaging many sensor reads. |
| **3** | **Architecture** | Instantiate the denoiser (`cnn` / `unet` / `nafnet`) with the chosen channels, depth, conv type and activation. |
| **4** | **Compiler** | Legalize operators for the target NPU, budget on-chip SRAM (tiling if needed), and select the quantization scheme (PTQ vs forced QAT). |
| **5** | **Calibration / Quantization** | Live-calibrate the model on the test frame and quantize FP32 → INT8, measuring the accuracy drop. |
| **6** | **Export** | Emit the export profile: `exported_model.onnx` plus a hardware-ready `.hef` / `.bin` / `.ort`. |

---

## Inputs

Edit `config.yaml`, or override any value on the command line.

| Flag | Choices | Meaning |
|------|---------|---------|
| `--hardware` | `rpi5_cpu` \| `hailo8` \| `deepx` | Target export profile (Level 6) |
| `--model-family` | `cnn` \| `unet` \| `nafnet` | Architecture (Level 3) |
| `--base-channels` | `16` \| `32` \| `64` | Width |
| `--block-depth` | `2` \| `4` \| `8` | Depth |
| `--conv-type` | `standard` \| `depthwise` | Convolution style |
| `--activation` | `relu` \| `gelu` \| `silu` | Activation (`gelu` on DeepX forces QAT) |
| `--input-raw` | *path* | A real IMX662 Bayer RAW (`.npy` / image). Omitted ⇒ synthetic frame |
| `--gain` | `256` \| `512` | Analog gain of the challenge frame |
| `--steps` | *int* | Calibration steps (lower = faster demo) |
| `--no-window` | — | Skip the pop-up validation window |

---

## Outputs

1. **Live compilation log** — every level prints to the terminal as it runs,
   including constraint warnings such as:

   ```
   ▲ GELU activation detected for DeepX target. Forcing QAT layer injection
     to prevent compilation failure...
   ```

2. **Model artifacts** (in `outputs/`):
   - `exported_model.onnx` — the FP32 baseline graph (validated with `onnx.checker`)
   - `hardware_ready.hef` (Hailo) / `hardware_ready.bin` (DeepX) / `hardware_ready.ort` (Pi CPU)
     — a real, self-describing INT8 container (header manifest + packed weights + scales)

3. **Visual validation matrix** — `outputs/validation_panel.png` and a pop-up
   window with three panels: **A** raw input · **B** ground truth · **C** model output.

4. **Pareto fitness report** — a scorecard balancing image quality, latency and
   INT8 robustness into a single `FINAL PARETO FITNESS SCORE`.

---

## Quick start

```bash
pip install -r requirements.txt

# Desktop UI (Raspberry Pi Imager styling) - recommended for the demo
python nsa_gui.py

# Or the CLI:
# Default demo (Raspberry Pi 5 + Hailo-8, NAFNet, 512x gain)
python run_demo.py

# Show the DeepX + GELU -> forced QAT compiler path
python run_demo.py --hardware deepx --activation gelu --model-family nafnet

# Lightweight CNN for the Pi 5 CPU at 256x gain
python run_demo.py --hardware rpi5_cpu --model-family cnn --gain 256

# Fast run for a quick demo (fewer calibration steps)
python run_demo.py --steps 80
```

> On Windows, run with UTF-8 so the Raspberry Pi glyphs render:
> `$env:PYTHONUTF8=1; python run_demo.py`

---

## Notes on fidelity

- The denoiser is **really trained and run** on the live frame — the PSNR
  numbers and the before/after panels are measured, not scripted.
- INT8 quantization uses **per-channel symmetric weight quantization plus
  per-tensor activation quantization**, so the FP32→INT8 accuracy drop is real.
- The `.hef` / `.bin` are **not** produced by the vendor SDK (Hailo Dataflow
  Compiler / DeepX toolchain are not installed on this demo box). They are real,
  non-empty, self-describing INT8 containers that stand in for the vendor binary;
  swapping in the vendor compiler at Level 6 is the only change needed for silicon.
- On-device latency is an **estimate** derived from model FLOPs and a per-target
  cost model; it is clearly labelled as an estimate in the log.
