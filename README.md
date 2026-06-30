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

## Running it in the terminal — step by step

### 1. Open a terminal in the project folder

```powershell
cd "C:\Users\isump\Downloads\RPi-Hardware-NSA-"
```

### 2. (First time only) install the dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Windows only) enable UTF-8 for the current session

The live log uses Raspberry Pi glyphs (✓, ▲, ·, box characters). Run this once
per terminal session so they render instead of erroring on the legacy console:

```powershell
$env:PYTHONUTF8 = "1"
```

> On macOS / Linux you can skip this step.

### 4. Launch it

**Option A — Desktop UI (recommended for the demo):**

```powershell
python nsa_gui.py
```

Pick your options in the window, then press **RUN COMPILE**. The sidebar lights
up level-by-level and the validation matrix opens at the end.

**Option B — Command line:** run the full pipeline with the defaults in
`config.yaml`:

```powershell
python run_demo.py
```

You'll see Levels 1–6 stream live, a calibration progress bar, the saved
artifacts, and finally the Pareto scorecard. A 3-panel window opens at the end
(close it to let the script finish).

### 5. Drive it with flags

Override any option from `config.yaml` on the command line:

```powershell
# The DeepX + GELU -> forced QAT compiler path (great talking point)
python run_demo.py --hardware deepx --activation gelu --model-family nafnet

# Lightweight CNN for the Pi 5 CPU at 256x gain
python run_demo.py --hardware rpi5_cpu --model-family cnn --gain 256

# Heavy model that overflows DeepX SRAM -> automatic tiling kicks in
python run_demo.py --hardware deepx --base-channels 64 --block-depth 8

# Fast run (fewer calibration steps) and no pop-up window
python run_demo.py --steps 70 --no-window

# Feed a real IMX662 Bayer RAW frame instead of the synthetic one
python run_demo.py --input-raw "path\to\frame.npy"

# See every available flag
python run_demo.py --help
```

### 6. Find the results

Everything is written to the `outputs/` folder:

| File | What it is |
|------|------------|
| `exported_model.onnx` | FP32 baseline graph (ONNX) |
| `hardware_ready.hef` / `.bin` / `.ort` | hardware-ready INT8 artifact for the chosen target |
| `validation_panel.png` | the 3-panel before / ground-truth / after image |

```powershell
# Open the outputs folder
explorer outputs
```

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
