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
| **1** | **Sensor / Input** | Pick a sensor from the **sensor library** (legacy IMX219, current IMX662, or an unreleased next-gen low-light part); ingest a real Bayer RAW or synthesise a physically-plausible one from that sensor's noise profile at extreme analog gain (256× / 512×); demosaic to linear RGB. |
| **2** | **Data / Ground Truth** | Build a clean reference by temporally averaging many sensor reads. |
| **3** | **Architecture** | Instantiate the denoiser (`cnn` / `unet` / `nafnet`) with the chosen channels, depth, conv type and activation. |
| **4** | **Compiler** | Legalize operators for the target NPU, budget on-chip SRAM (tiling if needed), and select the quantization scheme (PTQ vs forced QAT). |
| **5** | **Calibration / Quantization** | Live-calibrate the model on the test frame and quantize FP32 → INT8, measuring the accuracy drop. |
| **6** | **Export** | Emit the export profile: `exported_model.onnx` plus a hardware-ready `.hef` / `.bin` / `.ort`. |

---

## The sensor library (why this is a framework, not a one-off)

Level 1 is driven by a **sensor profile** — a set of datasheet-style physical
parameters (quantum efficiency, read-noise floor, full-well capacity, PRNU,
chroma cross-talk). Because the noisy frame is generated from these numbers, NSA
can build a faithful training/validation frame for **any** sensor, including one
that has not shipped yet, straight from its specification.

| Profile | Role | Character |
|---------|------|-----------|
| `imx219` | Legacy (Camera Module v2) | High read noise, messy chroma splotches — hard to clean |
| `imx662` | Current (Starvis 2) | Low read noise, mostly photon-shot limited |
| `imxng` | **Unreleased** next-gen low-light | Shot-noise dominated, very uniform — a small NAFNet cleans it almost perfectly |

This means we can target the **unreleased** low-light sensor today: model its
physical noise, auto-compile an ultra-light denoiser for it, and have a
production-ready, hardware-accelerated pipeline the day the chip leaves the
factory. (Add a new sensor by appending one entry to `nsa/sensors.py`.)

### Real captures vs. simulated physics

Two ways to feed Level 1:

* **Simulated** (default) — `--sensor imx662` / `imxng`: the noisy frame is
  synthesised from the sensor's physical noise profile, and the clean reference
  is a temporal average of many independent reads.
* **Real capture** — `--real --dataset <folder>` (or set `sensor.real_capture`
  and `sensor.dataset_path` in `config.yaml`): an actual frame from the folder is
  used **as** the noisy input. Since a single capture has no temporal ground
  truth, a clean reference is derived from it (NL-means + edge-preserving
  denoise) so calibration and PSNR still work. Supported files: `.npy`, common
  images (`.png/.tif/.jpg/...`), and `.dng` (needs `rawpy`). Point `dataset_path`
  at a cloned repo of IMX219 frames and they flow straight into the stack.

In the GUI this is the **Capture Source** toggle at Level 1: *Real IMX219*,
*Simulated IMX662*, or *Simulated IMX-NG*.

---

## Inputs

Edit `config.yaml`, or override any value on the command line.

| Flag | Choices | Meaning |
|------|---------|---------|
| `--sensor` | `imx219` \| `imx662` \| `imxng` | Image sensor noise profile (Level 1) |
| `--hardware` | `rpi5_cpu` \| `hailo8` \| `deepx` | Target export profile (Level 6) |
| `--model-family` | `cnn` \| `unet` \| `nafnet` | Architecture (Level 3) |
| `--base-channels` | `16` \| `32` \| `64` | Width |
| `--block-depth` | `2` \| `4` \| `8` | Depth |
| `--conv-type` | `standard` \| `depthwise` | Convolution style |
| `--activation` | `relu` \| `gelu` \| `silu` | Activation (`gelu` on DeepX forces QAT) |
| `--input-raw` | *path* | A real Bayer RAW (`.npy` / image). Omitted ⇒ synthetic frame |
| `--dataset` | *path* | Folder (or file) of real captures for real-capture mode |
| `--real` | — | Use real captures from `--dataset` / `dataset_path` as the noisy input |
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

## Running it in the terminal — Linux (primary)

This is the main, intended way to run NSA (Raspberry Pi OS / Ubuntu / Debian).

### 1. Open a terminal in the project folder

```bash
cd ~/RPi-Hardware-NSA-
```

### 2. Create and activate a virtual environment

Keeping the dependencies in a project-local `.venv` avoids polluting your system
Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. To leave it later, run `deactivate`.

> **`python3-venv` missing?** On Debian/Ubuntu/Raspberry Pi OS install it once:
> ```bash
> sudo apt update && sudo apt install -y python3-venv python3-tk
> ```
> `python3-tk` is the Tk toolkit the desktop UI needs (it is not bundled with
> the headless Python on some distros).

### 3. (First time only) install the dependencies

With the environment active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch it

**Option A — Desktop UI (recommended for the demo):**

```bash
python run_demo.py            # CLI, or:
python nsa_gui.py             # desktop window
```

In the UI, pick your options and press **RUN COMPILE**. The sidebar lights up
level-by-level and the validation matrix opens at the end.

**Option B — Command line:** run the full pipeline with the defaults in
`config.yaml`:

```bash
python run_demo.py
```

You'll see Levels 1–6 stream live, a calibration progress bar, the saved
artifacts, and finally the Pareto scorecard. A 3-panel window opens at the end
(close it to let the script finish).

> **Headless box (no display)?** Skip the pop-up window with `--no-window`; the
> panel is still saved to `outputs/validation_panel.png`.

### 5. Drive it with flags

Override any option from `config.yaml` on the command line:

```bash
# The DeepX + GELU -> forced QAT compiler path (great talking point)
python run_demo.py --hardware deepx --activation gelu --model-family nafnet

# Lightweight CNN for the Pi 5 CPU at 256x gain
python run_demo.py --hardware rpi5_cpu --model-family cnn --gain 256

# Heavy model that overflows DeepX SRAM -> automatic tiling kicks in
python run_demo.py --hardware deepx --base-channels 64 --block-depth 8

# Fast run (fewer calibration steps) and no pop-up window
python run_demo.py --steps 70 --no-window

# Feed a real IMX662 Bayer RAW frame instead of the synthetic one
python run_demo.py --input-raw path/to/frame.npy

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

```bash
xdg-open outputs            # or: ls -lh outputs
```

---

## Running it in the terminal — Windows (spare)

Same steps in PowerShell, with two Windows-only extras.

```powershell
# 1. Go to the project folder
cd "C:\Users\isump\Downloads\RPi-Hardware-NSA-"

# 2. Create + activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Enable UTF-8 for this session so the Pi glyphs render
$env:PYTHONUTF8 = "1"

# 5. Launch (UI or CLI)
python nsa_gui.py
python run_demo.py --hardware deepx --activation gelu

# 6. Open the results
explorer outputs
```

> **PowerShell blocks the activate script?** If you see *"running scripts is
> disabled on this system"*, allow local scripts for your user once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> Or use the classic prompt: `.\.venv\Scripts\activate.bat` (cmd).

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
