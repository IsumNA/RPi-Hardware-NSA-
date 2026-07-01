# NSA — Neural Architecture Search
### A working 6-Level neural architecture search stack for hardware-aware RAW image denoising

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
| **3** | **Architecture** | Instantiate the denoiser (`cnn` / `dncnn` / `unet` / `rednet` / `ridnet` / `nafnet`) with the chosen channels, depth, conv type and activation. |
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

In the GUI this is the **Capture Source** toggle at Level 1 (*Simulated capture*
vs *Real captures*) plus the **Sensor Profile** dropdown.

### Real datasets, paired ground truth, and batches

The real-dataset ingestion reuses the conventions of
[`davidplowman/denoise-hw`](https://github.com/davidplowman/denoise-hw):

* **Paired folders** — a folder containing a `noisy.*` and a `gt.*` frame is
  treated as a paired capture, and the `gt` frame is used as **real ground
  truth** (better than a derived reference). Point `--dataset` at a tree of such
  folders (e.g. `PI_RAW/<scene>/imx219_ag12/{noisy,gt}.dng`).
* **Keyword filter** — `--filter imx219 ag12` keeps only folders whose path
  contains *all* tokens (same semantics as denoise-hw's `--filter`).
* **Detail-scored crops** — the working patch is chosen by a Laplacian-variance
  detail score, so the demo lands on a sharp, interesting region.
* **Batch mode** — `--batch N` loads up to N frames and calibrates across random
  crops drawn from all of them (denoise-hw's "patches across many images"), then
  reports averaged PSNR.
* **DNG** — `.dng` decoding uses `rawpy` if installed; `.npy`/`.png`/`.tif`/...
  always work.

> Credit: the real-dataset structure, keyword filtering, paired noisy/gt
> convention, and detail-scored patch selection are adapted from
> [denoise-hw](https://github.com/davidplowman/denoise-hw).

**Testing on denoise-hw images** — the denoise-hw repo is training code only;
captures live in a `PI_RAW/Data/…` tree (on a Pi, usually `/opt/datasets/PI_RAW`).
NSA ships the same folder layout under `datasets/PI_RAW` and defaults
`config.yaml` to real captures:

```bash
python setup_denoise_hw_data.py                    # sample PNG pairs + instructions
python setup_denoise_hw_data.py --link /opt/datasets/PI_RAW   # Pi: use real DNGs
python run_demo.py --real --dataset datasets/PI_RAW --filter imx219 ag12 --sensor imx219
```

This uses the same test scene as denoise-hw's `test.py`
(`cabinet_D50_100/imx219_ag12_test`).

---

## Inputs

Edit `config.yaml`, or override any value on the command line.

| Flag | Choices | Meaning |
|------|---------|---------|
| `--sensor` | `imx219` \| `imx662` \| `imxng` | Image sensor noise profile (Level 1) |
| `--hardware` | `rpi5_cpu` \| `hailo8` \| `deepx` | Target export profile (Level 6) |
| `--model-family` | `cnn` \| `dncnn` \| `unet` \| `rednet` \| `ridnet` \| `nafnet` \| `ffdnet` \| `drunet` \| `restormer` | Architecture (Level 3) — 9 families |
| `--base-channels` | `16` \| `32` \| `64` | Width |
| `--block-depth` | `2` \| `4` \| `8` | Depth |
| `--conv-type` | `standard` \| `depthwise` | Convolution style |
| `--activation` | `relu` \| `gelu` \| `silu` | Activation (`gelu` on DeepX forces QAT) |
| `--input-raw` | *path* | A real Bayer RAW (`.npy` / image). Omitted ⇒ synthetic frame |
| `--dataset` | *path* | Folder (or file) of real captures for real-capture mode |
| `--real` | — | Use real captures from `--dataset` / `dataset_path` as the noisy input |
| `--simulate-noise` | — | Inject the selected sensor's noise on top of loaded frames |
| `--filter` | *words* | Keyword filter for dataset folders (denoise-hw style, e.g. `imx219 ag12`) |
| `--batch` | *int* | Batch mode: process up to N frames and average the metrics |
| `--temporal` | — | Temporal video-denoise mode (recursive burst denoising) |
| `--burst` | *int* | Frames in a temporal-denoise burst (default 8) |
| `--frames` | *int* | Temporal frames averaged for the synthetic ground truth |
| `--qat` | — | Quantization-aware training (fake-quant in the loop, STE) |
| `--nafnet-enc` | *ints* | Custom NAFNet encoder block counts, e.g. `1 2 2` |
| `--nafnet-middle` | *int* | Custom NAFNet bottleneck block count |
| `--nafnet-dec` | *ints* | Custom NAFNet decoder block counts, e.g. `2 2 1` |
| `--gain` | `256` \| `512` | Analog gain of the challenge frame |
| `--steps` | *int* | Calibration steps (lower = faster demo) |
| `--export` | — | **Compile & export**: after the run, bundle a transferable hardware package (`outputs/deployment/…zip`) |
| `--no-window` | — | Skip the pop-up validation window |

### Companion tools

| Tool | What it does |
|------|--------------|
| `python search.py --hardware hailo8` | Automated **Pareto sweep** over the design space (grid, or `--optuna N` for a TPE search). Add `--all-sensors` to also sweep every sensor profile (IMX219 · IMX662 · IMX-NG). Prints a Pareto front + winner and writes `outputs/pareto.json`. |
| `python live.py` | **Live testing** — runs the last-compiled model (`outputs/model.pt`) on a live camera feed and shows the **raw sensor frame next to the denoised output** in real time, with live latency / FPS and a noise-reduction readout. Auto-detects the Raspberry Pi CSI camera (picamera2, e.g. the IMX662 low-light module) → a USB webcam (`--source opencv`) → a simulated low-light stream (`--source sim`) so it works even with no camera. |
| `python hf_search.py --query qwen --size small` | **Hugging Face model sourcing** — searches the Hub filtered strictly to Apache-2.0 / MIT, tiered by size (small 1-8B → mid 8-20B → large 20-80B). `--freeze MODEL` locks the exact commit SHA into `outputs/hf_lock.json` (add `--download` to pull a pinned snapshot into `models/frozen/`); `--list-locked` shows what's frozen. |
| `python cache.py --dataset DIR --per-image 6` | **Patch-cache builder** — detail-scored crops → `outputs/patch_cache/` for full training runs (denoise-hw `dataset.py` idea). |
| `python setup_denoise_hw_data.py` | **denoise-hw test images** — prepare `datasets/PI_RAW` (sample PNG pairs or `--link /opt/datasets/PI_RAW` on Pi); patches `config.yaml` with `--write-config`. |
| `python deploy.py` | **Deployment package** — bundles the compiled artifacts + `FLASH_INSTRUCTIONS.md` + `manifest.json` into `outputs/deployment/…zip` (flashing still needs the vendor SDK + device). Run automatically when you pass `--export` (or tick *Compile & export* in the GUI). |

All of the above are also available as controls in the desktop GUI. The GUI is a
**step-by-step wizard**: it asks what you want to do first (*test one specific
model* vs *sweep & rank many*), then walks you one page at a time through the
image sensor, the capture source & data, the model architecture, and the
hardware/calibration, finishing on a **Review & run** page that summarises every
choice before you launch (*Back* / *Next* navigate; the run button reads *RUN
COMPILE* or *RUN SWEEP* to match your choice). On the sensor page you can tick
**Test ALL sensor profiles** (sweeps only) to vary IMX219 / IMX662 / IMX-NG as
well, so the leaderboard shows which model suits which camera.

The sweep trains **all 9 model families** and produces a **ranked, clickable
leaderboard** — click any model row to review its config and run that exact model
(or *Load into form*); *Use winner* loads the top result. A **"Best for" filter**
(All chips / Pi 5 CPU / Hailo-8 / DeepX) re-ranks the board so the models that are
actually suitable for the chip you care about float to the top, each row showing
that chip's verdict (`RUNS WELL` / `WITH CAVEATS` / `NOT REC.`), its FPS, a
*standout* tag (`top pick` / `sharpest` / `fastest` / `leanest`), and — when you
ran an all-sensors sweep — a **SENSOR** column. Package export is an on-demand
action on the results screen (*Export Package*), so a normal compile doesn't
write a package every time.

### Sourcing external models from Hugging Face

The model step also has a **Browse Hugging Face** button (and the standalone
`hf_search.py` CLI) that brings the methodology in-house. The GUI browser
**auto-loads a relevant model list as soon as it opens** — no need to invent a
search query. A **Category** dropdown (Low-light enhancement, Image denoising,
Image restoration, Super-resolution, …) re-loads the relevant models instantly;
an optional *Refine* keyword and *Size* filter narrow it further.

1. **Filter by License** — only **Apache-2.0 / MIT** models are ever returned, so
   legal risk is eliminated up front.
2. **Pick a relevant category** — the list is pre-filtered to image-to-image
   denoising / restoration models that fit this project (most are *tiny*, so the
   *Size* filter defaults to **any**). For LLM-style sourcing, the CLI still
   supports the small (1-8B) → mid (8-20B) → large (20-80B) tiers.
3. **Test the Gap** — switch Category or bump the *Size* selector to see whether
   a bigger model's accuracy jump justifies the extra compute.
4. **Freeze the Weights** — **FREEZE** locks that model's exact commit SHA into
   `outputs/hf_lock.json` (your secure manifest); `--download` additionally pulls
   a pinned snapshot into `models/frozen/`, so an upstream update can never
   silently break your production pipeline.

Search, license-vetting and freezing the commit hash use only the Python
standard library (the public Hub API); downloading a pinned snapshot uses the
optional `huggingface_hub` package. This is a discovery + vetting + freeze
front-end — the on-device 6-level compile still targets the built-in denoisers.

### Live testing on the Pi camera

After a compile, the results screen has a **LIVE TESTING** button that opens an
**in-app live view styled like the rest of the UI** (white surface, raspberry
accents, the official logo) — no separate OpenCV window. It loads the exact model
just compiled (`outputs/model.pt`) and runs it on a live camera stream, showing
the **raw sensor feed beside the denoised output** in real time — the on-device
proof that the optimization actually cleans up the low-light camera. The header
shows which camera is active, and themed stat chips report per-frame latency,
throughput (FPS), and a live noise-reduction percentage (high-frequency noise of
the raw vs denoised frame). **SAVE SNAPSHOT** writes the current raw-vs-denoised
frame to `outputs/live_preview.png`. (You can also run it standalone with
`python live.py`, which uses an OpenCV window.)

Camera backends are auto-detected (override with `--source`):

1. **picamera2** — the Raspberry Pi CSI camera (usually **already on Pi OS** — no
   `sudo apt` needed; recreate the venv with `--system-site-packages`).
2. **rpicam-vid** — CSI via the preinstalled `rpicam-vid` / `libcamera-vid` CLI
   when picamera2 isn't importable (also no extra apt).
3. **OpenCV** — USB / V4L2 webcam (`--source opencv --camera-index N`).
4. **simulated** — synthetic low-light stream for dev machines with no camera.

**No sudo?** On the Pi run `python pi_camera_check.py` — it tells you exactly
what works and how to fix the venv. Most common fix (picamera2 already installed
on the image):

```bash
deactivate
rm -rf .venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python pi_camera_check.py
```

If picamera2 truly isn't on the image and you can't use apt, try
`pip install -r requirements-pi.txt` (pip-only wheels for Bookworm).

In the GUI view, click **CLOSE** (or press `ESC`) to stop. With the standalone
`python live.py`, press `q` or `ESC` in the window; on a headless box it saves a
side-by-side sample to `outputs/live_preview.png` instead of opening a window.

On Windows the webcam is opened via the DirectShow backend (the default MSMF
backend often opens the device but never delivers frames); the opener probes
camera indices 0–2 and verifies a real frame arrives before using a camera, so
it only falls back to the simulated stream when there is genuinely no camera.

The Level-3 options are also **contextual**: pick a model family first and only
the parameters that apply appear (e.g. NAFNet and Restormer hide the activation
and conv-type rows because they use a built-in SimpleGate / transposed-attention
graph instead).

### Run history (don't re-run tests you've already done)

Every compile and sweep is **automatically archived** to `outputs/history/` — a
timestamped snapshot folder (the trained `model.pt`, `summary.json`, the
validation panel, ONNX / device artifact and any package `.zip`) plus a one-line
record in `outputs/history/index.jsonl`. Nothing is overwritten, so past results
*and* models stay around.

Open it from the **HISTORY** button (on the first wizard step and on the results
screen). Each past run shows its profile, target chip, sensor and key metrics
(PSNR in→out, FPS, latency, fitness/grade), with actions to:

- **Open folder** / **View panel** — inspect the saved artifacts and before/after image.
- **Use for live** — load that exact model as `outputs/model.pt` and jump straight
  into live camera testing — no recompile.
- **Load config** — reload that run's configuration into the wizard.

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

4. **Resolution vs TOPS scaling** — `outputs/resolution_tops_scaling.png` plots
   how effective throughput scales with input pixels for each Pi-class target
   (Pi 5 CPU, Hailo-8, DeepX); peak TOPS shown as dashed lines.

5. **Pareto fitness report** — a scorecard balancing image quality, latency and
   INT8 robustness into a single `FINAL PARETO FITNESS SCORE`.

6. **Target suitability matrix** — a cross-chip verdict (`✓ SUITABLE` /
   `▲ WITH CAVEATS` / `✗ NOT RECOMMENDED`) telling you whether this exact model
   is deployable on each Raspberry Pi-class target (Pi 5 CPU, Hailo-8, DeepX),
   derived from each chip's specs: precision, native ops, on-chip SRAM budget
   (with tiling), and estimated FPS. Shown in the terminal report, in the GUI
   results screen, and in `outputs/summary.json` (`targets`).

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
# (use a family that has a pickable activation, e.g. dncnn/cnn/unet)
python run_demo.py --hardware deepx --activation gelu --model-family dncnn

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
