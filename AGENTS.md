# AGENTS.md

## Cursor Cloud specific instructions

NSA is a **single-product, CPU-only Python toolchain** for hardware-aware RAW image
denoising (a "6-Level Optimization Stack"). There is no web server, database, or
long-running service — everything is a batch CLI run or a Tkinter desktop GUI that
runs to completion and writes artifacts to `outputs/`.

### Environment / running
- Python deps live in a project-local venv at `.venv` (created by the update script,
  which runs `pip install -r requirements.txt`). Activate with `source .venv/bin/activate`.
- The GUI needs the system Tk package (`python3-tk`) and a virtual X display; both are
  already present in this environment (display `:1`, provided by tigervnc). These are
  system packages, not pip deps, so they are intentionally **not** in the update script.
- First CLI/GUI run downloads ~233 MB of LPIPS AlexNet weights into `~/.cache/torch`
  (cached afterward), so the very first run is slower and needs network access.

### How to run (core product)
- Headless CLI (fastest end-to-end check): `python run_demo.py --no-window --steps 70`
  — runs Levels 1-6 and writes `outputs/{exported_model.onnx,hardware_ready.hef,validation_panel.png,summary.json}`.
- Desktop GUI (flagship demo): `DISPLAY=:1 python nsa_gui.py`, then click **▶ COMPILE**.
  Note the GUI's default compile uses 300 calibration steps and takes a few minutes;
  the CLI `--steps` flag is the quick path.
- Other entry points and all flags are documented in `README.md` (e.g. `search.py`
  Pareto sweep, `live.py` live camera, `hf_search.py`, `deploy.py`).

### Lint / tests
- There is **no** automated test suite and **no** lint/formatter config in this repo.
  Validation is done by running the pipeline and inspecting `outputs/`.

### Gotchas
- `torch`'s ONNX exporter (dynamo path in modern torch) requires `onnxscript`; without
  it the Level 6 `exported_model.onnx` step is silently skipped with a misleading
  "'onnx' package unavailable" warning (the pipeline still finishes and writes the
  hardware artifact). `onnxscript` is pinned in `requirements.txt` for this reason.
- Raspberry Pi / CSI-camera / SSH features (`requirements-pi.txt`, `*.ps1`, `nsa/pi_remote.py`)
  are hardware-specific and not needed for local dev; the pipeline runs fully simulated.
