# Live IMX662 denoise on the Raspberry Pi + Hailo-10H AI HAT

This is the launch playbook for the on-Pi Hailo path. Everything runs on the
Pi; this AI server is used only to build the calibration set and compile the
HEF with the Hailo Dataflow Compiler.

> **Board note:** the AI HAT on this rig is the **Hailo-10H** (40 TOPS INT8,
> integrated LPDDR4X). The DFC compile step targets `--hw-arch hailo10h` and
> requires Hailo DFC **≥ 5.x** — a Hailo-8-only 3.x install will reject the
> arch string. HailoRT / `hailortcli` / `hailo_platform` are arch-agnostic and
> stay as-is on the Pi.

Target metrics:

| Metric      | Target                     |
|-------------|----------------------------|
| Backend     | `hailo` (device present)   |
| Resolution  | packed 968×544 (full FOV)  |
| FPS         | ≥ 30                       |
| Latency e2e | < 33 ms                    |
| Telemetry   | backend / res / FPS / latency / dropped frames rendered on the video |

## One-shot deploy

Once P1 and P2 below are green:

```bash
PI_HEF=outputs/cfm_deploy/hailo10h/cfm_student_real.hef \
scripts/deploy_live_hailo_pipeline.sh
```

The script SSHes to the Pi, verifies `/dev/hailo0`, pushes the HEF, launches
`pi_live_cfm.py --backend hailo --stream-port 8890`, and prints the viewer URL
(`http://10.3.35.18:8890/`). Ctrl-C tears the Pi-side process down.

Env vars: `PI_IP` (default `10.3.35.18`), `PORT`, `MAX_SIDE` (`968`),
`TILE` (`256`), `HCG` (`1`), `ANALOG_GAIN` (`16`), `SHUTTER_US` (`30000`).

## Telemetry — how it's measured

`pi_live_cfm.py` exposes `/api/stats` on the same port as the MJPEG stream.
Values shown in the right-hand panel and burned into the video overlay:

- **preview_fps** — 1000 / EMA of the whole capture→display loop wall time.
- **denoise_fps** — 1000 / last Hailo/ORT inference wall time.
- **preview_ms / denoise_ms** — per-frame latency components (`time.perf_counter`
  around `cam.capture_packed()` and `runtime.run(stacked)` respectively).
- **backend / infer_backend** — reflects the current runtime (`hailo` when the
  AI HAT is present and the HEF loaded; `cpu` otherwise). Source: `nsa.hailo_live.probe_hailo`.
- **hailo_device / hef** — device serial + loaded HEF path from `hailortcli`.
- **packed** — WxH of the packed frame currently fed to the model. With
  `--max-side 968 --tile 256` this reads `968x544` and tiling handles the
  256-static graph.

Glass-to-glass p50/p95 latency is derived by adding a monotonically
increasing frame counter to the JPEG buffer and measuring the delta between
capture timestamp (`preview_ms` start) and the display timestamp (browser
receive) via a small JS probe on the viewer page.

## Bring-up before DFC

The Hailo Dataflow Compiler needs a Developer-Zone signup. Until that's
approved, use pre-compiled Model Zoo HEFs to shake out the Pi-side
HailoRT plumbing (device probe, HEF load, vstream I/O). Real numeric
quality still needs the CFM HEF.

### Stand-in HEF (no signup)

```bash
bash scripts/fetch_hailo_zoo_hef.sh
# → outputs/hailo_zoo_hef/{dncnn_color_blind,dncnn3,zero_dce}.hef  (hailo10h)
scp outputs/hailo_zoo_hef/dncnn3.hef \
    pi@10.3.35.18:~/RPi-Hardware-NSA-/outputs/hailo_zoo_hef/
ssh pi@10.3.35.18 \
    'cd ~/RPi-Hardware-NSA- && .venv/bin/python pi_live_cfm.py \
        --backend hailo \
        --hef outputs/hailo_zoo_hef/dncnn3.hef \
        --stream-port 8890'
```

The stand-in HEFs are 321×481 fixed-shape RGB uint8 — **NOT** our 16-ch
packed-Bayer CFM student (1×16×256×256 f32). Denoise output will look
wrong; that's expected. What you're validating is `/dev/hailo0` bring-up,
HEF load, one round-trip infer, teardown, and telemetry.

### Signup → real HEF

1. Request access: <https://hailo.ai/developer-zone/request-access/>.
2. In the "target application" field, paste:
   > Real-time INT8 low-light RAW denoiser on Raspberry Pi 5 + Hailo-10H
   > AI HAT, custom NAFNet student, 16-ch packed Bayer 256×256, targeting
   > 30 FPS &lt;33 ms.
3. Once approved, download **Hailo AI SW Suite ≥ v5.4** (includes DFC —
   the standalone DFC 3.x is Hailo-8-only and rejects `--hw-arch hailo10h`).
4. Install per the SW-Suite guide (`./hailo_ai_sw_suite_installer.sh`).
5. One-liner:

   ```bash
   bash scripts/compile_hailo_hef.sh
   # → outputs/cfm_deploy/hailo10h/cfm_student_real.hef
   ```

   Auto-detects `hailomz` vs raw `hailo` CLI; YAML + `.alls` live under
   `outputs/cfm_deploy/hailo10h/`.

## P1 — kernel + driver fix

The Pi shipped booting the 16 KB-page `kernel8.img`; the DKMS-built
`hailo_pci` links against 4 KB-page kernel headers → module can't load →
`/dev/hailo0` is missing.

```bash
scp scripts/pi_fix_kernel_for_hailo.sh pi@10.3.35.18:/tmp/
ssh pi@10.3.35.18 'sudo bash /tmp/pi_fix_kernel_for_hailo.sh'
# waits for reboot (~30 s)
scp scripts/pi_verify_hailo.sh pi@10.3.35.18:/tmp/
ssh pi@10.3.35.18 'bash /tmp/pi_verify_hailo.sh'
```

The verify script prints kernel version, `lsmod | grep hailo`,
`ls /dev/hailo0`, `hailortcli fw-control identify`, and `dmesg | grep hailo`
— exactly the acceptance data P1 asks for.

## P2 — real HEF compile

Requires the Hailo Dataflow Compiler on this AI server. The compiler is not
on PyPI and not shipped with `hailo-all` on the Pi; download from
[Hailo Developer Zone](https://hailo.ai/developer-zone/) (`hailo_ai_sw_suite`
or `hailo-dataflow-compiler` wheels).

Calibration set is already built:

- 500 tiles at (1, 16, 256, 256) float32, balanced 100/100/100/100/100 across
  `dark / 5000k_10l / 25l / 100l / 500l` scenes from the CTT rsync corpus.
- Location: `outputs/cfm_deploy/hailo10h/calibration/`
- Manifest: `outputs/cfm_deploy/hailo10h/calibration_manifest.json`

Rebuild:

```bash
.venv/bin/python scripts/build_hailo_calib.py \
    --dngs datasets/pi_ctt/_incoming \
    --out  outputs/cfm_deploy/hailo10h/calibration \
    --num 500 --tile 256 --temporal 4
```

Compile (after DFC install):

```bash
scripts/compile_hailo_hef.sh
# → outputs/cfm_deploy/hailo10h/cfm_student_real.hef
```

The script auto-detects `hailomz` (Model Zoo) vs the raw `hailo` CLI and
picks the right compile flow (parser → optimize → compiler).

Pre-compile sanity (optional, DFC-free — uses ONNX QDQ as a proxy):

```bash
.venv/bin/python scripts/emulate_hailo_int8_quality.py \
    --onnx outputs/cfm_deploy/hailo10h/cfm_student_static.onnx \
    --calib outputs/cfm_deploy/hailo10h/calibration \
    --n-calib 128 --n-eval 32
# → outputs/cfm_deploy/hailo10h/cfm_student_static_int8.onnx
# → outputs/cfm_deploy/hailo10h/quantization_report.json  (PSNR/MAE deltas)
```

**Input-tensor-name note (2026-07-17 fix):** the ONNX exporter uses
`input_names=["packed"]`, so `cfm_student.yaml` and `scripts/compile_hailo_hef.sh`
now feed `{"packed": [1,16,256,256]}` to the parser (older revs used `"input"`
and would fail with `KeyError: 'input'` on the start-node shape lookup).

## Fallback: server-offload (Pi ↔ AI server GPU)

If either P1 or P2 stalls, the earlier server-offload work is still on disk:

- `scripts/bench_ov.py` benchmarks OpenVINO CPU/NPU on this box.
- OpenVINO NPU is enabled by exporting
  `LD_LIBRARY_PATH="$PWD/.npu_driver/prefix/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"`
  (OV then reports `['CPU', 'NPU']`).

Baseline numbers on this AI server (Intel Core Ultra 7 365; no NVIDIA GPU is
attached — this machine is Panther-Lake-class integrated only):

| Model                                 | Shape (H×W) | CPU (LATENCY hint) | NPU        |
|---------------------------------------|-------------|---------------------|------------|
| `outputs/cfm_deploy/cfm_student.onnx` | 968×544     | 1262 ms / 0.8 FPS   | 217 ms / 4.6 FPS |
| ``                                    | 484×272     | 331 ms  / 3.0 FPS   |  50 ms / 19.8 FPS |
| ``                                    | 240×144     |  62 ms  / 16.0 FPS  |  14 ms / 69.6 FPS |

Interpretation: on this specific AI server the NPU can only reach the 30 FPS
target below ~400×220 packed; and the WiFi link between server and Pi is
~52 Mbit/s (Pi→server, measured with a raw-socket iperf-alike), which caps
Pi→server frame transport at ~1.5 FPS for fp16 968×544×4 (4.2 MB/frame).
This is why the Hailo-on-Pi path is now the primary route.
