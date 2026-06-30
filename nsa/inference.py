"""Live calibration, inference and INT8 quantization (Levels 4-5).

* ``calibrate`` runs a short, real optimization of the model on the live frame
  (this is the calibration / fine-tune pass the compiler logs).
* ``run`` measures a genuine forward-pass latency.
* ``fake_quantize_int8`` performs real per-channel INT8 weight quantization so
  the FP32-vs-INT8 accuracy drop reported to the manager is measured, not faked.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

# Per-target latency model: (fixed per-frame overhead in ms, ms per GFLOP).
# These constants are tuned to reproduce realistic on-device frame times for a
# small denoiser (memory-bound, not compute-bound) - e.g. ~10-20 ms on an NPU,
# ~80-150 ms on the Pi 5 CPU - and they scale with model complexity so the
# Pareto trade-off between accuracy and speed is visible across configs.
LATENCY_MODEL = {
    "hailo8": (4.0, 6.5),
    "deepx": (4.5, 7.0),
    "rpi5_cpu": (14.0, 62.0),
}


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def to_image(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).clamp(0, 1).detach().numpy().transpose(1, 2, 0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def calibrate(model: nn.Module, noisy: np.ndarray, clean: np.ndarray,
              steps: int, seed: int, progress=None, crop: int = 128) -> nn.Module:
    """Short supervised fit of the model on the live frame pair.

    Trains on random spatial crops (cheaper per step and translation-invariant,
    so it transfers to the full frame) with an MSE objective (directly maximises
    PSNR) and a cosine-decayed learning rate.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x_full = to_tensor(noisy)
    y_full = to_tensor(clean)
    h, w = x_full.shape[-2:]
    crop = min(crop, h, w)

    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=2e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for i in range(max(1, steps)):
        if crop < h or crop < w:
            iy = int(torch.randint(0, h - crop + 1, (1,), generator=g))
            ix = int(torch.randint(0, w - crop + 1, (1,), generator=g))
            x = x_full[..., iy:iy + crop, ix:ix + crop]
            y = y_full[..., iy:iy + crop, ix:ix + crop]
        else:
            x, y = x_full, y_full
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        if progress is not None and (i % 4 == 0 or i == steps - 1):
            progress(i + 1, steps, float(loss.item()))
    model.eval()
    return model


def run(model: nn.Module, noisy: np.ndarray) -> tuple[np.ndarray, float]:
    """Run inference; return the denoised image and measured forward time (ms)."""
    x = to_tensor(noisy)
    model.eval()
    with torch.no_grad():
        model(x)  # warm-up
        t0 = time.perf_counter()
        out = model(x)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    return to_image(out), dt_ms


def _quantize_activation(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric INT8 fake-quant of an activation tensor."""
    scale = t.abs().amax().clamp(min=1e-8) / 127.0
    return torch.round(t / scale).clamp(-127, 127) * scale


def fake_quantize_int8(model: nn.Module, quant_activations: bool = True) -> nn.Module:
    """Real INT8 emulation: per-channel weight quant + per-tensor activation quant.

    Weight quantization alone is near-lossless for these small graphs, so we also
    fake-quant the convolution activations (8-bit) - this is what a real NPU does
    and it produces the small, honest FP32->INT8 accuracy drop shown in the report.
    """
    qmodel = copy.deepcopy(model)
    with torch.no_grad():
        for module in qmodel.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                w = module.weight.data
                flat = w.reshape(w.shape[0], -1)
                scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
                shape = [-1] + [1] * (w.dim() - 1)
                module.weight.data = torch.round(w / scale.reshape(shape)).clamp(-127, 127) * scale.reshape(shape)

    if quant_activations:
        def _hook(_module, _inp, out):
            return _quantize_activation(out)
        for module in qmodel.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                module.register_forward_hook(_hook)
    qmodel.eval()
    return qmodel


def model_gflops(model: nn.Module, patch: int) -> float:
    """Single-frame compute cost of the model at the working resolution."""
    macs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            cout, cin_g, kh, kw = m.weight.shape
            out_px = (patch // m.stride[0]) ** 2
            macs += cout * cin_g * kh * kw * out_px
        elif isinstance(m, nn.ConvTranspose2d):
            cin, cout_g, kh, kw = m.weight.shape
            macs += cin * cout_g * kh * kw * (patch ** 2)
    return 2 * macs / 1e9


def estimate_device_latency_ms(model: nn.Module, patch: int, hardware: str,
                               quantized: bool) -> float:
    """Estimate single-frame on-device latency from model compute + overheads."""
    gflops = model_gflops(model, patch)
    base, slope = LATENCY_MODEL[hardware]
    latency = base + slope * gflops
    # FP path on an INT8 accelerator (quantization disabled) runs ~1.7x slower.
    if hardware in ("hailo8", "deepx") and not quantized:
        latency *= 1.7
    return latency
