"""Live calibration, inference and INT8 quantization (Levels 4-5).

* ``calibrate`` runs a short, real optimization of the model on the live frame
  (this is the calibration / fine-tune pass the compiler logs).
* ``run`` measures a genuine forward-pass latency.
* ``fake_quantize_int8`` performs real per-channel INT8 weight quantization so
  the FP32-vs-INT8 accuracy drop reported to the manager is measured, not faked.
"""

from __future__ import annotations

import copy
import math
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The loss vocabulary (term names, aliases, default weights) lives in config so
# the CLI/GUI can reason about it without importing torch.
from .config import DEFAULT_LOSS_WEIGHTS, LOSS_TERMS, parse_loss_terms

# Per-target latency model: (fixed per-frame overhead in ms, ms per GFLOP).
# These constants are tuned to reproduce realistic on-device frame times for a
# small denoiser (memory-bound, not compute-bound) - e.g. ~10-20 ms on an NPU,
# ~80-150 ms on the Pi 5 CPU - and they scale with model complexity so the
# Pareto trade-off between accuracy and speed is visible across configs.
LATENCY_MODEL = {
    "hailo8": (4.0, 6.5),
    "deepx": (4.5, 7.0),
    "rpi5_cpu": (14.0, 62.0),
    "intel_npu": (2.0, 2.5),   # OpenVINO on Intel AI Boost (estimate; measured at export)
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


def _metric_nchw(img: np.ndarray) -> torch.Tensor:
    """HxWxC (or HxW) image in [0,1] -> 1xCx H xW float tensor for metrics."""
    arr = np.ascontiguousarray(img).astype(np.float32)
    if arr.max() > 1.5:                       # tolerate 0-255 inputs
        arr = arr / 255.0
    t = torch.from_numpy(arr)
    if t.dim() == 2:
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1).unsqueeze(0)       # -> 1,C,H,W
    return t.clamp(0.0, 1.0)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index (0-1, higher is better).

    Unlike PSNR, SSIM compares local luminance/contrast/structure, so it
    penalises blur and lost texture that a pixel-wise metric would ignore.
    """
    ta, tb = _metric_nchw(a), _metric_nchw(b)
    with torch.no_grad():
        return float(_ssim_index(ta, tb).clamp(-1.0, 1.0).item())


_LPIPS_NET = None


def _get_lpips_net():
    """Lazily build and cache the LPIPS (AlexNet) perceptual network."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips as _lpips_mod          # required dependency
        net = _lpips_mod.LPIPS(net="alex", verbose=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _LPIPS_NET = net
    return _LPIPS_NET


def lpips(a: np.ndarray, b: np.ndarray) -> float:
    """Learned Perceptual Image Patch Similarity (lower is better).

    Compares deep CNN feature activations rather than pixels, so it tracks how
    different two images look to a human and strongly penalises the kind of
    over-smoothing/blur that PSNR happily rewards. Returns a perceptual distance
    (0 = identical); typical denoising values land in ~0.0-0.5.
    """
    net = _get_lpips_net()
    ta, tb = _metric_nchw(a), _metric_nchw(b)
    if ta.shape[1] == 1:                       # LPIPS expects 3 channels
        ta = ta.repeat(1, 3, 1, 1)
    if tb.shape[1] == 1:
        tb = tb.repeat(1, 3, 1, 1)
    ta, tb = ta * 2.0 - 1.0, tb * 2.0 - 1.0    # LPIPS wants [-1, 1]
    with torch.no_grad():
        return float(net(ta, tb).item())


def quality_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Convenience bundle: PSNR (dB), SSIM (0-1) and LPIPS (distance)."""
    return {"psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
            "lpips": lpips(pred, target)}


# ---------------------------------------------------------------------------
# Quantization-Aware Training (true fake-quant-in-the-loop, STE gradients)
# ---------------------------------------------------------------------------
class _STEQuant(torch.autograd.Function):
    """Symmetric INT8 fake-quant with a straight-through-estimator gradient."""

    @staticmethod
    def forward(ctx, x, scale):
        return torch.round(x / scale).clamp(-127, 127) * scale

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None      # STE: gradient passes through the round()


def _fq_weight(w: torch.Tensor) -> torch.Tensor:
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    shape = [-1] + [1] * (w.dim() - 1)
    return _STEQuant.apply(w, scale.reshape(shape))


def _fq_act(t: torch.Tensor) -> torch.Tensor:
    scale = t.detach().abs().amax().clamp(min=1e-8) / 127.0
    return _STEQuant.apply(t, scale)


def enable_qat(model: nn.Module) -> nn.Module:
    """Insert fake-quant nodes into every conv so training sees INT8 rounding.

    Weights are fake-quantized per output channel and activations per tensor,
    both with straight-through gradients, so the optimizer learns weights that
    survive INT8 deployment (this is what recovers the quantization loss).
    """
    def _conv_fwd(self, x):
        wq = _fq_weight(self.weight)
        out = F.conv2d(x, wq, self.bias, self.stride, self.padding,
                       self.dilation, self.groups)
        return _fq_act(out)

    def _convT_fwd(self, x):
        wq = _fq_weight(self.weight)
        out = F.conv_transpose2d(x, wq, self.bias, self.stride, self.padding,
                                 self.output_padding, self.groups, self.dilation)
        return _fq_act(out)

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward
            m.forward = types.MethodType(_conv_fwd, m)
        elif isinstance(m, nn.ConvTranspose2d) and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward
            m.forward = types.MethodType(_convT_fwd, m)
    return model


def disable_qat(model: nn.Module) -> nn.Module:
    """Restore the original (non-fake-quant) conv forwards after QAT training."""
    for m in model.modules():
        if hasattr(m, "_orig_forward"):
            m.forward = m._orig_forward
            del m._orig_forward
    return model


def _charbonnier(pred: torch.Tensor, target: torch.Tensor,
                 eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier (smooth-L1) loss — the denoising-standard objective.

    Robust to outliers and much less blurry than plain MSE, so it recovers
    sharper edges/texture and typically a higher PSNR at the same budget. ``eps``
    controls the L2->L1 transition: smaller is closer to pure L1 (sharper, more
    robust), larger is smoother near zero.
    """
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def _l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error — sharp, robust to outliers, no parameters."""
    return (pred - target).abs().mean()


def _l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error — maximises PSNR directly but tends to blur texture."""
    return ((pred - target) ** 2).mean()


def _huber(pred: torch.Tensor, target: torch.Tensor,
           delta: float = 1.0) -> torch.Tensor:
    """Huber / smooth-L1 loss. ``delta`` is the L2->L1 crossover threshold."""
    return F.smooth_l1_loss(pred, target, beta=max(1e-6, float(delta)))


def _gaussian_window(window: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window, dtype=dtype, device=device) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, window, window).contiguous()


def _ssim_index(pred: torch.Tensor, target: torch.Tensor,
                window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean structural-similarity (SSIM) index over the image (higher is better)."""
    channels = pred.shape[1]
    window = max(3, int(window) | 1)                 # force odd, >=3
    w = _gaussian_window(window, sigma, channels, pred.device, pred.dtype)
    pad = window // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=channels)
    mu2 = F.conv2d(target, w, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, w, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, w, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
                ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)))
    return ssim_map.mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
               window: int = 11) -> torch.Tensor:
    """Structural-dissimilarity loss (1 - SSIM); preserves perceived structure."""
    return 1.0 - _ssim_index(pred, target, window)


def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on horizontal/vertical image gradients (finite differences).

    Matching gradients directly penalises smeared edges, so the network is
    pushed to keep high-frequency detail that a pure pixel loss lets it blur
    away. Cheap, parameter-free and fully differentiable.
    """
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def _perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable LPIPS (AlexNet) perceptual loss.

    Reuses the frozen LPIPS network from the metrics path (its parameters carry
    no gradient, so only the inputs are optimised). Comparing deep feature
    activations rather than pixels strongly penalises the over-smoothing/blur
    that L1 alone tolerates.
    """
    net = _get_lpips_net().to(device=pred.device, dtype=pred.dtype)
    p, t = pred, target
    if p.shape[1] == 1:                        # LPIPS expects 3 channels
        p = p.repeat(1, 3, 1, 1)
        t = t.repeat(1, 3, 1, 1)
    p, t = p * 2.0 - 1.0, t * 2.0 - 1.0        # LPIPS wants [-1, 1]
    return net(p, t).mean()


def _term_loss(term: str, *, charbonnier_eps: float, huber_delta: float,
               ssim_window: int):
    """Return the ``loss(pred, target)`` callable for a single loss term."""
    if term == "l1":
        return _l1
    if term == "l2":
        return _l2
    if term == "charbonnier":
        return lambda p, t: _charbonnier(p, t, charbonnier_eps)
    if term == "huber":
        return lambda p, t: _huber(p, t, huber_delta)
    if term == "ssim":
        return lambda p, t: _ssim_loss(p, t, ssim_window)
    if term == "perceptual":
        return _perceptual_loss
    if term == "edge":
        return _edge_loss
    raise KeyError(term)


def build_loss(name: str = "charbonnier", *,
               charbonnier_eps: float = 1e-3,
               huber_delta: float = 1.0,
               ssim_window: int = 11,
               ssim_weight: float = 0.2,
               weights: dict | None = None):
    """Return a ``loss(pred, target) -> scalar`` callable for the named objective.

    ``name`` is one of the individual terms — ``l1``, ``l2``, ``charbonnier``
    (eps), ``huber`` (delta), ``ssim`` (window), ``perceptual`` (LPIPS), ``edge``
    (gradient-L1) — or a ``+``-joined composite of them (e.g.
    ``l1+perceptual+edge``), summed as ``Σ weight_i · term_i``. A lone term is
    used unscaled; composite weights come from ``weights`` (falling back to
    ``DEFAULT_LOSS_WEIGHTS``). The special ``charbonnier_ssim`` preset keeps its
    ``(1-w)·charbonnier + w·(1-SSIM)`` blend. Unknown names fall back to
    Charbonnier.
    """
    key = (name or "charbonnier").strip().lower()
    if key == "charbonnier_ssim":
        w = float(min(max(ssim_weight, 0.0), 1.0))
        return lambda p, t: ((1.0 - w) * _charbonnier(p, t, charbonnier_eps)
                             + w * _ssim_loss(p, t, ssim_window))

    terms = [t for t in parse_loss_terms(key) if t in LOSS_TERMS] or ["charbonnier"]
    build = lambda term: _term_loss(term, charbonnier_eps=charbonnier_eps,
                                    huber_delta=huber_delta, ssim_window=ssim_window)
    if len(terms) == 1:                       # single term: use it unscaled
        return build(terms[0])

    wmap = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        wmap.update({str(k).strip().lower(): float(v) for k, v in weights.items()})
    parts = [(wmap.get(t, 1.0), build(t)) for t in terms]

    def _composite(p, t):
        total = None
        for wt, fn in parts:
            term_val = wt * fn(p, t)
            total = term_val if total is None else total + term_val
        return total

    return _composite


def _augment_pair(x: torch.Tensor, y: torch.Tensor,
                  g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random dihedral augmentation (8×): h/v flips + 90° rotations.

    Turns a single frame into an effectively much larger, symmetry-complete
    training set so the on-frame fit generalises to the whole image.
    """
    if torch.rand(1, generator=g).item() < 0.5:
        x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
    if torch.rand(1, generator=g).item() < 0.5:
        x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    if k:
        x, y = torch.rot90(x, k, [-2, -1]), torch.rot90(y, k, [-2, -1])
    return x, y


def _sample_batch(tensors, crop: int, batch: int,
                  g: torch.Generator,
                  weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble an augmented minibatch of same-sized crops across frames.

    ``weights`` (one per frame) biases which frames the crops come from —
    used to oversample the hard high-gain / low-light captures.
    """
    c = min([crop] + [min(t[0].shape[-2:]) for t in tensors])
    if weights is not None:
        idxs = torch.multinomial(weights, batch, replacement=True, generator=g)
    else:
        idxs = torch.randint(0, len(tensors), (batch,), generator=g)
    xs, ys = [], []
    for k in range(batch):
        xi, yi = tensors[int(idxs[k])]
        h, w = xi.shape[-2:]
        iy = int(torch.randint(0, h - c + 1, (1,), generator=g))
        ix = int(torch.randint(0, w - c + 1, (1,), generator=g))
        xc = xi[..., iy:iy + c, ix:ix + c]
        yc = yi[..., iy:iy + c, ix:ix + c]
        xc, yc = _augment_pair(xc, yc, g)
        xs.append(xc)
        ys.append(yc)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def _train(model: nn.Module, tensors, steps: int, seed: int, progress,
           crop: int, batch: int, qat: bool, lr: float, loss_fn=None,
           weights=None) -> nn.Module:
    """Shared training loop: configurable loss + 8× aug + warmup-cosine LR + grad clip.

    Gradient clipping and a short warmup keep even deep transpose-conv nets
    (DRUNet / RED-Net) from diverging at an aggressive learning rate, while the
    minibatch of augmented crops gives stable, low-variance gradients. ``loss_fn``
    defaults to Charbonnier (see ``build_loss``). ``weights`` (one per frame)
    biases crop sampling towards the hard high-gain / low-light captures.
    """
    if loss_fn is None:
        loss_fn = _charbonnier
    wt = None
    if weights is not None and len(weights) == len(tensors):
        wt = torch.as_tensor(list(weights), dtype=torch.float32).clamp(min=1e-6)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    steps = max(1, steps)
    warmup = max(1, steps // 10)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * (1.0 - 0.02) + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    if qat:
        enable_qat(model)
    model.train()
    for i in range(steps):
        xb, yb = _sample_batch(tensors, crop, batch, g, weights=wt)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if progress is not None and (i % 4 == 0 or i == steps - 1):
            progress(i + 1, steps, float(loss.item()))
    if qat:
        disable_qat(model)
    model.eval()
    return model


def calibrate(model: nn.Module, noisy: np.ndarray, clean: np.ndarray,
              steps: int, seed: int, progress=None, crop: int = 160,
              qat: bool = False, batch: int = 4, lr: float = 3e-3,
              loss_fn=None) -> nn.Module:
    """Short supervised fit on one frame pair (configurable loss + augmentation)."""
    tensors = [(to_tensor(noisy), to_tensor(clean))]
    return _train(model, tensors, steps, seed, progress, crop, batch, qat, lr, loss_fn)


def calibrate_multi(model: nn.Module, pairs, steps: int, seed: int,
                    progress=None, crop: int = 160, qat: bool = False,
                    batch: int = 4, lr: float = 3e-3, loss_fn=None,
                    weights=None) -> nn.Module:
    """Calibrate across a set of (noisy, clean) frames (batch / multi-image fit).

    Draws an augmented minibatch of crops across all frames each step — the
    "patches across many images" strategy from denoise-hw, with 8× dihedral
    augmentation, a configurable loss (see ``build_loss``) and gradient clipping
    for stability. ``weights`` (one per pair) oversamples the hard high-gain /
    low-light captures (see ``raw_io.training_sample_weights``).
    """
    if not pairs:
        raise ValueError("calibrate_multi needs at least one (noisy, clean) pair")
    tensors = [(to_tensor(n), to_tensor(c)) for n, c in pairs]
    return _train(model, tensors, steps, seed, progress, crop, batch, qat, lr,
                  loss_fn, weights=weights)


def temporal_denoise(model: nn.Module, burst, alpha: float = 0.6):
    """Recursive temporal video denoise over an ordered burst of noisy frames.

    Runs the spatial denoiser per frame, then applies a first-order temporal
    IIR blend ``out_t = a·model(x_t) + (1-a)·out_{t-1}`` (classic low-motion
    video denoising). Returns ``(outputs, per_frame_ms)`` where ``outputs`` is a
    list of denoised RGB frames in the input order.
    """
    model.eval()
    outputs = []
    total_ms = 0.0

    # Video families (ReMoNet/EMVD/MSTMN) carry their own recurrent state, so use
    # their genuine temporal mechanism instead of the generic IIR blend.
    step = getattr(model, "temporal_step", None)
    if callable(step):
        state = None
        with torch.no_grad():
            for noisy in burst:
                t0 = time.perf_counter()
                out_t, state = step(to_tensor(noisy), state)
                total_ms += (time.perf_counter() - t0) * 1000.0
                outputs.append(np.clip(to_image(out_t), 0.0, 1.0))
        return outputs, total_ms / max(1, len(burst))

    prev = None
    with torch.no_grad():
        for noisy in burst:
            spatial, dt = run(model, noisy)
            total_ms += dt
            if prev is None:
                blended = spatial
            else:
                blended = alpha * spatial + (1.0 - alpha) * prev
            blended = np.clip(blended, 0.0, 1.0)
            outputs.append(blended)
            prev = blended
    per_frame = total_ms / max(1, len(burst))
    return outputs, per_frame


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
