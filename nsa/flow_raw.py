"""Conditional rectified-flow restoration in packed RAW.

Why this (and not another NAFNet + loss bake-off)
-------------------------------------------------
Regression denoisers predict E[clean | noisy] — the posterior *mean* — which is
soft under heavy gain (the middle column of the user's panels). Generative
conditional flow matching learns a velocity field from noise → clean **given**
the noisy observation, then **samples**. Samples sit on the sharp data manifold
instead of the blurry mean.

Training uses the same lab setup the user already has:
  * condition y  = one real burst frame (motion-safe at deploy: single frame)
  * target x0    = multi-frame packed-RAW average (sharp GT from static bursts)
  * many y's per scene from the full burst (diversity they asked for)

Inference (deploy under motion): one noisy DNG → sample with a few ODE steps.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gt_match import (
    BurstScene, RawUNet, _augment, _to_tensor, build_scene_cache, discover_bursts,
    load_packed_any, make_synthetic_burst_dir,
)
from .models import _NAFBlock
from .raw_domain import packed_to_rgb


# ---------------------------------------------------------------------------
# Time embedding + conditional velocity network
# ---------------------------------------------------------------------------

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1], shape (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalFlowNet(nn.Module):
    """Velocity field v_θ(x_t, t | y): concat(x_t, y) + FiLM from time.

    Predicts v = x1 - x0 along the rectified-flow path (noise → clean),
    conditioned on the noisy packed RAW ``y``.
    """

    def __init__(self, base_channels: int = 48, depths=(2, 2, 4, 2)):
        super().__init__()
        c = base_channels
        self.time_emb = SinusoidalTimeEmb(c)
        self.time_mlp = nn.Sequential(
            nn.Linear(c, c * 4), nn.SiLU(), nn.Linear(c * 4, c * 2),
        )
        # 4 (x_t) + 4 (y) = 8 input channels
        self.head = nn.Conv2d(8, c, 3, padding=1)
        self.enc1 = nn.Sequential(*[_NAFBlock(c) for _ in range(depths[0])])
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[_NAFBlock(c * 2) for _ in range(depths[1])])
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.mid = nn.Sequential(*[_NAFBlock(c * 4) for _ in range(depths[2])])
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(*[_NAFBlock(c * 2) for _ in range(depths[3])])
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(*[_NAFBlock(c) for _ in range(depths[0])])
        self.tail = nn.Conv2d(c, 4, 3, padding=1)
        # FiLM at bottleneck
        self.film = nn.Linear(c * 2, c * 8)  # scale/shift for c*4 mid channels

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.view(-1)
        temb = self.time_mlp(self.time_emb(t))          # (B, 2c)
        h = self.head(torch.cat([x_t, y], dim=1))
        e1 = self.enc1(h)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        # FiLM
        film = self.film(temb).view(t.shape[0], -1, 1, 1)
        gamma, beta = film.chunk(2, dim=1)
        m = m * (1 + gamma) + beta
        d2 = self.dec2(self.up2(m) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        return self.tail(d1)


# ---------------------------------------------------------------------------
# Rectified flow: train + sample
# ---------------------------------------------------------------------------

def rf_train_step(net: ConditionalFlowNet, clean: torch.Tensor, noisy: torch.Tensor,
                  rng: torch.Generator | None = None,
                  w_x0: float = 0.5, w_edge: float = 0.15) -> torch.Tensor:
    """Rectified-flow matching + light x0/edge aux (still generative at sample).

    Path: x_t = (1-t)·clean + t·ε   (ε ~ N(0,I)), conditioned on noisy ``y``.
    Target velocity: v = ε - clean.
    Aux: reconstruct x0 = x_t - t·v from the predicted velocity and match clean
    edges — sharpens samples without collapsing infer to an MMSE regressor
    (inference still starts from fresh noise).
    """
    b = clean.shape[0]
    device = clean.device
    eps = torch.randn_like(clean)
    t = torch.rand(b, device=device, dtype=clean.dtype)
    # avoid t≈0 numerical junk in x0 reconstr
    t = t.clamp(0.02, 1.0)
    t_ = t.view(b, 1, 1, 1)
    x_t = (1.0 - t_) * clean + t_ * eps
    v_tgt = eps - clean
    v_pred = net(x_t, t, noisy)
    loss = F.mse_loss(v_pred, v_tgt)
    if w_x0 > 0:
        x0_pred = x_t - t_ * v_pred
        loss = loss + w_x0 * F.mse_loss(x0_pred, clean)
        if w_edge > 0:
            # gradient match — fights soft x0 predictions
            pdx = x0_pred[..., :, 1:] - x0_pred[..., :, :-1]
            pdy = x0_pred[..., 1:, :] - x0_pred[..., :-1, :]
            tdx = clean[..., :, 1:] - clean[..., :, :-1]
            tdy = clean[..., 1:, :] - clean[..., :-1, :]
            loss = loss + w_edge * (
                (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()
            )
    return loss


@torch.no_grad()
def rf_sample(net: ConditionalFlowNet, noisy: torch.Tensor, *,
              steps: int = 10, stochastic: bool = False,
              temperature: float = 1.0) -> torch.Tensor:
    """Integrate noise → clean conditioned on ``noisy`` (Euler).

    ``stochastic=True`` adds a little noise each step (helps escape mean-seeking
    Euler discretisation); ``temperature`` scales the initial noise.
    """
    net.eval()
    b = noisy.shape[0]
    device = noisy.device
    x = torch.randn_like(noisy) * temperature
    dt = 1.0 / max(int(steps), 1)
    for i in range(steps):
        t_val = 1.0 - i * dt
        t = torch.full((b,), t_val, device=device, dtype=noisy.dtype)
        v = net(x, t, noisy)
        x = x - dt * v                       # move toward t=0 (clean)
        if stochastic and i < steps - 1:
            x = x + (0.15 * math.sqrt(dt)) * torch.randn_like(x)
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def rf_sample_heun(net: ConditionalFlowNet, noisy: torch.Tensor, *,
                   steps: int = 8) -> torch.Tensor:
    """Heun (improved Euler) — sharper / stabler than plain Euler at few steps."""
    net.eval()
    b = noisy.shape[0]
    device = noisy.device
    x = torch.randn_like(noisy)
    dt = 1.0 / max(int(steps), 1)
    for i in range(steps):
        t_val = 1.0 - i * dt
        t_next = 1.0 - (i + 1) * dt
        t = torch.full((b,), t_val, device=device, dtype=noisy.dtype)
        v1 = net(x, t, noisy)
        x_euler = x - dt * v1
        if i == steps - 1:
            x = x_euler
            break
        t2 = torch.full((b,), t_next, device=device, dtype=noisy.dtype)
        v2 = net(x_euler, t2, noisy)
        x = x - 0.5 * dt * (v1 + v2)
    return torch.clamp(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Training loop on real / synth bursts
# ---------------------------------------------------------------------------

@dataclass
class FlowTrainConfig:
    bursts_root: str
    out_dir: str = "outputs/flow_raw"
    gains: list[int] = field(default_factory=lambda: [128, 256, 512])
    scenes: list[str] | None = None
    gt_frames: int = 100
    steps: int = 12000
    crop: int = 192
    batch: int = 4
    lr: float = 2e-3
    base_channels: int = 48
    sample_steps: int = 10
    seed: int = 662
    device: str = "cuda"
    eval_every: int = 500


def train_flow_raw(cfg: FlowTrainConfig, log=print) -> dict:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        log("CUDA unavailable — CPU (use AI server for real training)")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenes = discover_bursts(
        cfg.bursts_root, gains=cfg.gains, scenes=cfg.scenes,
        gt_frames=cfg.gt_frames, min_frames=max(16, cfg.gt_frames // 4),
    )
    if not scenes:
        raise FileNotFoundError(f"No bursts under {cfg.bursts_root}")
    log(f"found {len(scenes)} scenes | conditional FLOW (noise→clean | noisy RAW)")
    cache = build_scene_cache(scenes, log=log)

    net = ConditionalFlowNet(base_channels=cfg.base_channels).to(device)
    npar = sum(p.numel() for p in net.parameters())
    log(f"ConditionalFlowNet ch={cfg.base_channels} params={npar:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    warmup = max(1, cfg.steps // 10)

    def lr_at(i):
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, cfg.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    net.train()
    t0 = time.time()
    history = []

    for step in range(cfg.steps):
        cleans, noisies = [], []
        for _ in range(cfg.batch):
            entry = cache[rng.randrange(len(cache))]
            pool = entry["train_files"]
            y_path = pool[rng.randrange(len(pool))]
            noisy = load_packed_any(y_path)
            gt = entry["gt"]
            h, w = gt.shape[:2]
            c = min(cfg.crop, h, w)
            y0 = int(np_rng.integers(0, h - c + 1))
            x0 = int(np_rng.integers(0, w - c + 1))
            n_c = noisy[y0:y0 + c, x0:x0 + c]
            g_c = gt[y0:y0 + c, x0:x0 + c]
            nt = _to_tensor(n_c).squeeze(0)
            gt_t = _to_tensor(g_c).squeeze(0)
            (nt,), gt_t = _augment([nt], gt_t, rng)
            noisies.append(nt.unsqueeze(0))
            cleans.append(gt_t.unsqueeze(0))
        yb = torch.cat(noisies, 0).to(device)
        xb = torch.cat(cleans, 0).to(device)

        opt.zero_grad(set_to_none=True)
        loss = rf_train_step(net, xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == cfg.steps - 1:
            log(f"  step {step:5d}/{cfg.steps}  flow_loss {loss.item():.5f}  "
                f"{(time.time() - t0) / max(1, step + 1):.2f}s/it")
            history.append({"step": step, "loss": float(loss.item())})

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            metrics = eval_flow(net, cache, device=device,
                                sample_steps=cfg.sample_steps, log=log)
            (out / "metrics_latest.json").write_text(json.dumps(metrics, indent=2))

    ckpt = out / "flow_raw.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "base_channels": cfg.base_channels,
        "sample_steps": cfg.sample_steps,
        "gt_frames": cfg.gt_frames,
        "config": dict(cfg.__dict__),
    }, ckpt)
    log(f"saved {ckpt}")

    metrics = eval_flow(net, cache, device=device,
                        sample_steps=cfg.sample_steps, log=log)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    write_flow_panel(net, cache, out / "panel.png", device=device,
                     sample_steps=cfg.sample_steps)
    return {"checkpoint": str(ckpt), "metrics": metrics, "out_dir": str(out)}


def _hf(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x, 0)).mean() + np.abs(np.diff(x, 1)).mean())


@torch.no_grad()
def eval_flow(net, cache, *, device, sample_steps: int, log=print) -> dict:
    net.eval()
    results = {}
    for entry in cache:
        sc: BurstScene = entry["scene"]
        gt = entry["gt"]
        pool = entry["train_files"]
        if not pool:
            continue
        noisy = load_packed_any(pool[len(pool) // 2])
        y = _to_tensor(noisy).to(device)
        # generative sample
        pred = rf_sample_heun(net, y, steps=sample_steps)
        pred = pred.cpu().numpy()[0].transpose(1, 2, 0)
        # regression baseline: just mild blur of noisy (proxy for MMSE soft look)
        import cv2
        blur = np.stack([cv2.GaussianBlur(noisy[..., i], (0, 0), 1.8)
                         for i in range(4)], -1)
        h, w = gt.shape[:2]
        m = min(8, h // 8, w // 8)
        sl = (slice(m, h - m or None), slice(m, w - m or None))
        def psnr(a, b):
            mse = float(np.mean((a - b) ** 2))
            return 10.0 * math.log10(1.0 / max(mse, 1e-12))
        p_flow = psnr(pred[sl], gt[sl])
        p_blur = psnr(blur[sl], gt[sl])
        sharp_flow = _hf(pred[sl]) / max(_hf(gt[sl]), 1e-8)
        sharp_blur = _hf(blur[sl]) / max(_hf(gt[sl]), 1e-8)
        results[sc.name] = {
            "psnr_flow": p_flow,
            "psnr_blur_proxy": p_blur,
            "sharp_ratio_flow": sharp_flow,
            "sharp_ratio_blur": sharp_blur,
            "flow_sharper_than_blur": sharp_flow > sharp_blur + 0.05,
        }
        log(f"  eval {sc.name}: flow PSNR {p_flow:.2f} sharp={sharp_flow:.3f} | "
            f"blur-proxy PSNR {p_blur:.2f} sharp={sharp_blur:.3f}")
    net.train()
    return results


@torch.no_grad()
def write_flow_panel(net, cache, path: Path, *, device, sample_steps: int,
                     display_gain: float = 8.0):
    from PIL import Image
    import cv2
    net.eval()
    strips = []
    for entry in cache[:5]:
        gt = entry["gt"]
        pool = entry["train_files"]
        if not pool:
            continue
        noisy = load_packed_any(pool[len(pool) // 2])
        y = _to_tensor(noisy).to(device)
        pred = rf_sample_heun(net, y, steps=sample_steps)
        pred = pred.cpu().numpy()[0].transpose(1, 2, 0)
        blur = np.stack([cv2.GaussianBlur(noisy[..., i], (0, 0), 1.8)
                         for i in range(4)], -1)
        # columns: noisy | blur-proxy (regression look) | FLOW sample | GT
        strip = np.concatenate([
            packed_to_rgb(noisy, display_gain),
            packed_to_rgb(blur, display_gain),
            packed_to_rgb(pred, display_gain),
            packed_to_rgb(gt, display_gain),
        ], axis=1)
        strips.append(strip)
    if strips:
        img = (np.clip(np.concatenate(strips, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path)
    net.train()


def load_flow_model(ckpt: Path | str, device: str = "cpu") -> tuple[ConditionalFlowNet, dict]:
    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    net = ConditionalFlowNet(base_channels=int(blob.get("base_channels", 48)))
    net.load_state_dict(blob["state_dict"])
    meta = {
        "sample_steps": int(blob.get("sample_steps", 10)),
        "base_channels": int(blob.get("base_channels", 48)),
    }
    return net.to(device).eval(), meta


@torch.no_grad()
def infer_flow(net: ConditionalFlowNet, path: Path | str, *, device="cpu",
               steps: int = 10, heun: bool = True) -> np.ndarray:
    packed = load_packed_any(Path(path))
    y = _to_tensor(packed).to(device)
    if heun:
        out = rf_sample_heun(net, y, steps=steps)
    else:
        out = rf_sample(net, y, steps=steps, stochastic=True)
    return out.cpu().numpy()[0].transpose(1, 2, 0)


def proof_flow_synth(out_dir: Path | str, *, steps: int = 600,
                     device: str = "cpu") -> dict:
    """Train a tiny conditional flow on synth bursts; must beat blur on sharpness."""
    out_dir = Path(out_dir)
    bursts = out_dir / "_synth"
    make_synthetic_burst_dir(bursts, n_frames=64, h=128, w=128, gain=0.12)
    cfg = FlowTrainConfig(
        bursts_root=str(bursts), out_dir=str(out_dir),
        gains=[512], gt_frames=32, steps=steps, crop=96, batch=2,
        lr=2e-3, base_channels=32, sample_steps=8,
        device=device, eval_every=max(200, steps // 2),
    )
    result = train_flow_raw(cfg)
    # Held-out comparison
    import cv2
    files = sorted((bursts / "synth_scene" / "ag512").glob("*.npy"))
    gt = np.mean([load_packed_any(p) for p in files[:32]], 0)
    noisy = load_packed_any(files[50])
    net, meta = load_flow_model(result["checkpoint"], device=device)
    pred = infer_flow(net, files[50], device=device, steps=meta["sample_steps"])
    blur = np.stack([cv2.GaussianBlur(noisy[..., i], (0, 0), 1.8) for i in range(4)], -1)

    def psnr(a, b):
        return float(10 * math.log10(1 / max(np.mean((a - b) ** 2), 1e-12)))

    metrics = {
        "psnr_flow": psnr(pred, gt),
        "psnr_blur": psnr(blur, gt),
        "sharp_flow": _hf(pred) / max(_hf(gt), 1e-8),
        "sharp_blur": _hf(blur) / max(_hf(gt), 1e-8),
        "checkpoint": result["checkpoint"],
    }
    metrics["pass_sharper"] = metrics["sharp_flow"] > metrics["sharp_blur"] + 0.03
    metrics["pass_psnr"] = metrics["psnr_flow"] > metrics["psnr_blur"] + 0.3
    metrics["pass"] = bool(metrics["pass_sharper"] or metrics["pass_psnr"])
    (out_dir / "proof_flow.json").write_text(json.dumps(metrics, indent=2))
    from PIL import Image
    panel = np.concatenate([
        packed_to_rgb(noisy, 1.0),
        packed_to_rgb(blur, 1.0),
        packed_to_rgb(pred, 1.0),
        packed_to_rgb(gt, 1.0),
    ], 1)
    Image.fromarray((np.clip(panel, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        out_dir / "proof_flow_panel.png")
    return metrics
