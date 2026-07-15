"""Ground-truth-matching RAW burst denoiser.

Why previous single-frame RGB training always looked blurry
----------------------------------------------------------
Pixel losses (L1 / L2 / Charbonnier / SWT) minimise expected error. Under heavy
analogue gain the conditional mean of the clean image given one noisy frame is
*soft* — fine texture is indistinguishable from grain, so the network correctly
(for PSNR) washes it out. That is not a bug in NAFNet; it is the mathematics of
regression.

What actually matches a ~100-frame average
------------------------------------------
The clean reference *is* a multi-frame average. To reproduce it without blur:

1. Train in **packed RAW** (Poisson–Gaussian, no demosaic smear).
2. Build GT from **many** frames (default 100), not ~12.
3. Train on **every** burst frame (and random K-frame stacks), not one PNG pair.
4. Prefer **multi-frame fusion at inference** when a burst is available — with
   K≥8–32 the problem becomes easy and outputs approach the long average.
5. Use an **anti-blur** objective (Charbonnier + edge + high-frequency FFT) so
   the single-frame (K=1) mode stays as sharp as information allows.

This module is the production path for that recipe.
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

from .models import _NAFBlock
from .raw_domain import load_packed, pack_raw, packed_to_rgb


def load_packed_any(path: Path | str) -> np.ndarray:
    """Packed RAW from a DNG (rawpy) or HxWx4 / Bayer .npy (smoke tests)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim == 2:
            return pack_raw(arr)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            return arr
        raise ValueError(f"unsupported .npy shape {arr.shape} for {path}")
    return load_packed(path)


def burst_clean_any(paths, limit: int = 128) -> np.ndarray:
    acc = None
    n = 0
    for p in list(paths)[:limit]:
        pk = load_packed_any(p)
        acc = pk if acc is None else acc + pk
        n += 1
    return acc / max(n, 1)


_BURST_EXTS = {".dng", ".npy", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

@dataclass
class BurstScene:
    """One static scene/gain burst with a held-out GT window."""

    name: str                 # e.g. cabinet_H_2/ag512
    files: list[Path]
    gt_frames: int = 100
    sensor_tag: str = "imx662"  # imx662 | imx662h (informational)

    def gt_packed(self) -> np.ndarray:
        n = min(self.gt_frames, len(self.files))
        return burst_clean_any(self.files, limit=n)

    def train_files(self) -> list[Path]:
        """Frames usable as noisy inputs.

        Prefer frames *outside* the GT average window so evaluation is honest.
        If the burst is short, fall back to all frames.
        """
        n = min(self.gt_frames, len(self.files))
        held = self.files[n:]
        return held if len(held) >= 4 else list(self.files)

    def sample_stack(self, k: int, rng: random.Random) -> list[Path]:
        pool = self.train_files()
        k = max(1, min(int(k), len(pool)))
        return rng.sample(pool, k) if k < len(pool) else list(pool[:k])


def discover_bursts(
    bursts_root: Path | str,
    *,
    gains: list[int] | None = None,
    scenes: list[str] | None = None,
    min_frames: int = 16,
    gt_frames: int = 100,
    sensor_tag: str = "imx662",
) -> list[BurstScene]:
    """Walk ``bursts/<scene>/ag<N>/*.dng`` (and optional ``*_hcg`` / ``imx662h``)."""
    root = Path(bursts_root).expanduser()
    if not root.is_dir():
        return []
    want_gains = {int(g) for g in gains} if gains else None
    want_scenes = {s for s in scenes} if scenes else None
    out: list[BurstScene] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if want_scenes and scene_dir.name not in want_scenes:
            continue
        for gain_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            name = gain_dir.name.lower()
            # accept ag512, ag512_hcg, imx662h_ag512, …
            digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
            if not digits:
                continue
            g = int(digits[0])
            if want_gains is not None and g not in want_gains:
                continue
            files = sorted(
                p for p in gain_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _BURST_EXTS
            )
            if len(files) < min_frames:
                continue
            tag = sensor_tag
            if "662h" in name or "hcg" in name:
                tag = "imx662h"
            out.append(BurstScene(
                name=f"{scene_dir.name}/{gain_dir.name}",
                files=files,
                gt_frames=min(gt_frames, len(files)),
                sensor_tag=tag,
            ))
    return out


# ---------------------------------------------------------------------------
# Anti-blur losses (RAW-domain safe)
# ---------------------------------------------------------------------------

def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 5e-4) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def fft_highfreq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match FFT magnitudes with emphasis on high spatial frequencies.

    Blur removes HF energy; matching |FFT| forces the network to keep the same
    micro-contrast as the multi-frame GT instead of the soft posterior mean.
    """
    # pred/target: B,C,H,W
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    mp, mt = fp.abs(), ft.abs()
    b, c, h, w = mp.shape
    yy = torch.linspace(0, 1, h, device=pred.device, dtype=pred.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=pred.device, dtype=pred.dtype).view(1, 1, 1, w)
    # radial weight in [1, 2] — mild HF emphasis (too strong → noise restore)
    weight = 1.0 + (yy ** 2 + xx ** 2).clamp(0, 1)
    return ((mp - mt).abs() * weight).mean()


def gt_match_loss(pred: torch.Tensor, target: torch.Tensor,
                  *, w_pix: float = 1.0, w_edge: float = 0.5,
                  w_fft: float = 0.05, eps: float = 5e-4) -> torch.Tensor:
    """Composite that stays sharp while still killing grain."""
    return (w_pix * charbonnier(pred, target, eps)
            + w_edge * edge_loss(pred, target)
            + w_fft * fft_highfreq_loss(pred, target))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RawUNet(nn.Module):
    """Compact U-shaped NAFNet on packed RAW (4 or 4K input channels)."""

    def __init__(self, in_ch: int = 4, out_ch: int = 4,
                 base_channels: int = 32, depths=(2, 2, 4, 2)):
        super().__init__()
        c = base_channels
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.enc1 = nn.Sequential(*[_NAFBlock(c) for _ in range(depths[0])])
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[_NAFBlock(c * 2) for _ in range(depths[1])])
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.mid = nn.Sequential(*[_NAFBlock(c * 4) for _ in range(depths[2])])
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(*[_NAFBlock(c * 2) for _ in range(depths[3])])
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(*[_NAFBlock(c) for _ in range(depths[0])])
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual around the *reference* frame (first 4 channels).
        ref = x[:, :4]
        h = self.head(x)
        e1 = self.enc1(h)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        return torch.clamp(ref + self.tail(d1), 0.0, 1.0)


class BurstFusionDenoiser(nn.Module):
    """Fuse up to ``max_frames`` packed-RAW frames into one clean packed RAW.

    Input layout: ``(B, 4 * K, H, W)`` with K ≤ max_frames. Missing slots are
    zeros + a validity mask channel per frame is *not* required for static
    tripod bursts (we always pass real frames); at inference pad by repeating
    the last frame so the network always sees max_frames or use ``forward_k``.
    """

    def __init__(self, max_frames: int = 8, base_channels: int = 32,
                 depths=(2, 2, 4, 2)):
        super().__init__()
        self.max_frames = int(max_frames)
        self.net = RawUNet(in_ch=4 * self.max_frames, out_ch=4,
                           base_channels=base_channels, depths=depths)

    def pack_stack(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """``frames``: list of (B,4,H,W) → (B, 4*max_frames, H, W), pad by repeat."""
        assert frames, "need at least one frame"
        k = len(frames)
        if k >= self.max_frames:
            chosen = frames[:self.max_frames]
        else:
            chosen = list(frames) + [frames[-1]] * (self.max_frames - k)
        return torch.cat(chosen, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_frames(self, frames: list[torch.Tensor]) -> torch.Tensor:
        return self.forward(self.pack_stack(frames))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    bursts_root: str
    out_dir: str = "outputs/gt_match"
    gains: list[int] = field(default_factory=lambda: [128, 256, 512])
    scenes: list[str] | None = None
    gt_frames: int = 100
    max_frames: int = 8          # K at train/infer (1..max)
    min_k: int = 1               # randomly sample K in [min_k, max_frames]
    steps: int = 8000
    crop: int = 192
    batch: int = 4
    lr: float = 2e-3
    base_channels: int = 32
    seed: int = 662
    w_pix: float = 1.0
    w_edge: float = 0.5
    w_fft: float = 0.05
    device: str = "cuda"
    eval_every: int = 500


def _to_tensor(packed: np.ndarray) -> torch.Tensor:
    # (H,W,4) -> (1,4,H,W)
    t = torch.from_numpy(np.ascontiguousarray(packed.transpose(2, 0, 1)))
    return t.unsqueeze(0).float()


def _random_crop_stack(stacks: list[np.ndarray], gt: np.ndarray,
                       crop: int, rng: np.random.Generator):
    h, w = gt.shape[:2]
    c = min(crop, h, w)
    y = int(rng.integers(0, h - c + 1))
    x = int(rng.integers(0, w - c + 1))
    crops = [s[y:y + c, x:x + c] for s in stacks]
    return crops, gt[y:y + c, x:x + c]


def _augment(xs: list[torch.Tensor], y: torch.Tensor, rng: random.Random):
    if rng.random() < 0.5:
        xs = [torch.flip(x, [-1]) for x in xs]
        y = torch.flip(y, [-1])
    if rng.random() < 0.5:
        xs = [torch.flip(x, [-2]) for x in xs]
        y = torch.flip(y, [-2])
    k = rng.randrange(4)
    if k:
        xs = [torch.rot90(x, k, [-2, -1]) for x in xs]
        y = torch.rot90(y, k, [-2, -1])
    return xs, y


def build_scene_cache(scenes: list[BurstScene], log=print) -> list[dict]:
    """Preload GT + index train files (lazy-load noisy frames per step)."""
    cache = []
    for sc in scenes:
        log(f"  GT {sc.name}: {len(sc.files)} frames, "
            f"avg first {sc.gt_frames}, train pool {len(sc.train_files())}")
        cache.append({
            "scene": sc,
            "gt": sc.gt_packed(),
            "train_files": sc.train_files(),
        })
    return cache


def train_gt_match(cfg: TrainConfig, log=print) -> dict:
    """Train BurstFusionDenoiser on real bursts. Returns metrics + paths."""
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        log("CUDA unavailable — training on CPU (slow; use the AI server GPU)")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenes = discover_bursts(
        cfg.bursts_root, gains=cfg.gains, scenes=cfg.scenes,
        gt_frames=cfg.gt_frames, min_frames=max(16, cfg.gt_frames // 4),
    )
    if not scenes:
        raise FileNotFoundError(
            f"No bursts with enough frames under {cfg.bursts_root}. "
            "Expected datasets/imx662_project/bursts/<scene>/ag<N>/*.dng"
        )
    log(f"found {len(scenes)} burst scenes")
    cache = build_scene_cache(scenes, log=log)

    model = BurstFusionDenoiser(
        max_frames=cfg.max_frames, base_channels=cfg.base_channels,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters())
    log(f"BurstFusionDenoiser max_k={cfg.max_frames} "
        f"ch={cfg.base_channels} params={npar:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    warmup = max(1, cfg.steps // 10)

    def lr_at(i):
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, cfg.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    model.train()
    t0 = time.time()
    history = []

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        batch_pred = []
        batch_gt = []
        for _ in range(cfg.batch):
            entry = cache[rng.randrange(len(cache))]
            pool = entry["train_files"]
            k = rng.randint(cfg.min_k, cfg.max_frames)
            k = min(k, len(pool))
            picks = rng.sample(pool, k) if k < len(pool) else pool[:k]
            stacks = [load_packed_any(p) for p in picks]
            crops, gt_c = _random_crop_stack(stacks, entry["gt"], cfg.crop, np_rng)
            xs = [_to_tensor(c).squeeze(0) for c in crops]   # each (4,H,W)
            y = _to_tensor(gt_c).squeeze(0)
            xs, y = _augment(xs, y, rng)
            # rebuild list of (1,4,H,W) for pack_stack
            xs = [x.unsqueeze(0) for x in xs]
            pred = model.forward_frames(xs)
            batch_pred.append(pred)
            batch_gt.append(y.unsqueeze(0))
        pred_b = torch.cat(batch_pred, dim=0).to(device)
        gt_b = torch.cat(batch_gt, dim=0).to(device)
        loss = gt_match_loss(
            pred_b, gt_b, w_pix=cfg.w_pix, w_edge=cfg.w_edge, w_fft=cfg.w_fft,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == cfg.steps - 1:
            log(f"  step {step:5d}/{cfg.steps}  loss {loss.item():.5f}  "
                f"{(time.time() - t0) / max(1, step + 1):.2f}s/it")
            history.append({"step": step, "loss": float(loss.item())})

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            metrics = evaluate_heldout(model, cache, device=device, log=log)
            (out / "metrics_latest.json").write_text(json.dumps(metrics, indent=2))

    ckpt = out / "gt_match_denoiser.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "max_frames": cfg.max_frames,
        "base_channels": cfg.base_channels,
        "gt_frames": cfg.gt_frames,
        "config": cfg.__dict__,
    }, ckpt)
    log(f"saved {ckpt}")

    metrics = evaluate_heldout(model, cache, device=device, log=log)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    write_panel(model, cache, out / "panel.png", device=device, display_gain=8.0)
    return {"checkpoint": str(ckpt), "metrics": metrics, "out_dir": str(out)}


@torch.no_grad()
def evaluate_heldout(model: BurstFusionDenoiser, cache: list[dict],
                     *, device: str, k_eval: tuple[int, ...] = (1, 4, 8),
                     log=print) -> dict:
    """PSNR vs multi-frame GT for K=1/4/8 on held-out frames."""
    model.eval()
    results = {}
    # Deduplicate after clamping to model.max_frames / pool length.
    ks = []
    for k in k_eval:
        kk = min(int(k), model.max_frames)
        if kk not in ks:
            ks.append(kk)
    for entry in cache:
        sc: BurstScene = entry["scene"]
        gt = entry["gt"]
        pool = entry["train_files"]
        if len(pool) < 1:
            continue
        scene_res = {}
        for k in ks:
            kk = min(k, len(pool))
            if kk < 1:
                continue
            # take a deterministic slice far from the start
            start = max(0, len(pool) - kk - 3)
            picks = pool[start:start + kk]
            if not picks:
                continue
            frames = [_to_tensor(load_packed_any(p)).to(device) for p in picks]
            out = model.forward_frames(frames).cpu().numpy()[0].transpose(1, 2, 0)
            # centre crop metrics (avoid edge pad artefacts)
            h, w = gt.shape[:2]
            m = min(16, h // 8, w // 8)
            a, b = out[m:h - m or None, m:w - m or None], gt[m:h - m or None, m:w - m or None]
            mse = float(np.mean((a - b) ** 2))
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            # HF energy ratio (1.0 = same sharpness as GT)
            def hf(x):
                g = np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean()
                return float(g)
            sharp = hf(a) / max(hf(b), 1e-8)
            scene_res[f"k{kk}"] = {"psnr": psnr, "sharp_ratio": sharp,
                                   "frames": [p.name for p in picks]}
            log(f"  eval {sc.name} K={kk}: PSNR {psnr:.2f} dB  "
                f"sharp_ratio {sharp:.3f} (1.0=GT)")
        results[sc.name] = scene_res
    model.train()
    return results


@torch.no_grad()
def write_panel(model, cache, path: Path, *, device: str, display_gain: float = 8.0):
    from PIL import Image
    model.eval()
    strips = []
    for entry in cache[:6]:
        gt = entry["gt"]
        pool = entry["train_files"]
        if not pool:
            continue
        noisy = load_packed_any(pool[min(len(pool) // 2, len(pool) - 1)])
        # K=1 and K=max
        o1 = model.forward_frames([_to_tensor(noisy).to(device)])
        o1 = o1.cpu().numpy()[0].transpose(1, 2, 0)
        k = min(model.max_frames, len(pool))
        picks = pool[-k:]
        ok = model.forward_frames(
            [_to_tensor(load_packed_any(p)).to(device) for p in picks])
        ok = ok.cpu().numpy()[0].transpose(1, 2, 0)
        strip = np.concatenate([
            packed_to_rgb(noisy, display_gain),
            packed_to_rgb(o1, display_gain),
            packed_to_rgb(ok, display_gain),
            packed_to_rgb(gt, display_gain),
        ], axis=1)
        strips.append(strip)
    if not strips:
        return
    img = (np.clip(np.concatenate(strips, axis=0), 0, 1) * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    model.train()


@torch.no_grad()
def infer_burst(model: BurstFusionDenoiser, frame_paths: list[Path] | list[str],
                *, device: str = "cpu") -> np.ndarray:
    """Denoise a burst (or single frame) → packed RAW (H,W,4)."""
    model.eval()
    paths = [Path(p) for p in frame_paths]
    frames = [_to_tensor(load_packed_any(p)).to(device) for p in paths]
    out = model.forward_frames(frames)
    return out.cpu().numpy()[0].transpose(1, 2, 0)


def load_model(ckpt_path: Path | str, device: str = "cpu") -> BurstFusionDenoiser:
    blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        model = BurstFusionDenoiser(
            max_frames=int(blob.get("max_frames", 8)),
            base_channels=int(blob.get("base_channels", 32)),
        )
        model.load_state_dict(blob["state_dict"])
    else:
        model = BurstFusionDenoiser()
        model.load_state_dict(blob)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Synthetic smoke-test data (no DNG / no AI server required)
# ---------------------------------------------------------------------------

def make_synthetic_burst_dir(root: Path, *, n_frames: int = 48,
                             h: int = 128, w: int = 128, gain: float = 0.08) -> Path:
    """Write a fake packed-as-.npy burst for CI / local smoke tests."""
    scene = root / "synth_scene" / "ag512"
    scene.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = (0.25 + 0.2 * np.sin(xx / 7) * np.cos(yy / 5)
             + 0.15 * ((xx // 16 + yy // 16) % 2)).astype(np.float32)
    clean = np.clip(clean, 0, 1)
    for i in range(n_frames):
        noisy = np.clip(clean + rng.normal(0, gain, clean.shape).astype(np.float32),
                        0, 1)
        np.save(scene / f"frame_{i:04d}.npy", pack_raw(noisy))
    return scene
