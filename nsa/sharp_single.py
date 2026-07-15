"""Sharp single-frame RAW denoising — for scenes where you cannot average (motion).

Lab GT is still a multi-frame average of a *static* capture. At deploy time the
camera / subject moves, so you only get one usable frame. This module trains a
packed-RAW network to map that one noisy frame → the sharp static GT, using an
anti-blur objective (Charbonnier + edge + HF-FFT + optional PatchGAN) so it does
not collapse to the soft posterior mean.
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
    BurstScene, RawUNet, _to_tensor, _augment, build_scene_cache, discover_bursts,
    edge_loss, fft_highfreq_loss, charbonnier, load_packed_any, make_synthetic_burst_dir,
)
from .models import _NAFBlock
from .raw_domain import pack_raw, packed_to_rgb


class SharpRawDenoiser(nn.Module):
    """Single-frame packed-RAW → clean packed-RAW (residual U-NAFNet)."""

    def __init__(self, base_channels: int = 48, depths=(2, 2, 4, 2)):
        super().__init__()
        self.net = RawUNet(in_ch=4, out_ch=4, base_channels=base_channels, depths=depths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN — pushes outputs off the blurry mean toward GT texture."""

    def __init__(self, in_ch: int = 4, base: int = 32):
        super().__init__()
        layers = []
        ch = in_ch
        for i, mult in enumerate((1, 2, 4, 4)):
            out = base * mult
            layers += [
                nn.Conv2d(ch, out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if i > 0:
                layers.insert(-1, nn.InstanceNorm2d(out, affine=True))
            ch = out
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sharp_loss(pred, target, *, w_pix=1.0, w_edge=0.75, w_fft=0.1, eps=5e-4):
    """Heavier edge/FFT than plain gt_match — biased against plastic blur."""
    return (w_pix * charbonnier(pred, target, eps)
            + w_edge * edge_loss(pred, target)
            + w_fft * fft_highfreq_loss(pred, target))


def gan_loss_d(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    r = disc(real)
    f = disc(fake.detach())
    return 0.5 * (F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean())


def gan_loss_g(disc: nn.Module, fake: torch.Tensor) -> torch.Tensor:
    return -disc(fake).mean()


@dataclass
class SharpTrainConfig:
    bursts_root: str
    out_dir: str = "outputs/sharp_single"
    gains: list[int] = field(default_factory=lambda: [128, 256, 512])
    scenes: list[str] | None = None
    gt_frames: int = 100
    steps: int = 10000
    crop: int = 192
    batch: int = 4
    lr: float = 1.5e-3
    base_channels: int = 48
    seed: int = 662
    w_edge: float = 0.75
    w_fft: float = 0.1
    w_gan: float = 0.05          # 0 = off
    device: str = "cuda"
    eval_every: int = 500


def train_sharp_single(cfg: SharpTrainConfig, log=print) -> dict:
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        log("CUDA unavailable — CPU training (use AI server GPU for real runs)")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenes = discover_bursts(
        cfg.bursts_root, gains=cfg.gains, scenes=cfg.scenes,
        gt_frames=cfg.gt_frames, min_frames=max(16, cfg.gt_frames // 4),
    )
    if not scenes:
        raise FileNotFoundError(f"No bursts under {cfg.bursts_root}")
    log(f"found {len(scenes)} burst scenes (GT={cfg.gt_frames} static avg → single-frame map)")
    cache = build_scene_cache(scenes, log=log)

    model = SharpRawDenoiser(base_channels=cfg.base_channels).to(device)
    disc = PatchDiscriminator().to(device) if cfg.w_gan > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr * 0.5, weight_decay=1e-4) if disc else None

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
        xs, ys = [], []
        for _ in range(cfg.batch):
            entry = cache[rng.randrange(len(cache))]
            pool = entry["train_files"]
            # ONE noisy frame — deploy constraint under motion
            fpath = pool[rng.randrange(len(pool))]
            noisy = load_packed_any(fpath)
            gt = entry["gt"]
            h, w = gt.shape[:2]
            c = min(cfg.crop, h, w)
            y0 = int(np_rng.integers(0, h - c + 1))
            x0 = int(np_rng.integers(0, w - c + 1))
            n_c = noisy[y0:y0 + c, x0:x0 + c]
            g_c = gt[y0:y0 + c, x0:x0 + c]
            xt = _to_tensor(n_c).squeeze(0)
            yt = _to_tensor(g_c).squeeze(0)
            (xt,), yt = _augment([xt], yt, rng)
            xs.append(xt.unsqueeze(0))
            ys.append(yt.unsqueeze(0))
        xb = torch.cat(xs, 0).to(device)
        yb = torch.cat(ys, 0).to(device)

        if disc is not None and opt_d is not None:
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = model(xb)
            loss_d = gan_loss_d(disc, yb, fake)
            loss_d.backward()
            opt_d.step()

        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = sharp_loss(pred, yb, w_edge=cfg.w_edge, w_fft=cfg.w_fft)
        if disc is not None:
            loss = loss + cfg.w_gan * gan_loss_g(disc, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == cfg.steps - 1:
            log(f"  step {step:5d}/{cfg.steps}  loss {loss.item():.5f}  "
                f"{(time.time() - t0) / max(1, step + 1):.2f}s/it")
            history.append({"step": step, "loss": float(loss.item())})

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            metrics = eval_sharp(model, cache, device=device, log=log)
            (out / "metrics_latest.json").write_text(json.dumps(metrics, indent=2))

    ckpt = out / "sharp_single.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "base_channels": cfg.base_channels,
        "gt_frames": cfg.gt_frames,
        "config": {k: v for k, v in cfg.__dict__.items()},
    }, ckpt)
    log(f"saved {ckpt}")
    metrics = eval_sharp(model, cache, device=device, log=log)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    write_sharp_panel(model, cache, out / "panel.png", device=device)
    return {"checkpoint": str(ckpt), "metrics": metrics, "out_dir": str(out)}


@torch.no_grad()
def eval_sharp(model, cache, *, device, log=print) -> dict:
    model.eval()
    results = {}
    for entry in cache:
        sc: BurstScene = entry["scene"]
        gt = entry["gt"]
        pool = entry["train_files"]
        if not pool:
            continue
        noisy = load_packed_any(pool[len(pool) // 2])
        pred = model(_to_tensor(noisy).to(device)).cpu().numpy()[0].transpose(1, 2, 0)
        h, w = gt.shape[:2]
        m = min(16, h // 8, w // 8)
        a, b = pred[m:h - m or None, m:w - m or None], gt[m:h - m or None, m:w - m or None]
        mse = float(np.mean((a - b) ** 2))
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))

        def hf(x):
            return float(np.abs(np.diff(x, 0)).mean() + np.abs(np.diff(x, 1)).mean())

        sharp_ratio = hf(a) / max(hf(b), 1e-8)
        # blurry baseline for comparison
        blur = np.stack([
            __import__("cv2").GaussianBlur(noisy[..., i], (0, 0), 2.0)
            for i in range(4)
        ], -1)
        bmse = float(np.mean((blur[m:h - m or None, m:w - m or None] - b) ** 2))
        bpsnr = 10.0 * math.log10(1.0 / max(bmse, 1e-12))
        results[sc.name] = {
            "psnr": psnr, "sharp_ratio": sharp_ratio,
            "psnr_blur_baseline": bpsnr,
            "psnr_gain_over_blur": psnr - bpsnr,
        }
        log(f"  eval {sc.name}: PSNR {psnr:.2f} (blur-base {bpsnr:.2f}, "
            f"+{psnr - bpsnr:.2f})  sharp_ratio {sharp_ratio:.3f}")
    model.train()
    return results


@torch.no_grad()
def write_sharp_panel(model, cache, path: Path, *, device, display_gain=8.0):
    from PIL import Image
    model.eval()
    strips = []
    for entry in cache[:5]:
        gt = entry["gt"]
        pool = entry["train_files"]
        if not pool:
            continue
        noisy = load_packed_any(pool[len(pool) // 2])
        pred = model(_to_tensor(noisy).to(device)).cpu().numpy()[0].transpose(1, 2, 0)
        import cv2
        blur = np.stack([cv2.GaussianBlur(noisy[..., i], (0, 0), 2.0) for i in range(4)], -1)
        strip = np.concatenate([
            packed_to_rgb(noisy, display_gain),
            packed_to_rgb(blur, display_gain),
            packed_to_rgb(pred, display_gain),
            packed_to_rgb(gt, display_gain),
        ], 1)
        strips.append(strip)
    if strips:
        img = (np.clip(np.concatenate(strips, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path)
    model.train()


def load_sharp_model(ckpt: Path | str, device="cpu") -> SharpRawDenoiser:
    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = SharpRawDenoiser(base_channels=int(blob.get("base_channels", 48)))
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer_sharp(model: SharpRawDenoiser, path: Path | str, *, device="cpu") -> np.ndarray:
    packed = load_packed_any(Path(path))
    out = model(_to_tensor(packed).to(device))
    return out.cpu().numpy()[0].transpose(1, 2, 0)


def proof_sharp_synth(out_dir: Path | str, *, steps: int = 400, device: str = "cpu") -> dict:
    """Train a tiny sharp single-frame net on synthetic bursts; prove vs blur."""
    out_dir = Path(out_dir)
    bursts = out_dir / "_synth"
    make_synthetic_burst_dir(bursts, n_frames=64, h=128, w=128, gain=0.1)
    cfg = SharpTrainConfig(
        bursts_root=str(bursts), out_dir=str(out_dir),
        gains=[512], gt_frames=32, steps=steps, crop=96, batch=2,
        lr=2e-3, base_channels=24, w_gan=0.03, w_edge=0.8, w_fft=0.12,
        device=device, eval_every=max(100, steps // 2),
    )
    result = train_sharp_single(cfg)
    # Chirp-style contrast on packed green channel of held-out frame
    from .solve import chirp_contrast
    import cv2
    entry_files = sorted((bursts / "synth_scene" / "ag512").glob("*.npy"))
    gt = np.mean([load_packed_any(p) for p in entry_files[:32]], axis=0)
    noisy = load_packed_any(entry_files[48])
    model = load_sharp_model(result["checkpoint"], device=device)
    pred = infer_sharp(model, entry_files[48], device=device)
    blur = np.stack([cv2.GaussianBlur(noisy[..., i], (0, 0), 2.0) for i in range(4)], -1)

    def contrast(p):
        # use packed→rgb preview
        return chirp_contrast(packed_to_rgb(p, 1.0), row_frac=0.5, x0_frac=0.1, x1_frac=0.9)

    metrics = {
        **result["metrics"],
        "chirp_gt": contrast(gt),
        "chirp_noisy": contrast(noisy),
        "chirp_blur": contrast(blur),
        "chirp_pred": contrast(pred),
        "psnr_pred": float(10 * math.log10(1 / max(np.mean((pred - gt) ** 2), 1e-12))),
        "psnr_blur": float(10 * math.log10(1 / max(np.mean((blur - gt) ** 2), 1e-12))),
    }
    metrics["pred_beats_blur_psnr"] = metrics["psnr_pred"] > metrics["psnr_blur"] + 1.0
    (out_dir / "proof_sharp.json").write_text(json.dumps(metrics, indent=2))
    from PIL import Image
    panel = np.concatenate([
        packed_to_rgb(noisy, 1.0), packed_to_rgb(blur, 1.0),
        packed_to_rgb(pred, 1.0), packed_to_rgb(gt, 1.0),
    ], 1)
    Image.fromarray((np.clip(panel, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        out_dir / "proof_sharp_panel.png")
    return metrics
