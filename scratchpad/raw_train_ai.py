#!/usr/bin/env python3
"""RAW-domain multi-frame denoiser trained on the REAL ag512 bursts (AI server, GPU).

For each scene we have a burst of raw DNG frames of a static scene at analog gain
512 (the 2-lux / ISO-512 stress case). We build:
  * clean GT   = temporal mean of the first N raw frames (packed Bayer)  -> true clean
  * noisy in   = individual raw frames                                   -> training inputs
  * held-out   = raw frames from OUTSIDE the GT-average window           -> unseen noise
Then train a RawDenoiser in the packed-raw domain (noise is simple there) and report
honest held-out raw PSNR/SSIM plus a NOISY | DENOISED | GT panel per scene.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import math
from PIL import Image

ROOT = Path("/home/isum.nanomi-arachchige/RPi-Hardware-NSA-")
sys.path.insert(0, str(ROOT))
from nsa.raw_domain import load_packed, burst_clean, packed_to_rgb, RawDenoiser
from nsa.inference import build_loss, _sample_batch, to_tensor, to_image, psnr, ssim

BURSTS = ROOT / "datasets/imx662_project/bursts"
OUT = ROOT / "outputs/raw_ag512"; OUT.mkdir(parents=True, exist_ok=True)
GAIN = "ag512"
SCENES = ["cabinet_H_2", "cabinet_H_10", "cabinet_F11_25", "cabinet_D50_100", "colour_stripes"]

GT_FRAMES   = 256          # frames averaged for the clean reference
TRAIN_STRIDE = 6           # take every Nth frame in [0:GT_FRAMES] as a noisy input
EVAL_IDX    = [400, 440, 480]   # held-out (outside GT window) -> unseen noise
STEPS  = 4000
CROP   = 224
BATCH  = 4
DISPLAY_GAIN = 8.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, flush=True)


def build_dataset():
    pairs, evals = [], []            # evals: (scene, gt, [noisy frames])
    for sc in SCENES:
        d = BURSTS / sc / GAIN
        files = sorted(d.glob("*.dng"))
        if not files:
            log(f"  {sc}: no frames, skip"); continue
        n = len(files)
        gt = burst_clean(files, limit=min(GT_FRAMES, n))
        train_idx = list(range(0, min(GT_FRAMES, n), TRAIN_STRIDE))
        for i in train_idx:
            pairs.append((load_packed(files[i]), gt))
        ev_idx = [i for i in EVAL_IDX if i < n]
        ev_noisy = [(i, load_packed(files[i])) for i in ev_idx]
        evals.append((sc, gt, ev_noisy))
        log(f"  {sc}: {n} frames | {len(train_idx)} train | eval {ev_idx} | "
            f"gt.mean={gt.mean():.4f}")
    return pairs, evals


def train(model, pairs):
    tensors = [(to_tensor(n), to_tensor(c)) for n, c in pairs]
    loss_fn = build_loss("charbonnier+swt", charbonnier_eps=5e-4,
                         weights={"charbonnier": 1.0, "swt": 0.1})
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    warmup = max(1, STEPS // 10)

    def lr_at(i):
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1 + math.cos(math.pi * t)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(662)
    model.train(); t0 = time.time()
    for i in range(STEPS):
        xb, yb = _sample_batch(tensors, CROP, BATCH, g)
        xb, yb = xb.to(DEV), yb.to(DEV)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if i % 100 == 0 or i == STEPS - 1:
            log(f"  step {i:4d}/{STEPS}  loss {loss.item():.5f}  "
                f"{(time.time()-t0)/max(1,i):.2f}s/it")
    model.eval()
    log(f"trained in {(time.time()-t0)/60:.1f} min")
    return model


@torch.no_grad()
def infer(model, packed):
    x = to_tensor(packed).to(DEV)
    out = model(x)
    return to_image(out.cpu())


def main():
    log(f"device={DEV}  torch={torch.__version__}")
    log("building dataset from real ag512 bursts...")
    pairs, evals = build_dataset()
    log(f"total train pairs: {len(pairs)}")

    model = RawDenoiser(base_channels=64, block_depth=8)
    npar = sum(p.numel() for p in model.parameters())
    log(f"RawDenoiser 64ch/8blk = {npar:,} params")
    model = train(model, pairs)

    torch.save(model.state_dict(), OUT / "raw_ag512_denoiser.pt")  # save first (insurance)
    results = {}
    strips = []
    for sc, gt, ev in evals:
        if not ev:
            log(f"{sc}: no held-out frames, skipping"); continue
        for i, noisy in ev:
            out = infer(model, noisy)
            pin, pout = psnr(noisy, gt), psnr(out, gt)
            sin, sout = ssim(noisy, gt), ssim(out, gt)
            results.setdefault(sc, []).append(
                dict(frame=i, psnr_in=pin, psnr_out=pout, ssim_in=sin, ssim_out=sout))
            log(f"{sc} f{i}: PSNR {pin:.2f}->{pout:.2f} (+{pout-pin:.2f})  "
                f"SSIM {sin:.3f}->{sout:.3f}")
        i0, noisy0 = ev[0]
        out0 = infer(model, noisy0)
        strip = np.concatenate([packed_to_rgb(noisy0, DISPLAY_GAIN),
                                packed_to_rgb(out0, DISPLAY_GAIN),
                                packed_to_rgb(gt, DISPLAY_GAIN)], axis=1)
        strips.append(strip)

    img = (np.clip(np.concatenate(strips, axis=0), 0, 1) * 255 + 0.5).astype(np.uint8)
    ppath = OUT / "raw_ag512_panel.png"
    Image.fromarray(img).save(ppath)
    (OUT / "raw_ag512_metrics.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), OUT / "raw_ag512_denoiser.pt")

    allrows = [r for rs in results.values() for r in rs]
    dpsnr = np.mean([r["psnr_out"] - r["psnr_in"] for r in allrows])
    dssim = np.mean([r["ssim_out"] - r["ssim_in"] for r in allrows])
    log(f"\n=== SUMMARY over {len(allrows)} held-out frames ===")
    log(f"mean PSNR gain {dpsnr:+.2f} dB   mean SSIM gain {dssim:+.3f}")
    log(f"panel   -> {ppath}")
    log(f"metrics -> {OUT/'raw_ag512_metrics.json'}")
    log("cols: NOISY | RAW-DENOISED | RAW-GT  (rows=scenes, brightened %gx)" % DISPLAY_GAIN)


if __name__ == "__main__":
    main()
