#!/usr/bin/env python3
"""Visual denoise eval: render full frames + zoomed detail crops (noisy|out|GT).

Purpose: judge sharpness/cleanliness by EYE, not just PSNR. Emits one PNG with,
per held-out frame: a full-res row and N zoomed crop rows at nearest-neighbour
upscale so pixel-level softness is visible.

  .venv/bin/python eval_visual.py --ckpt outputs/model.pt --out /tmp/eval.png \
      --filter imx662 ag512 --n 2 --crop 150 --zoom 3 --crops-per 3
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsa.config import ModelConfig
from nsa.models import build_model, count_params
from nsa.raw_io import load_training_pairs


def cfg_from_ckpt(ck):
    m = ck.get("model") if isinstance(ck, dict) else None
    if not isinstance(m, dict):
        m = {}
    return ModelConfig(
        model_family=m.get("family") or m.get("model_family") or "nafnet",
        base_channels=int(m.get("base_channels") or 16),
        block_depth=int(m.get("block_depth") or 4),
        conv_type=m.get("conv_type") or "depthwise",
        activation=m.get("activation") or "relu",
        nafnet_enc_blocks=list(m.get("nafnet_enc") or [1, 1, 2]),
        nafnet_middle_blocks=int(m.get("nafnet_middle") or 2),
        nafnet_dec_blocks=list(m.get("nafnet_dec") or [1, 1, 1]),
    )


def load_model(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    cfg = cfg_from_ckpt(ck)
    model = build_model(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)
    return model, cfg, count_params(model)


def infer(model, noisy, dev):
    x = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        y = model(x)
    return y.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def lap_var(g):
    from scipy.ndimage import laplace
    return float(laplace(g).var())


def pick_crops(gt, k, csz):
    """Pick k detail-rich, non-overlapping crop top-left coords from GT."""
    h, w, _ = gt.shape
    g = gt.mean(2)
    # simple laplacian via np gradient magnitude of gradient
    gy, gx = np.gradient(g)
    lap = np.abs(np.gradient(gx, axis=1)) + np.abs(np.gradient(gy, axis=0))
    step = max(16, csz // 2)
    cands = []
    for y in range(0, h - csz, step):
        for x in range(0, w - csz, step):
            s = lap[y:y + csz, x:x + csz].mean()
            cands.append((s, y, x))
    cands.sort(reverse=True)
    chosen = []
    for s, y, x in cands:
        if all(abs(y - cy) > csz * 0.7 or abs(x - cx) > csz * 0.7 for cy, cx in chosen):
            chosen.append((y, x))
        if len(chosen) >= k:
            break
    return chosen


def to_u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def upscale(a, z):
    im = Image.fromarray(to_u8(a))
    return im.resize((im.width * z, im.height * z), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="datasets/PI_RAW")
    ap.add_argument("--filter", nargs="*", default=["imx662"])
    ap.add_argument("--gain", type=int, default=512)
    ap.add_argument("--n", type=int, default=2, help="num held-out frames")
    ap.add_argument("--crop", type=int, default=150)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--crops-per", type=int, default=3)
    ap.add_argument("--full-w", type=int, default=360, help="full-frame display width")
    ap.add_argument("--label", default="")
    ap.add_argument("--npz", default=None, help="load (noisy,gt) pairs from npz "
                    "(keys noisy_i/gt_i/name_i) instead of the dataset loader")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, nparams = load_model(args.ckpt, dev)

    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        i = 0
        trips = []
        while f"noisy_{i}" in z:
            nm = str(z[f"name_{i}"]) if f"name_{i}" in z else f"pair{i}"
            trips.append((nm, z[f"noisy_{i}"].astype(np.float32),
                          z[f"gt_{i}"].astype(np.float32)))
            i += 1
        trips = trips[:args.n]
    else:
        trips = load_training_pairs(args.dataset, args.filter or None, sensor="imx662",
                                    gain=args.gain, with_names=True, tile=0, max_side=1024)
        if not trips:
            trips = load_training_pairs(args.dataset, None, sensor="imx662",
                                        gain=args.gain, with_names=True, tile=0, max_side=1024)
        if not trips:
            print("NO PAIRS FOUND", file=sys.stderr); sys.exit(2)
        # noisiest first (highest std) so we stress the model
        trips = sorted(trips, key=lambda t: -float(np.std(t[1])))[:args.n]

    z, csz, fw = args.zoom, args.crop, args.full_w
    rows = []  # each row is a PIL image (horizontal strip)
    from PIL import Image as PImage

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)

    for name, noisy, gt in trips:
        out = infer(model, noisy, dev)
        # full-frame strip
        h, w, _ = noisy.shape
        fh = int(fw * h / w)
        def full(a):
            return Image.fromarray(to_u8(a)).resize((fw, fh), Image.BILINEAR)
        strip = Image.new("RGB", (fw * 3 + 20, fh), (20, 20, 20))
        for j, a in enumerate((noisy, out, gt)):
            strip.paste(full(a), (j * (fw + 10), 0))
        d = ImageDraw.Draw(strip)
        d.text((4, 4), f"{name} ag{args.gain}  PSNR {psnr(out,gt):.2f}dB  [noisy|OUT|GT]",
               fill=(0, 255, 0))
        rows.append(strip)
        # zoom crop strips
        for (cy, cx) in pick_crops(gt, args.crops_per, csz):
            cn = upscale(noisy[cy:cy+csz, cx:cx+csz], z)
            co = upscale(out[cy:cy+csz, cx:cx+csz], z)
            cg = upscale(gt[cy:cy+csz, cx:cx+csz], z)
            cw, ch = cn.size
            cstrip = Image.new("RGB", (cw * 3 + 20, ch), (20, 20, 20))
            for j, cim in enumerate((cn, co, cg)):
                cstrip.paste(cim, (j * (cw + 10), 0))
            rows.append(cstrip)

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 8 * len(rows)
    canvas = Image.new("RGB", (W, H + 24), (10, 10, 10))
    dd = ImageDraw.Draw(canvas)
    dd.text((6, 4), f"{args.label or Path(args.ckpt).name}  |  {cfg.model_family} "
                    f"base{cfg.base_channels} enc{cfg.nafnet_enc_blocks} "
                    f"mid{cfg.nafnet_middle_blocks} dec{cfg.nafnet_dec_blocks}  "
                    f"|  {nparams/1000:.0f}K params", fill=(255, 255, 0))
    y = 24
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 8
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print("WROTE", args.out, canvas.size, "params", nparams)


if __name__ == "__main__":
    main()
