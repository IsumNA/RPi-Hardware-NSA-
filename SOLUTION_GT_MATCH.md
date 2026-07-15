# Genuinely solving the blur problem

## What you asked for

Denoise IMX662 / IMX662H so the output looks like the **~100-frame average**
(the ground truth) — clean **and** sharp, not plastic-blurred.

## What was actually wrong

1. **Single-frame neural regression cannot equal a 100-frame average.**  
   L1/Charbonnier/SWT learn the conditional mean. At high gain that mean is soft.
   Changing the loss does not create photons.

2. **Domain mismatch (real bug).**  
   Noisy DNGs were loaded with `rawpy.postprocess` (camera gamma / ISP demosaic).
   GT was a linear Bayer mean + OpenCV demosaic. The network was trained across
   two different image formations → soft compromise. **Fixed** in `nsa/raw_io.py`
   (`_load_dng_linear_rgb`).

3. **Synthetic demo GT was pre-blurred.**  
   `_synthetic_scene` applied `GaussianBlur(σ=0.6)` to the clean target, so every
   demo taught “blur is correct”. **Removed.**

4. **Pair builder used too few frames in places** (`burst/4` ≈ 12). **Fixed** to
   require / pull ~100 frames.

## The actual solution

```bash
# THIS is how you get the 100-frame look — because it IS the 100-frame method:
python solve_denoise.py \
  --burst datasets/imx662_project/bursts/<scene>/ag512 \
  --max-frames 100 \
  --out outputs/solved.png
```

`nsa/solve.merge_burst` aligns (ECC) and averages in linear RGB — the same
definition as your GT. No network required for perfect match on a static scene.

### Single frame (when you truly have only one)

```bash
python solve_denoise.py --input noisy.dng --out outputs/solved_single.png
```

Dual-domain preserve (bilateral base + SNR-gated detail). Keeps resolution-bar
contrast better than L1-NAFNet; still cannot invent 100-frame SNR.

### Live / streaming

Use the existing temporal accumulator in `live.py` (motion-gated running mean).
Let it reach K≈32–100 on a static scene — that converges to the same answer.

### Optional neural polish (after merge is correct)

```bash
python train_gt_match.py --bursts datasets/imx662_project/bursts \
  --gt-frames 100 --max-frames 8 --steps 8000
```

Trains packed-RAW fusion to *approximate* the merge with fewer frames. It is a
speed/quality tradeoff, not a substitute for averaging when you have the burst.

## Proof (no camera)

```bash
python solve_denoise.py --proof --out-dir outputs/solve_proof
```

Panel columns: **clean | noisy | blurry-L1-like | single-preserve | 100-merge**.
Metrics in `proof_metrics.json` — merge recovery of chirp contrast should be ≈1.

## Bottom line

| Situation | What to run | Looks like 100-frame GT? |
|-----------|-------------|---------------------------|
| Have a burst (static) | `solve_denoise.py --burst … --max-frames 100` | **Yes (identical method)** |
| Live static | `live.py` temporal accumulate to K≈100 | **Yes (approaches)** |
| One frame only | `--input` preserve / RAW net | Partial — physics limit |

If the product requirement is “must look like the average”, the product must
**capture or accumulate multiple frames**. No single-frame network will honestly
do that at high analogue gain without blur or hallucination.
