# GT-matching denoiser — what was wrong, what to run

## Do you understand the brief?

Yes. You need a denoiser whose output looks like the **~100-frame temporal
average** (sharp, clean, same as your burst GT) — not a soft/plastic version of
it. Pair-trained RGB models kept blurring no matter which loss you tried.

## Why everything blurred before

1. **Regression to the mean.** L1 / L2 / Charbonnier / SWT minimise average
   error. At high analogue gain, fine texture looks like noise, so the optimal
   PSNR answer *is* to blur it. Changing the loss weight cannot invent photons.
2. **Training on one noisy PNG vs one average.** The burst’s noise diversity
   never reached the model.
3. **Soft GT targets.** Some pair-building paths used a small frame count
   (`burst/4` ≈ 12 as a floor / pull subset). A 12-frame average still has
   residual grain *and* is softer than a 100-frame GT — the network learns to
   match that soft target.
4. **RGB / demosaic domain.** Demosaic correlates noise and already softens
   detail; packed RAW is the right place to denoise.

## The solution in this repo

**Multi-frame RAW fusion trained to match a 100-frame average.**

| Piece | Role |
|-------|------|
| `nsa/gt_match.py` | Burst discovery, anti-blur loss, `BurstFusionDenoiser` |
| `train_gt_match.py` | Train on AI server against real DNG bursts |
| `infer_gt_match.py` | K=1 single frame, or K=8–32 burst fusion |

### Train (AI server, GPU + real bursts)

```bash
python train_gt_match.py \
  --bursts datasets/imx662_project/bursts \
  --gains 128 256 512 \
  --gt-frames 100 \
  --max-frames 8 \
  --steps 8000 \
  --out outputs/gt_match
```

- GT = mean of the first **100** packed RAW frames  
- Inputs = **held-out** frames, randomly stacked as K=1..8  
- Loss = Charbonnier + edge + high-frequency FFT (anti-blur)  
- Writes `gt_match_denoiser.pt`, `metrics.json`, `panel.png`  
  (panel columns: NOISY | K=1 | K=max | GT)

### Infer

```bash
# best quality when you have a burst (approaches the 100-frame look)
python infer_gt_match.py --ckpt outputs/gt_match/gt_match_denoiser.pt \
  --burst datasets/imx662_project/bursts/cabinet_H_2/ag512 \
  --max-frames 16 --out outputs/fused.png

# single frame (as sharp as one noisy capture allows)
python infer_gt_match.py --ckpt outputs/gt_match/gt_match_denoiser.pt \
  --input path/to/noisy.dng --out outputs/single.png
```

### Smoke test (no DNGs)

```bash
python train_gt_match.py --synth --steps 200 --out outputs/gt_match_synth
```

## Honest limit

**One noisy frame cannot contain the information of 100 independent reads.**
K=1 can get close and stay sharper than old RGB training; **K≥8–16 at
inference** is what actually looks like your long average. For live static
scenes, capture a short burst (or temporal accumulate) and fuse — do not expect
magic from a single grainy frame.

## Also fixed

- Pair derive pull default **100** frames (`NSA_GT_PULL_MAX`)
- GUI no longer uses `burst/4` (~12) as the GT floor
- `scratchpad/raw_train_ai.py` redirects here (old stride-6 recipe retired)
