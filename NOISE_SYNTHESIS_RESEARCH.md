# IMX662 Noise Synthesis — Research Report

**Goal:** build a massive dataset of clean/noisy pairs where the synthetic noise matches the
real IMX662 noise profile, plus a rigorous way to prove the noise model is accurate.

**Date:** 2026-07-16 · All numbers below were measured directly from the DNGs in
`datasets/imx662_project/` on this machine (script snippets at the end).

---

## 1. Verdict

**The current approach is not the best way, but the data you captured is.** The calibration
captures are high quality — the read-noise figures I measure from them land exactly inside the
published IMX662 ranges — but three things are wrong today:

1. **The fitted noise model in `models/noise/imx662_gain256.json` is quantitatively wrong**
   (its own Phase-4 validation says `"ok": false`), because of fitting bugs, not bad data.
2. **The parametric model family is too weak** for this sensor at high gain: it misses heavy
   read-noise tails, fixed-pattern noise, black-level error, and the true quantization step.
3. **The synthesis pipeline operates in the wrong domain**: `simulate_dataset.py` writes 8-bit
   demosaiced RGB PNGs, while the actual denoiser (`train_stream_to_gt.py`) trains on packed
   12-bit linear Bayer. A perfect noise model would still be defeated by this domain gap.

The literature-backed best practice **given exactly the data you have** is a **hybrid
physics + dark-frame-sampling pipeline in the packed Bayer domain** (Section 4), validated by
a three-tier protocol whose gold standard uses your ~1000-frame bursts as held-out real pairs
(Section 5).

---

## 2. What you actually have (data inventory)

| Data | Contents | Usable for |
|---|---|---|
| `calibration/imx662_gain256/` (LCG) | 8 bias, 5 dark, 12 flat levels ×2 | PTC gain + read noise + row noise + FPN @ gain 256 |
| `calibration/imx662h_gain256/` (HCG) | same layout | same, HCG mode |
| `bursts/cabinet_D50_100/ag128` | 488 DNGs, static scene | real GT (average) + 488 real noise realizations |
| `bursts/cabinet_H_2/ag128` | 512 DNGs | same |
| `bursts/cabinet_D50_100/ag1` | 48 DNGs | near-clean source frame |
| `datasets/PI_RAW/Data/` | 7 scenes × ~10 gains (ag1–512, LCG + HCG), noisy singles, **no GT** | distribution checks per gain; not training pairs |

DNG format: 1096×1936 Bayer, 12-bit ADC left-shifted into uint16 (black level 3200 DN16,
white 65535). **Unique-code spacing is 128 DN16 = 8×LSB₁₂ at gain 256** — a digital-gain
stage quantizes far coarser than the assumed ±½ LSB of a 12-bit ADC.

### Measured sensor characteristics (from your calibration frames)

| Quantity | `imx662` (LCG) g256 | `imx662h` (HCG) g256 | Published IMX662 |
|---|---|---|---|
| PTC system gain K | 452.5 DN16/e⁻ | 2327.8 DN16/e⁻ (≈5.1× LCG) | HCG boost ≈5–6× on this family |
| Read noise (temporal σ from bias stack) | 1481 DN16 = **3.27 e⁻** | 1854 DN16 = **0.80 e⁻** | LCG 2.25–6.81 e⁻, HCG 0.48–1.11 e⁻ |
| Per-channel K (R/G1/G2/B) | 462 / 450 / 451 / 453 | — | near-uniform ✓ |

The match with published figures confirms both the capture protocol and my analysis: **the
raw material for an exact noise model already exists** (at gain 256; other gains still need
calibration).

### Measured noise structure that the current model does NOT capture

| Component | Measurement | Consequence if unmodeled |
|---|---|---|
| **Heavy-tailed read noise** | Best Tukey-lambda λ ≈ 0.043 (Gaussian = 0.14); P(\|n\|>5σ) = 7.4e-4 vs Gaussian 5.7e-7 (**~1300×**) | denoiser never learns to kill salt-and-pepper–like outliers / hot flickers |
| **Row banding** | per-row mean σ = 131 DN16 vs 33.7 expected if i.i.d. (**~4×**); Gaussian-shaped; column noise negligible | horizontal streaks survive denoising |
| **Fixed-pattern noise (DSNU/dark shading)** | spatial σ of mean bias frame = **588 DN16** (≈40 % of read σ), almost entirely per-pixel (not row/col); 0.13 % hot pixels beyond 5σ | denoiser hallucinates texture from the static pattern; PMN (TPAMI 23) shows this measurably hurts training |
| **Black-level error (BLE)** | frame-to-frame bias mean drift up to **137 DN16 ≈ 8.6 LSB₁₂** (frame 0 vs rest) | color/brightness bias in shadows, exactly the failure mode documented in ELD/PMN |
| **Coarse quantization** | true step 128 DN16 (8 LSB₁₂), model assumes ±½ LSB₁₂ (**8× too small**) | synthetic noise too smooth in shadows |

---

## 3. What's wrong with the current pipeline (evidence)

### 3.1 Fitting bugs (`nsa/noise_calib/fit.py`, `extract.py`)

- **Shot fit forced through the origin** (`fit_shot_poisson`, fit.py:80–85): flat-pair variance
  includes the read floor (intercept 2.49e6 DN16² — the *dominant* term at low signal), and with
  no intercept it is absorbed into `shot_a`. Result: `shot_a = 0.0240` stored vs
  `K/(white−black) = 452.5/62335 = 0.0073` true — **3.3× too high**. The saved validation
  agrees: `shot_variance rel_err = 1.19` (fail).
- **Read σ fitted ≈ 0.0480** normalized vs measured temporal σ `1481/62335 = 0.0238` —
  **2× too high** (holdout check also failed).
- Read distribution family is Gaussian-or-Gamma only; the data is Tukey-lambda-heavy-tailed.
- Everything is fitted at **one gain (256)** while training/streaming uses ag128–512; noise
  parameters change with gain and with the HCG/LCG switch (the V4L2 driver flips
  `FDG_SEL0` above a gain threshold — a hard regime change, not a smooth curve).

### 3.2 Synthesis gaps (`nsa/noise_calib/synthesize.py`)

- Shot noise is a **Gaussian approximation** to Poisson. At gain 256, K ≈ 28 LSB₁₂/e⁻ —
  shadows hold only a handful of electrons, where Poisson is visibly discrete and skewed.
- No FPN/DSNU term, no BLE term, no dark-current term, quantization 8× too fine (above).
- Row noise: OK in shape (Gaussian ✓ per measurement) but derived from the mis-fitted σ.

### 3.3 Domain mismatch (decisive)

- `simulate_dataset.py` → `nsa/dataset_sim.py` loads clean images as **demosaiced RGB**, adds
  noise there, writes **8-bit PNGs**.
- `train_stream_to_gt.py` and the CFM pipeline train on **packed Bayer (H/2,W/2,4), linear
  float from 12-bit DNGs** and never consume the simulated PNGs at all.
- Demosaicing correlates noise across channels and pixels; 8-bit destroys the shadow
  statistics entirely. Any noise realism achieved before this step is lost.

### 3.4 The real-pairs training set is noise-perfect but scene-starved

`train_stream_to_gt.py` uses real bursts (noise realism = perfect by construction) but only
**2–3 static cabinet scenes**. That is the actual motivation for synthesis: you need *scene
diversity* with *matched noise*, i.e. thousands of clean images + exact noise injection.

---

## 4. Recommended pipeline (best given your data)

This is the SFRN → "Noise Modeling in One Hour" (Sony Research, 2025) → AIM 2025 challenge
baseline recipe, which currently beats parametric ELD-style models and even supervised
real-pair training on public benchmarks, adapted to your setup. It needs **one automated
capture session** and **no new modeling code beyond what is measured above**.

### 4.1 Noise formation model

Work per Bayer channel, in black-subtracted linear DN:

```
noisy = K_g · Poisson(clean / K_g)        ← signal-dependent (shot), K_g per gain g
      + D                                 ← signal-independent: a REAL dark frame,
                                            dark-shading-corrected, sampled from a library
```

- **Shot noise:** exact Poisson (not Gaussian approx). `K_g` comes from your flat-field PTC —
  measured 452.5 DN16/e⁻ (LCG g256), 2327.8 (HCG g256). K scales linearly with analog gain
  within a conversion-gain mode; verify with flats at 2–3 more gains, or fall back to the
  "hypothesized K" result from the Sony paper (denoisers are robust to ±2× K error — <0.1 dB).
- **Signal-independent noise = real dark frames.** A sampled dark frame automatically contains
  the heavy tails, row banding, quantization, hot pixels, and residual FPN with the exact
  spatial correlations — everything Section 2 shows a parametric model misses. No profiling,
  no distribution fitting.
- **Dark-shading correction (PMN):** compute the per-gain mean dark frame `D̄_g`; subtract it
  from dark frames before sampling (so the network sees only stochastic noise) and subtract it
  from real frames **at inference**. Handle BLE by re-estimating the black offset per frame
  (mean of masked/optical-black region, or mean of the frame's dark-shading residual).

### 4.2 Required new capture (automatable with `nsa_ctt_capture.py`, ~1–2 h total)

1. **Dark-frame library:** lens cap, operating exposure, **100–400 frames per gain** for every
   gain you deploy (ag64…512 minimum), both LCG and HCG. This is the single most valuable
   capture. (You currently have 5–8 per gain — enough to *fit* a Gaussian, not enough to
   *sample* from.)
2. **Flats at 2–3 additional gains** (one grey-card level pair each) to confirm K-vs-gain
   linearity and locate the HCG switch point.
3. **2–3 more static burst scenes** at ag128–512 (256+ frames) — held out for validation only
   (Section 5), never trained on.

### 4.3 Clean-image sources (the "massive" part)

Priority order:

1. **Own captures (best):** static scenes at ag1 with 64–128-frame averaging → true
   IMX662-domain clean Bayer frames. Every scene you shoot is worth more than any external
   image because CFA, optics, and ISP conventions match exactly. Even 30–50 varied scenes ×
   random crops is a large effective dataset.
2. **Unprocessed external datasets (scale):** take high-quality sRGB datasets (DIV2K, Flickr2K)
   and invert the ISP (Brooks et al., "Unprocessing images for learned raw denoising",
   CVPR 19): inverse tone curve → inverse gamma → inverse CCM → inverse WB → mosaic to RGGB →
   scale to 12-bit with black level 3200/16. This gives thousands of scenes in an
   IMX662-plausible raw domain.
3. **Existing bursts:** cabinet averages as a sanity anchor (but hold at least one scene out).

### 4.4 Dataset mechanics

- **Synthesize on the fly in the training dataloader**, not as files on disk: every epoch sees
  a fresh noise realization + a fresh dark frame → effectively infinite pairs, and storage
  stays at (clean images + dark library ≈ a few GB).
- Keep everything **float32 linear packed Bayer**, black-subtracted, `(H/2, W/2, 4)` — the
  exact tensor `train_stream_to_gt.py` already consumes. Never route through 8-bit PNG.
- Sample gain per crop (uniform over deployed gains), pick K_g and a dark frame from the
  matching gain/mode library. For the temporal-stack input (T=4), draw T independent noise
  realizations of the same clean frame — this also reproduces the temporally-fixed FPN
  correlation across the stack automatically if you reuse one dark frame's FPN component per
  window (dark frames sampled per-frame + shared `D̄_g` handling covers this).

### 4.5 Fallback if no new captures are possible

Fix the parametric path instead (weaker, but workable with today's 8 bias frames):
fit `var = a·μ + b` **with intercept**; per-gain parameters; Tukey-lambda read noise
(λ ≈ 0.04, σ from bias stack); Gaussian row noise σ ≈ 131 DN16; quant step 128 DN16;
add the measured FPN map (mean bias frame minus its mean) plus a per-frame BLE offset
(σ ≈ 50–140 DN16). This is essentially ELD + PMN. Expect it to trail the dark-frame-sampling
recipe by ~0.3–0.7 dB on the final denoiser (per NMIH/PNNP ablations), mostly in the tails.

---

## 5. Proving the noise profile is accurate (three tiers)

### Tier 1 — Statistical, on held-out calibration frames (fast, per gain)

Fix and extend the Phase-4 checks in `nsa/noise_calib/validate.py`:

| Check | Method | Pass target |
|---|---|---|
| PTC overlay | mean–variance curve of synthetic flats vs held-out real flat pairs, per channel | rel. var. error < 10 % per level |
| Noise histogram | discrete KL divergence real-vs-synth noise residual histograms (NoiseFlow protocol), per gain & channel | KLD ≈ real-vs-real baseline |
| Tails | P(\|n\|>3σ), P(\|n\|>5σ) real vs synth | same order of magnitude |
| Row structure | σ of per-row means + row-mean power spectrum | within 10 % |
| FPN | correlation of synth static component with measured FPN map | > 0.9 (trivially satisfied by dark-frame sampling) |
| Black level | per-frame mean drift distribution | covered by BLE model |

### Tier 2 — Perceptual, on burst scenes (aligned by construction)

For a held-out burst scene: GT = 512-frame average; take its real noisy frames and synthetic
noisy versions of the same GT. Compute **LPIPS(real, synth)** and compare against the
**LPIPS(real, real′)** baseline between two real realizations (SNIC protocol, 2025). PSNR/SSIM
are meaningless between two noise realizations; LPIPS is the accepted metric. Target:
relative LPIPS gap → 0.

### Tier 3 — Task-level (gold standard, the number that actually matters)

Train two identical denoisers with your existing `train_stream_to_gt.py` recipe:

- **A:** real pairs (burst frames → burst GT) from scenes 1–2.
- **B:** synthetic pairs (clean sources + noise model) — no real noisy data.

Evaluate **both on a held-out real burst scene** (never used for training or calibration).
The PSNR/SSIM gap of B vs A **is** the real-to-synthetic gap. Literature reference points:
a good calibrated model gets within **0.3–0.5 dB**; the dark-frame hybrid recipe has matched
or *beaten* real-pair supervision (misaligned pairs) on SID/ELD/LRID. Repeat per gain.

This tier reuses infrastructure you already have (bursts, training script, PSNR eval), so it
is nearly free to run.

---

## 6. Ranked alternatives considered

| Approach | Why not primary |
|---|---|
| Pure parametric ELD (Poisson + Tukey-λ + row + quant) | needs careful per-gain distribution fitting; still misses FPN spatial structure & true quant step; consistently below dark-frame sampling on benchmarks |
| Learned noise generators (NoiseFlow, LRD, NoiseDiff, GANs) | need real paired data at scale to train the generator — exactly what you lack; adds training instability; overkill for a single fixed sensor |
| Only real pairs (more bursts) | noise is perfect but scene diversity is capped by capture time; static-scene requirement biases content; keep as validation gold standard instead |
| DNG NoiseProfile metadata | documented to be inaccurate/incomplete (SNIC 2025); your Pi DNGs are written by the capture stack, not a tuned vendor profile |
| Buying/using public raw datasets (SID etc.) directly | different sensors, different K/read/row/FPN — the exact mismatch you're trying to avoid; usable only for pre-training (LED-style) before IMX662 fine-tune |

---

## 7. Concrete next steps (in order)

1. Capture the dark-frame library + extra flats + 2 held-out burst scenes (one automated session).
2. Implement on-the-fly Bayer-domain synthesis (Poisson via measured K_g + dark-frame sampling
   + dark-shading correction) as a dataloader for `train_stream_to_gt.py`.
3. Fix `fit_shot_poisson` (add intercept) and the validation thresholds regardless — the
   calibrated JSON is used elsewhere in the GUI/demo paths.
4. Build the clean-image pool: burst-average own scenes + unprocess DIV2K/Flickr2K into
   IMX662 RGGB 12-bit.
5. Run the Tier 1–3 validation; iterate until Tier 3 gap < 0.5 dB per gain.
6. Add dark-shading subtraction + per-frame BLE correction to `pi_stream_denoise.py` inference
   so deployment matches the training-time correction.

---

## Appendix — measurement provenance

All statistics computed with `rawpy` on `raw_image_visible` (uint16, black 3200):

- Read noise: `std(bias_stack − mean(bias_stack))` over 8 (LCG) / 8 (HCG) bias frames.
- PTC: per flat level, `μ = mean((a+b)/2 − 3200)` and `σ² = var(a−b)/2` on unclipped pixels;
  least-squares line → slope K, intercept read variance. Per-channel fits agree within 3 %.
- Tails: empirical exceedance probabilities of temporal residuals; `scipy.stats.ppcc_max`
  Tukey-lambda fit; kurtosis 2.09 (excess).
- Row noise: σ of per-row means of temporal residual vs i.i.d. expectation σ/√W.
- FPN: spatial std of the 8-frame mean bias minus global mean; row/col-removed std ≈ unchanged
  → per-pixel pattern.
- BLE: per-frame bias means 3204.3 / 3341.7 / … (range ≈ 137 DN16).
- Quantization: `np.diff(np.unique(frame))` → 128 DN16 spacing.
- Bursts: frame-to-frame diff std stable first→last (2141.9 vs 2149.4) → static scenes, safe
  to average for GT.
