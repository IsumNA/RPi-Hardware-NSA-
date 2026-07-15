# Conditional flow matching — the anti-blur path

## Problem
Regression denoisers (NAFNet, RawDenoiser, any L1/Charbonnier/SWT net) predict
**E[clean|noisy]** → soft middle column on your panels. You already pushed that
family as far as it goes (`charbonnier+swt`, RAW domain, burst diversity).

## Solution
**Conditional rectified flow** in packed RAW:

- **Condition** `y` = one noisy burst frame (deploy = single frame → motion OK)
- **Target** `x0` = multi-frame average GT (bursts only for supervision)
- **Train** velocity noise→clean given `y`
- **Infer** sample with ~8–10 Heun ODE steps (not a posterior mean)

```bash
# AI server
python train_flow_raw.py \
  --bursts datasets/imx662_project/bursts \
  --gains 128 256 512 \
  --gt-frames 100 \
  --steps 12000 \
  --out outputs/flow_raw

# one live / moving frame
python train_flow_raw.py --infer path/to/noisy.dng \
  --ckpt outputs/flow_raw/flow_raw.pt \
  --out outputs/flow_denoised.png
```

Synth check:

```bash
python train_flow_raw.py --proof --steps 600 --out outputs/flow_proof
```

Panel columns: **noisy | blur-proxy | FLOW sample | GT**

## What stayed from earlier (still useful)
- Linear DNG load (no gamma ISP mismatch)
- Burst-frame training diversity
- HCG recovery / pair plumbing

Those feed this trainer; they are not a substitute for leaving MMSE.
