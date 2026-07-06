# IMX662 + existing PI_RAW dataset

## What your manager already captured (do not move)

```
PI_RAW/Data/
  cabinet_D50_100/
    imx219_ag1_test/   noisy.dng  noisy.png  gt.dng  gt.png
    imx219_ag2_test/   …
    imx219_ag4_test/   …
    imx219_ag8_test/   …
    imx219_ag12_test/  …
  cabinet_F11_25/      (same pattern)
  cabinet_H_10/
  colour_stripes/
```

`agN` is a **denoise-hw folder tag** (not the sensor register). IMX219 work used ag1–ag12.

## What you add for IMX662 noise synthesis

| Path | What |
|------|------|
| `calibration/imx662_gain256/` | NEW bias/dark/flat lab captures → noise model JSON |
| `clean_scenes/<scene>/` | Clean GT for synthesis — **USE EXISTING GT** copies `gt.*` from PI_RAW |
| `PI_RAW/Data/<scene>/imx662_ag24_test/` | **Generated** noisy+gt (night tags: ag12, ag24, ag48) |

Scaffold beside your existing PI_RAW:

```bash
python scaffold_imx662.py -o /opt/datasets    # parent of PI_RAW
```

Open **Dataset Studio** → point at `/opt/datasets/PI_RAW`.
