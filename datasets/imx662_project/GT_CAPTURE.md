# Ground truth for IMX662 synthesis

## Option A — reuse manager GT (fastest)

In Dataset Studio click **USE EXISTING GT** — copies ``gt.dng``/``gt.png`` from each
scene's best PI_RAW folder (e.g. imx219_ag12_test) into ``clean_scenes/<scene>/``.

## Option B — temporal burst (best quality)

1. Tripod, static scene, 32–128 RAW frames → ``bursts/<scene>/take01/``
2. ``python capture_gt.py --burst bursts/<scene>/take01 --output clean_scenes/<scene>/gt_01.png``

## Then synthesize

1. ``python calibrate_noise.py -i calibration/imx662_gain256``
2. ``python simulate_dataset.py -i clean_scenes -o PI_RAW --calibration models/noise/….json``
