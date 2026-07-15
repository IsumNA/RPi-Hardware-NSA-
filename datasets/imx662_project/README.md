# IMX662 noise synthesis project

## What is already on disk (manager dataset)

Your team's real captures live in **PI_RAW/Data/** — do not delete or move them::

    PI_RAW/Data/
      cabinet_D50_100/   imx219_ag1_test … imx219_ag12_test  (noisy.* + gt.*)
      cabinet_F11_25/
      cabinet_H_10/
      colour_stripes/

Each test folder contains up to four files: ``noisy.dng``, ``noisy.png``, ``gt.dng``, ``gt.png``.

## What you add for the noise pipeline

| Folder | Purpose |
|--------|---------|
| ``calibration/imx662_gain256/`` | NEW bias/dark/flat shoots → noise model JSON |
| ``clean_scenes/<scene>/`` | Clean GT for synthesis (copy from PI_RAW gt.* or burst average) |
| ``PI_RAW/Data/<scene>/imx662_ag24_test/`` | **Generated** pairs (night-vision tags: ag12, ag24, ag48) |

Legacy IMX219 tags: ag1, ag2, ag4, ag8, ag12. IMX662 low-light likely needs higher tags (ag12, ag24, ag48).

Open **Dataset Studio** in the NSA GUI and point **PI_RAW root** at this folder.

Scenes: cabinet_D50_100, cabinet_F11_25, cabinet_H_10, colour_stripes
