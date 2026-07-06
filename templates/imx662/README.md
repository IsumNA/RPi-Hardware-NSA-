# IMX662 dataset template

Run this once to create the full folder tree with capture guides on disk:

```bash
python scaffold_imx662.py --output datasets/imx662_project
```

On the AI server (if datasets live under `/opt/datasets/`):

```bash
python scaffold_imx662.py -o /opt/datasets/imx662_project
```

Then open **Dataset Studio** in the NSA GUI (`DATASET STUDIO` on the home screen)
to see what to shoot, what is already present, and thumbnail previews.

## Folder map

| Path | What goes here |
|------|----------------|
| `calibration/imx662_gain256/bias/` | Lens cap, min exposure, 5+ frames |
| `calibration/imx662_gain256/dark/` | Lens cap, normal exposure at target gain |
| `calibration/imx662_gain256/flat/level_XX/` | Uniform light pairs `a` + `b` |
| `bursts/<scene>/take01/` | Sequential RAW burst before GT averaging |
| `clean_scenes/<scene>/` | Temporally-averaged GT stills |
| `PI_RAW/Data/<scene>/imx662_ag12_test/` | `noisy.png` + `gt.png` training pairs |

See `GT_CAPTURE.md` in the scaffolded project for proper ground-truth capture.
