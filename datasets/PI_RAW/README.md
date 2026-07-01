# PI_RAW test captures (denoise-hw layout)

This folder mirrors the layout used by [davidplowman/denoise-hw](https://github.com/davidplowman/denoise-hw):

```
PI_RAW/Data/<scene>/<sensor_test>/noisy.{dng,png}
PI_RAW/Data/<scene>/<sensor_test>/gt.{dng,png}
```

## Quick start

```bash
python setup_denoise_hw_data.py          # build sample PNG pairs (or link real PI_RAW)
python run_demo.py --no-window           # uses config.yaml → real captures here
```

## Real Raspberry Pi captures

On a Pi with the full dataset at `/opt/datasets/PI_RAW`:

```bash
python setup_denoise_hw_data.py --link /opt/datasets/PI_RAW
```

Or set `PI_RAW=/path/to/PI_RAW` in the environment.

The canonical denoise-hw test folder is `Data/cabinet_D50_100/imx219_ag12_test`
(same path as `test.py` in denoise-hw's `run.sh`).
