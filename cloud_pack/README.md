# Cloud training pack

Compact overnight kit — **no DNGs required**.

## Contents

| Path | Purpose |
|------|---------|
| `checkpoints/cfm_edm_teacher.pt` | EDM teacher (128ch) |
| `checkpoints/cfm_l1_student_best.pt` | Strong baseline student |
| `checkpoints/perfect_r1_student_best.pt` | Latest sharp mid-run student |
| `data/*.npz` | 8 train + 2 eval packed Bayer stacks (float16) |
| `noise/*.json` | IMX662 gain noise models |

## Run on cloud GPU

```bash
python -u train_cfm_distill.py \
  --teacher cloud_pack/checkpoints/cfm_edm_teacher.pt \
  --init-student cloud_pack/checkpoints/perfect_r1_student_best.pt \
  --pack-dir cloud_pack/data \
  --method consistency \
  --sample-loss l1_hf \
  --gt-hf-weight 0.45 \
  --cd-weight 0 --restore-best --no-heun \
  --steps 4000 --batch 4 --crop 256 \
  --integrate-steps 4 \
  --panel-every 200 \
  --panel-dir outputs/cloud_panels \
  --out outputs/cloud_sharp
```

Or: `bash scripts/run_cloud_overnight.sh`

Judge success visually: middle panel ≈ right (GT), not soft/waxy.
