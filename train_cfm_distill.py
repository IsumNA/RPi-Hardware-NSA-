#!/usr/bin/env python3
"""Consistency Flow Matching distillation → 1-step Pi student.

The CFM teacher draws sharp textures with a multi-step ODE. The student is a
time-conditioned ConsistencyStudent distilled so that
    f_θ(x_t, t, cond) ≈ teacher_integrate(x_t, t→1, cond)
(plus optional classic EMA consistency distillation). At inference the Pi
evaluates only the boundary via BoundaryConsistencyWrapper::

    x₀ = noisy frame, t = 0  →  one forward, RawDenoiser I/O (cond 4T → packed 4).

Ablation ``--method regression_match`` restores the old L1-to-teacher-samples
RawDenoiser recipe.

Run on AI GPU after ``train_cfm_teacher.py``::

  .venv/bin/python -u train_cfm_distill.py \\
      --teacher outputs/cfm_teacher.pt \\
      --method consistency --steps 8000 --channels 64 --depth 6 --temporal 4
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nsa.flow_matching import (
    BoundaryConsistencyWrapper,
    ConsistencyStudent,
    FlowVelocityNet,
    append_gain_channel,
    consistency_distillation_loss,
    consistency_flow_matching_loss,
    euler_sample,
    grad_ratio,
    make_ema,
    update_ema,
)
from nsa.inference import psnr, ssim, to_image, to_tensor
from nsa.lite_cfm_student import LiteCfmStudent, build_cfm_student, wrap_deploy
from nsa.raw_domain import RawDenoiser
from train_stream_to_gt import (
    DEFAULT_GAINS,
    DEFAULT_SCENES,
    _export_onnx,
    build_pairs,
)
from train_cfm_teacher import (
    DISPLAY_GAIN,
    _rgb,
    _rgb_t,
    _sample_cond_clean,
    _load_dump_pairs,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_teacher(path: Path, device: torch.device) -> tuple[FlowVelocityNet, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("model", {})
    edm = bool(meta.get("edm_precond", False))
    gain_film = bool(meta.get("gain_film", blob.get("gain_film", False)))
    model = FlowVelocityNet(
        cond_ch=int(meta.get("cond_ch", 16)),
        out_ch=int(meta.get("out_ch", 4)),
        base_channels=int(meta.get("base_channels", 128)),
        block_depth=int(meta.get("block_depth", 8)),
        edm_precond=edm,
        sigma_data=float(meta.get("sigma_data") or 0.05),
        sigma_flow=float(meta.get("sigma_flow") or 0.06),
        gain_film=gain_film,
    )
    model.load_state_dict(blob["state_dict"])
    if edm:
        print(f"Teacher uses EDM precond (sigma_data={meta.get('sigma_data'):.5f}, "
              f"sigma_flow={meta.get('sigma_flow'):.5f}) — "
              "distill targets use Heun ρ=7", flush=True)
    if gain_film:
        print("Teacher uses analogue-gain FiLM (log2(gain/128))", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob


def _prep_student_cond(
    cond: torch.Tensor,
    gain: torch.Tensor | None,
    *,
    gain_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append gain channel for deploy I/O when student expects it."""
    if gain_channel and gain is not None:
        return append_gain_channel(cond, gain), gain
    return cond, gain


def _cond_for_eval(noisy: np.ndarray, ev: dict, device, *, gain_channel: bool):
    cond = to_tensor(noisy).to(device)
    if not gain_channel:
        return cond
    g = torch.tensor([float(ev["gain"])], device=device, dtype=torch.float32)
    return append_gain_channel(cond, g)


def _inv_luma_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-3,
    w_min: float = 0.5,
    w_max: float = 8.0,
) -> torch.Tensor:
    """Optional L1 vs teacher with per-pixel inverse-luminance weights.

    Dark pixels get higher weight so soft shadows/walls are penalised more.
    Opt-in via --sample-loss inv_luma / --inv-luma. Prefer final (last)
    weights over early best-score checkpoints, which can overshoot grain.
    """
    luma = target.detach().mean(dim=1, keepdim=True)
    w = (1.0 / (luma + eps)).clamp(w_min, w_max)
    w = w / w.mean().clamp_min(eps)
    return (w * (pred - target).abs()).mean()


def _highfreq_l1(pred: torch.Tensor, target: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Match detail bands: (x - blur(x)). Encourages texture without MMSE blur."""
    return F.l1_loss(pred - _lowpass(pred, k), target - _lowpass(target, k))


def _l1_hf(pred: torch.Tensor, target: torch.Tensor, hf_w: float = 0.35) -> torch.Tensor:
    """Plain L1 on sharp teacher samples + high-frequency texture match."""
    return F.l1_loss(pred, target) + float(hf_w) * _highfreq_l1(pred, target)


def _spatial_grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match finite-difference gradients pixelwise (can blur if target phase differs)."""
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def _grad_energy_match(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match mean |∇| only (scale) — targets grad_ratio≈1 without phase lock."""
    from nsa.flow_matching import grad_energy

    gp = grad_energy(pred)
    gt = grad_energy(target).detach().clamp_min(eps)
    return ((gp / gt) - 1.0).abs()


def _lowfreq_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernels: tuple[int, ...] = (16, 32),
) -> torch.Tensor:
    """Penalise low-frequency deviation from GT — the flat-region 'blotch' /
    slow colour drift the eye integrates but plain L1 ignores (measured
    amplitude ≈0.003–0.009, invisible to per-pixel L1). Compares non-overlapping
    block means at several large scales. Because it's a difference vs the
    target, legitimate low-freq shading/vignetting present in GT is preserved —
    only blotches GT lacks are punished."""
    total = pred.new_zeros(())
    n = 0
    for k in kernels:
        ks = int(min(k, pred.shape[-1], pred.shape[-2]))
        if ks < 2:
            continue
        total = total + F.l1_loss(F.avg_pool2d(pred, ks), F.avg_pool2d(target, ks))
        n += 1
    return total / max(n, 1)


def _l1_grad(
    pred: torch.Tensor,
    target: torch.Tensor,
    hf_w: float = 0.25,
    grad_w: float = 0.55,
) -> torch.Tensor:
    """L1 + high-freq + spatial-grad match — push texture toward grad_ratio≈1."""
    return (
        F.l1_loss(pred, target)
        + float(hf_w) * _highfreq_l1(pred, target)
        + float(grad_w) * _spatial_grad_l1(pred, target)
    )


def _packed_rgb_t(pk: torch.Tensor) -> torch.Tensor:
    """B×4×H×W packed → RGB for LPIPS (matches teacher eval path)."""
    r = pk[:, 0:1]
    g = 0.5 * (pk[:, 1:2] + pk[:, 2:3])
    b = pk[:, 3:4]
    return torch.clamp(torch.cat([r, g, b], dim=1) * DISPLAY_GAIN, 0.0, 1.0)


def _l1_lpips(
    pred: torch.Tensor, target: torch.Tensor, *, lpips_w: float = 0.1,
) -> torch.Tensor:
    """L1 on packed 4ch + LPIPS on display-gain RGB."""
    from nsa.inference import _perceptual_loss
    return (
        F.l1_loss(pred, target)
        + float(lpips_w) * _perceptual_loss(
            _packed_rgb_t(pred), _packed_rgb_t(target))
    )


def _make_sample_loss(*, name: str = "l1") -> tuple:
    """Teacher-endpoint match loss. Default plain L1 matches cfm_l1 baseline.

    Composite texture losses (charbonnier/edge/swt/swtrel) and inverse-luma
    weighting were A/B'd and underperformed plain L1 — keep them opt-in only.
    ``l2``/``mse`` is a fair ablation against the same CFM-CD recipe.
    ``l1_hf`` adds a high-frequency L1 term (detail matching) to push texture
    without the over-smooth of swtrel.
    ``l1_grad`` adds explicit ∇ match (best for targeting grad_ratio≈1).
    ``l1_lpips05`` softens LPIPS (less fake painted texture) vs ``l1_lpips``.
    """
    key = (name or "l1").strip().lower()
    if key in ("inv_luma", "inv_luma_l1", "inv-luma"):
        print("Sample-match loss: inv_luma_l1", flush=True)
        return _inv_luma_l1, "inv_luma_l1"
    if key in ("l1_grad", "l1+grad", "l1grad"):
        print("Sample-match loss: l1_grad (L1 + HF×0.25 + ∇×0.55)", flush=True)
        return _l1_grad, "l1_grad"
    if key in ("l1_hf", "l1+hf", "l1hf"):
        print("Sample-match loss: l1_hf (L1 + high-freq L1×0.35)", flush=True)
        return _l1_hf, "l1_hf"
    if key in ("l1_lpips05", "l1_lpips_soft", "l1lpips05"):
        print("Sample-match loss: l1_lpips05 (L1 + LPIPS×0.05 on RGB)", flush=True)
        return (lambda a, b: _l1_lpips(a, b, lpips_w=0.05)), "l1_lpips05"
    if key in ("l1_lpips", "l1+lpips", "l1lpips"):
        print("Sample-match loss: l1_lpips (L1 + LPIPS×0.1 on RGB)", flush=True)
        return (lambda a, b: _l1_lpips(a, b, lpips_w=0.1)), "l1_lpips"
    if key in ("l2", "mse"):
        print("Sample-match loss: l2 (mse)", flush=True)
        return (lambda a, b: F.mse_loss(a, b)), "l2"
    print("Sample-match loss: l1", flush=True)
    return (lambda a, b: F.l1_loss(a, b)), "l1"


def load_cloud_pack(pack_data: Path) -> tuple[
    list[tuple[np.ndarray, np.ndarray]], list[dict], dict,
]:
    """Load ``cloud_pack/data`` npz pairs (no DNGs needed on cloud)."""
    man_path = pack_data / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"Missing pack manifest: {man_path}")
    man = json.loads(man_path.read_text())
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pair_gains: list[int] = []
    for rec in man.get("pairs", []):
        blob = np.load(pack_data / rec["file"])
        noisy = np.asarray(blob["noisy"], dtype=np.float32)
        clean = np.asarray(blob["clean"], dtype=np.float32)
        gain = int(blob["gain"]) if "gain" in blob.files else int(rec.get("gain", 128))
        pairs.append((noisy, clean))
        pair_gains.append(gain)
    evals: list[dict] = []
    for rec in man.get("evals", []):
        blob = np.load(pack_data / rec["file"])
        noisy = np.asarray(blob["noisy"], dtype=np.float32)
        clean = np.asarray(blob["clean"], dtype=np.float32)
        frame = int(blob["frame"]) if "frame" in blob.files else int(rec.get("frame", 0))
        gain = int(blob["gain"]) if "gain" in blob.files else int(rec.get("gain", 128))
        scene = rec.get("scene") or (
            str(blob["scene"]) if "scene" in blob.files else "pack")
        evals.append({
            "scene": scene,
            "gain": gain,
            "gt": clean,
            "noisy": [(frame, noisy)],
        })
    if not pairs:
        raise RuntimeError(f"No pairs in {man_path}")
    meta = {
        "pair_gains": pair_gains,
        "pack": str(pack_data),
        "temporal": int(man.get("temporal", 4)),
    }
    print(f"Cloud pack: {len(pairs)} train pairs, {len(evals)} evals "
          f"from {pack_data}", flush=True)
    return pairs, evals, meta


def _load_dump_tensors(
    dump_root: Path,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    """Load (cond, target) pairs + parallel gains from dump/synth shards.

    Accepts either dump keys ``cond``/``teacher`` or synth keys ``noisy``/``gt``.
    Gain comes from the npz ``gain`` field or the index record (default 128).
    """
    idx_path = dump_root / "index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing dump index: {idx_path}")
    index = json.loads(idx_path.read_text())
    tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    pair_gains: list[int] = []
    n_dump = n_synth = 0
    for rec in index.get("samples", []):
        rel = rec.get("file") or rec.get("path")
        if not rel:
            continue
        path = dump_root / rel
        if path.suffix != ".npz":
            raise ValueError(f"Unsupported dump file: {path}")
        blob = np.load(path)
        keys = set(blob.files)
        if "cond" in keys and "teacher" in keys:
            cond = np.asarray(blob["cond"], dtype=np.float32)
            teacher = np.asarray(blob["teacher"], dtype=np.float32)
            n_dump += 1
        elif "noisy" in keys and "gt" in keys:
            cond = np.asarray(blob["noisy"], dtype=np.float32)
            teacher = np.asarray(blob["gt"], dtype=np.float32)
            n_synth += 1
        else:
            raise ValueError(
                f"Unsupported npz keys in {path}: {sorted(keys)} "
                f"(need cond/teacher or noisy/gt)")
        if cond.ndim == 4:
            cond = cond.squeeze(0).transpose(1, 2, 0)
        if teacher.ndim == 4:
            teacher = teacher.squeeze(0).transpose(1, 2, 0)
        if "gain" in keys:
            gain = int(np.asarray(blob["gain"]).reshape(-1)[0])
        else:
            gain = int(rec.get("gain", 128))
        tensors.append((to_tensor(cond), to_tensor(teacher)))
        pair_gains.append(gain)
    if not tensors:
        raise RuntimeError(f"No samples in {idx_path}")
    print(
        f"Loaded {len(tensors)} pairs from {dump_root} "
        f"(teacher_dumps={n_dump}, synth={n_synth})",
        flush=True,
    )
    return tensors, pair_gains


def _warm_start_expand_head(
    student: nn.Module,
    istate: dict,
) -> dict:
    """Copy checkpoint weights; expand head when adding a gain channel."""
    sw = student.state_dict()
    out = {k: v for k, v in istate.items() if k in sw}
    if "head.weight" not in out or "head.weight" not in sw:
        return out
    ow = out["head.weight"]
    nw = sw["head.weight"]
    if ow.shape == nw.shape:
        return out
    if (
        ow.ndim == 4 and nw.ndim == 4
        and ow.shape[0] == nw.shape[0]
        and ow.shape[2:] == nw.shape[2:]
        and nw.shape[1] == ow.shape[1] + 1
    ):
        expanded = nw.clone()
        expanded[:, : ow.shape[1]] = ow
        # New gain channel starts at 0 so warm-start matches no-gain behavior.
        expanded[:, ow.shape[1] :] = 0
        out["head.weight"] = expanded
        print(
            f"Warm-start: expanded head {tuple(ow.shape)} → {tuple(nw.shape)} "
            "(gain ch zeros)",
            flush=True,
        )
    return out


def _train_regression_from_dumps(
    student: RawDenoiser,
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    sample_loss=None,
    sample_loss_name: str = "l1",
    best_path: Path | None = None,
    early_abort_soft: bool = False,
    early_abort_after: int = 200,
    pair_gains: list[int] | None = None,
    gain_channel: bool = False,
    gt_hf_weight: float = 0.0,
    gt_grad_energy_weight: float = 0.0,
    gt_lowfreq_weight: float = 0.0,
) -> RawDenoiser:
    """Stage A: match fixed offline teacher ODE outputs (no live teacher forward)."""
    wts = torch.tensor(
        [1.0 / max(float(n[..., :4].mean()), 1e-3) for n, _ in tensors],
        dtype=torch.float32)
    wts = (wts / wts.mean()).clamp(0.25, 8.0)
    use_gain = bool(gain_channel) and pair_gains is not None
    gt_hf_weight = float(gt_hf_weight)
    gt_grad_energy_weight = float(gt_grad_energy_weight)
    gt_lowfreq_weight = float(gt_lowfreq_weight)

    if sample_loss is None:
        sample_loss, sample_loss_name = _make_sample_loss()
    print(
        f"Distill loss: {sample_loss_name} vs offline teacher dumps"
        + (f" + GT-HF×{gt_hf_weight}" if gt_hf_weight > 0 else "")
        + (f" + GT-|∇|×{gt_grad_energy_weight}" if gt_grad_energy_weight > 0 else "")
        + (f" + GT-LF×{gt_lowfreq_weight}" if gt_lowfreq_weight > 0 else "")
        + ("  [gain channel]" if use_gain else ""),
        flush=True,
    )
    if early_abort_soft:
        print(f"Early-abort soft: after step {early_abort_after} if best≤100 "
              f"and probe_gr softens below 0.97", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(664)
    student = student.to(device)
    student.train()
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_score = -1.0
    best_state = None
    best_step = 0
    skipped = 0
    soft_streak = 0

    for i in range(steps):
        if use_gain:
            cond, target, gain = _sample_cond_clean(
                tensors, crop, batch, g, wts, pair_gains)
            cond, gain = _prep_student_cond(cond, gain, gain_channel=True)
            cond = cond.to(device)
            target = target.to(device)
        else:
            cond, target = _sample_cond_clean(tensors, crop, batch, g, wts)
            cond, target = cond.to(device), target.to(device)
        opt.zero_grad(set_to_none=True)
        pred = student(cond)
        loss = sample_loss(pred, target)
        # Synth/dump targets are clean GT — HF / |∇| push sharpness (grad≈1).
        if gt_hf_weight > 0:
            loss = loss + gt_hf_weight * _highfreq_l1(pred, target)
        if gt_grad_energy_weight > 0:
            loss = loss + gt_grad_energy_weight * _grad_energy_match(pred, target)
        # Flat-region blotch / colour-drift suppression (see _lowfreq_l1).
        if gt_lowfreq_weight > 0:
            loss = loss + gt_lowfreq_weight * _lowfreq_l1(pred, target)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        opt.step()
        sched.step()
        if i % 50 == 0 or i == steps - 1:
            print(f"  step {i+1}/{steps}  loss={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/max(i,1):.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            panel_ev = next((e for e in evals if e.get("gain", 0) >= 256), evals[0])
            pout, gr = _save_eval_panel(
                student, panel_ev, device, panel_dir, i + 1,
                gain_channel=gain_channel)
            probe, probe_gr = _quick_probe(
                student, evals[:6], device, gain_channel=gain_channel)
            print(f"  probe mean PSNR={probe:.2f} dB  grad_ratio={probe_gr:.3f}",
                  flush=True)
            # Prefer sharpness (grad_ratio≈1) over PSNR — PSNR alone picks mush
            score = (0.25 * pout + 0.25 * probe
                     + 0.5 * (20.0 * max(0.0, 1.0 - abs(probe_gr - 1.0))))
            if score > best_score:
                best_score = score
                best_step = i + 1
                soft_streak = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in student.state_dict().items()}
                if best_path is not None:
                    torch.save({
                        "state_dict": best_state,
                        "step": i + 1,
                        "score": best_score,
                        "panel_psnr": pout,
                        "probe_psnr": probe,
                        "probe_grad_ratio": probe_gr,
                        "gain_channel": bool(gain_channel),
                    }, best_path)
                print(f"  ★ best@{i+1}: panel={pout:.2f} probe={probe:.2f} "
                      f"grad_r={probe_gr:.3f} score={best_score:.2f}", flush=True)
            elif probe_gr < 0.97:
                soft_streak += 1
            # Skip the rest when early peak already happened and we're mushing.
            if (early_abort_soft
                    and (i + 1) >= int(early_abort_after)
                    and best_step > 0 and best_step <= 100
                    and soft_streak >= 2):
                print(f"  early-abort soft @step {i+1}: best@{best_step} "
                      f"probe_gr={probe_gr:.3f} soft_streak={soft_streak} "
                      f"— skipping likely reject", flush=True)
                break
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"Restored best student (score={best_score:.2f})", flush=True)
    if skipped:
        print(f"Skipped {skipped} non-finite steps", flush=True)
    student.eval()
    return student


def _lowpass(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cheap box blur — keep colour / large structure aligned to GT."""
    pad = k // 2
    c = x.shape[1]
    w = torch.ones(c, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    return F.conv2d(x, w, padding=pad, groups=c)


def _deploy_model(student: nn.Module, method: str) -> nn.Module:
    if method == "consistency":
        return BoundaryConsistencyWrapper(student)  # type: ignore[arg-type]
    return student


def _train_consistency(
    student: nn.Module,
    teacher: FlowVelocityNet,
    pairs,
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    integrate_steps: int,
    cd_weight: float,
    cd_intervals: int,
    ema_decay: float,
    gt_weight: float,
    heun: bool,
    sample_loss=None,
    sample_loss_name: str = "mse",
    best_path: Path | None = None,
    pair_gains: list[int] | None = None,
    restore_best: bool = False,
    gt_grad_weight: float = 0.0,
    gt_grad_energy_weight: float = 0.0,
    gt_hf_weight: float = 0.0,
) -> nn.Module:
    tensors = [(to_tensor(n), to_tensor(c[..., :4] if c.shape[-1] > 4 else c))
               for n, c in pairs]
    wts = torch.tensor(
        [1.0 / max(float(n[..., :4].mean()), 1e-3) for n, _ in pairs],
        dtype=torch.float32)
    # Bias toward darker / high-gain frames (wider clamp for dark texture).
    wts = (wts / wts.mean()).clamp(0.25, 8.0)
    gain_channel = bool(getattr(student, "gain_channel", False))
    use_gain = pair_gains is not None and (
        gain_channel or bool(getattr(teacher, "gain_film", False)))

    print(
        "Distill loss: Consistency Flow Matching "
        f"(teacher integrate×{integrate_steps}, heun={heun}, "
        f"endpoint match={sample_loss_name})"
        + (f" + classic CD×{cd_weight}" if cd_weight > 0 else "")
        + (f" + lowpass GT×{gt_weight}" if gt_weight > 0 else "")
        + (f" + GT-∇×{gt_grad_weight}" if gt_grad_weight > 0 else "")
        + (f" + GT-|∇|×{gt_grad_energy_weight}" if gt_grad_energy_weight > 0 else "")
        + (f" + GT-HF×{gt_hf_weight}" if gt_hf_weight > 0 else "")
        + ("  [gain channel→FiLM]" if gain_channel else ""),
        flush=True,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(663)
    student = student.to(device)
    student.train()
    ema = make_ema(student) if cd_weight > 0 else None
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_score = -1.0
    best_state = None
    skipped = 0
    deploy = _deploy_model(student, "consistency")

    for i in range(steps):
        if use_gain:
            cond, clean, gain = _sample_cond_clean(
                tensors, crop, batch, g, wts, pair_gains)
            gain = gain.to(device)
        else:
            cond, clean = _sample_cond_clean(tensors, crop, batch, g, wts)
            gain = None
        cond, clean = cond.to(device), clean.to(device)
        cond_s, gain_s = _prep_student_cond(
            cond, gain, gain_channel=gain_channel)
        opt.zero_grad(set_to_none=True)
        loss = consistency_flow_matching_loss(
            student, teacher, clean, cond_s, gain=gain_s,
            integrate_steps=integrate_steps, heun=heun, fixed_noise=True,
            sample_loss=sample_loss)
        if cd_weight > 0 and ema is not None:
            loss = loss + cd_weight * consistency_distillation_loss(
                student, ema, teacher, clean, cond_s, gain=gain_s,
                num_intervals=cd_intervals, ode_steps=1, heun=heun,
                fixed_noise=True)
        if (gt_weight > 0 or gt_grad_weight > 0 or gt_grad_energy_weight > 0
                or gt_hf_weight > 0):
            # Deploy boundary (x₀=noisy, t=0) — GT terms measured here
            boundary = deploy(cond_s)
            if gt_weight > 0:
                # Colour lock only (low-pass); avoids MMSE blur from full GT L1
                loss = loss + gt_weight * F.l1_loss(
                    _lowpass(boundary), _lowpass(clean))
            if gt_grad_weight > 0:
                # Pixelwise ∇ match (can oversmooth if GT phase ≠ teacher)
                loss = loss + gt_grad_weight * _spatial_grad_l1(boundary, clean)
            if gt_grad_energy_weight > 0:
                # Scale-only |∇| match → grad_ratio≈1 (can restore grain not texture)
                loss = loss + gt_grad_energy_weight * _grad_energy_match(
                    boundary, clean)
            if gt_hf_weight > 0:
                # High-pass band vs GT — real edge/texture structure, not grain energy
                loss = loss + gt_hf_weight * _highfreq_l1(boundary, clean)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        opt.step()
        sched.step()
        if ema is not None:
            update_ema(ema, student, decay=ema_decay)
        if i % 50 == 0 or i == steps - 1:
            print(f"  step {i+1}/{steps}  loss={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/max(i,1):.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            panel_ev = next((e for e in evals if e.get("gain", 0) >= 256), evals[0])
            pout, gr = _save_eval_panel(
                deploy, panel_ev, device, panel_dir, i + 1,
                gain_channel=gain_channel)
            probe, probe_gr = _quick_probe(
                deploy, evals[:6], device, gain_channel=gain_channel)
            print(f"  probe mean PSNR={probe:.2f} dB  grad_ratio={probe_gr:.3f}",
                  flush=True)
            # grad_ratio → 1.0 scores 20; blur or residual noise both penalised
            # Prefer sharpness (grad_ratio≈1) over PSNR — PSNR alone picks mush
            score = (0.25 * pout + 0.25 * probe
                     + 0.5 * (20.0 * max(0.0, 1.0 - abs(probe_gr - 1.0))))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in student.state_dict().items()}
                if best_path is not None:
                    torch.save({
                        "state_dict": best_state,
                        "step": i + 1,
                        "score": best_score,
                        "panel_psnr": pout,
                        "probe_psnr": probe,
                        "probe_grad_ratio": probe_gr,
                    }, best_path)
                print(f"  ★ best@{i+1}: panel={pout:.2f} probe={probe:.2f} "
                      f"grad_r={probe_gr:.3f} score={best_score:.2f}", flush=True)
    # Persist last weights. Default keeps last (early "best" can overshoot grain
    # for inv-luma). Pass restore_best=True when targeting grad_ratio≈1.
    if best_path is not None:
        last_path = best_path.with_name("cfm_student_last.pt")
        torch.save({
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in student.state_dict().items()},
            "step": steps,
            "score": None,
            "note": "final weights",
        }, last_path)
        print(f"Saved last student → {last_path}", flush=True)
    if best_state is not None:
        if restore_best:
            student.load_state_dict(best_state)
            print(f"Restored best score={best_score:.2f} from {best_path} "
                  f"(grad_ratio-aware)", flush=True)
        else:
            print(f"Best score={best_score:.2f} kept at {best_path}; "
                  f"using final weights (not restoring best)", flush=True)
    if skipped:
        print(f"Skipped {skipped} non-finite steps", flush=True)
    student.eval()
    return student


def _train_regression_match(
    student: RawDenoiser,
    teacher: FlowVelocityNet,
    pairs,
    steps: int,
    *,
    crop: int,
    batch: int,
    lr: float,
    device: torch.device,
    panel_every: int,
    panel_dir: Path,
    evals: list[dict],
    teacher_steps: int,
    gt_weight: float,
    sample_loss=None,
    sample_loss_name: str = "l1",
    best_path: Path | None = None,
    pair_gains: list[int] | None = None,
    gain_channel: bool = False,
) -> RawDenoiser:
    """Ablation: match teacher Euler samples with a composite loss (old recipe)."""
    tensors = [(to_tensor(n), to_tensor(c[..., :4] if c.shape[-1] > 4 else c))
               for n, c in pairs]
    wts = torch.tensor(
        [1.0 / max(float(n[..., :4].mean()), 1e-3) for n, _ in pairs],
        dtype=torch.float32)
    wts = (wts / wts.mean()).clamp(0.25, 8.0)
    use_gain = bool(gain_channel) and pair_gains is not None

    if sample_loss is None:
        sample_loss, sample_loss_name = _make_sample_loss()
    print(
        f"Distill loss: {sample_loss_name} vs teacher samples "
        "(regression_match ablation)"
        + ("  [gain channel]" if use_gain else ""),
        flush=True,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)

    def lr_at(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(663)
    student = student.to(device)
    student.train()
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_score = -1.0
    best_state = None
    skipped = 0

    for i in range(steps):
        if use_gain:
            cond, clean, gain = _sample_cond_clean(
                tensors, crop, batch, g, wts, pair_gains)
            # Teacher sees stream-only cond; student gets gain channel appended.
            stream = cond.to(device)
            clean = clean.to(device)
            cond, _ = _prep_student_cond(stream, gain.to(device), gain_channel=True)
        else:
            cond, clean = _sample_cond_clean(tensors, crop, batch, g, wts)
            cond, clean = cond.to(device), clean.to(device)
            stream = cond
        with torch.no_grad():
            target = euler_sample(teacher, stream, steps=teacher_steps)
        opt.zero_grad(set_to_none=True)
        pred = student(cond)
        loss = sample_loss(pred, target)
        if gt_weight > 0:
            loss = loss + gt_weight * F.l1_loss(_lowpass(pred), _lowpass(clean))
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        opt.step()
        sched.step()
        if i % 50 == 0 or i == steps - 1:
            print(f"  step {i+1}/{steps}  loss={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/max(i,1):.2f}s/it", flush=True)
        if panel_every > 0 and ((i + 1) % panel_every == 0 or i == steps - 1):
            panel_ev = next((e for e in evals if e.get("gain", 0) >= 256), evals[0])
            pout, gr = _save_eval_panel(
                student, panel_ev, device, panel_dir, i + 1,
                gain_channel=gain_channel)
            probe, probe_gr = _quick_probe(
                student, evals[:6], device, gain_channel=gain_channel)
            print(f"  probe mean PSNR={probe:.2f} dB  grad_ratio={probe_gr:.3f}",
                  flush=True)
            # grad_ratio → 1.0 scores 20; blur or residual noise both penalised
            # Prefer sharpness (grad_ratio≈1) over PSNR — PSNR alone picks mush
            score = (0.25 * pout + 0.25 * probe
                     + 0.5 * (20.0 * max(0.0, 1.0 - abs(probe_gr - 1.0))))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in student.state_dict().items()}
                if best_path is not None:
                    torch.save({
                        "state_dict": best_state,
                        "step": i + 1,
                        "score": best_score,
                        "panel_psnr": pout,
                        "probe_psnr": probe,
                        "probe_grad_ratio": probe_gr,
                        "gain_channel": bool(gain_channel),
                    }, best_path)
                print(f"  ★ best@{i+1}: panel={pout:.2f} probe={probe:.2f} "
                      f"grad_r={probe_gr:.3f} score={best_score:.2f}", flush=True)
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"Restored best student (score={best_score:.2f})", flush=True)
    if skipped:
        print(f"Skipped {skipped} non-finite steps", flush=True)
    student.eval()
    return student


@torch.no_grad()
def _quick_probe(model, evals, device, *, gain_channel: bool = False) -> tuple[float, float]:
    model.eval()
    vals, grs = [], []
    for ev in evals:
        _, noisy = ev["noisy"][0]
        gt = ev["gt"]
        cond = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
        out = to_image(model(cond).cpu())
        vals.append(psnr(_rgb(out), _rgb(gt)))
        grs.append(grad_ratio(
            _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0)),
            _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)),
        ))
    model.train()
    return (float(np.mean(vals)) if vals else 0.0,
            float(np.mean(grs)) if grs else 0.0)


@torch.no_grad()
def _save_eval_panel(
    model, ev, device, panel_dir, step, *, gain_channel: bool = False,
) -> tuple[float, float]:
    was = model.training
    model.eval()
    idx, noisy = ev["noisy"][0]
    gt = ev["gt"]
    cond = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
    out = to_image(model(cond).cpu())
    nr, gr, or_ = _rgb(noisy[..., :4]), _rgb(gt), _rgb(out)
    pin, pout = psnr(nr, gr), psnr(or_, gr)
    g_ratio = grad_ratio(
        _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0)),
        _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)),
    )
    strip = np.concatenate([nr, or_, gr], axis=1)
    img = (np.clip(strip, 0, 1) * 255 + 0.5).astype(np.uint8)
    path = panel_dir / f"step_{step:05d}.png"
    Image.fromarray(img).save(path)
    latest = panel_dir / "latest.png"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        shutil.copy2(path, latest)
    # Pred|GT zoom crops for visual sharpness checks (not score-only).
    try:
        crop_dir = panel_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        h, w = or_.shape[:2]
        rois = {
            "siemens": (int(h * 0.28), int(w * 0.18), int(h * 0.22), int(w * 0.22)),
            "paper": (int(h * 0.72), int(w * 0.35), int(h * 0.18), int(w * 0.28)),
            "yarn": (int(h * 0.42), int(w * 0.48), int(h * 0.18), int(w * 0.18)),
        }
        for name, (y0, x0, ch, cw) in rois.items():
            y0 = max(0, min(y0, h - ch))
            x0 = max(0, min(x0, w - cw))
            pair = np.concatenate(
                [or_[y0:y0 + ch, x0:x0 + cw], gr[y0:y0 + ch, x0:x0 + cw]],
                axis=1,
            )
            pair_u8 = (np.clip(pair, 0, 1) * 255 + 0.5).astype(np.uint8)
            cpath = crop_dir / f"step_{step:05d}_{name}_pred_gt.png"
            Image.fromarray(pair_u8).save(cpath)
            latest_c = crop_dir / f"latest_{name}_pred_gt.png"
            try:
                if latest_c.exists() or latest_c.is_symlink():
                    latest_c.unlink()
                latest_c.symlink_to(cpath.name)
            except OSError:
                shutil.copy2(cpath, latest_c)
    except Exception as exc:  # noqa: BLE001 — crops are best-effort
        print(f"  WARN: visual crops failed ({exc})", flush=True)
    print(f"  panel step {step}: {ev['scene']}/ag{ev['gain']} "
          f"frame {idx}  {pin:.1f}→{pout:.1f} dB  grad_r={g_ratio:.3f}  "
          f"(1-step student)", flush=True)
    if was:
        model.train()
    return float(pout), float(g_ratio)


@torch.no_grad()
def evaluate(model, evals, device, *, gain_channel: bool = False) -> list[dict]:
    rows = []
    model.eval()
    for ev in evals:
        gt = ev["gt"]
        gr = _rgb(gt)
        gt_t = _rgb_t(torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0))
        for idx, noisy in ev["noisy"]:
            cond = _cond_for_eval(noisy, ev, device, gain_channel=gain_channel)
            out = to_image(model(cond).cpu())
            nr, or_ = _rgb(noisy[..., :4]), _rgb(out)
            out_t = _rgb_t(torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0))
            rows.append({
                "scene": ev["scene"], "gain": ev["gain"], "frame": idx,
                "psnr_in": round(psnr(nr, gr), 2),
                "psnr_out": round(psnr(or_, gr), 2),
                "ssim_out": round(ssim(or_, gr), 4),
                "grad_ratio": round(grad_ratio(out_t, gt_t), 4),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", type=Path, default=ROOT / "outputs/cfm_teacher.pt")
    ap.add_argument(
        "--pack-dir", type=Path, default=None,
        help="cloud_pack/data (npz pairs) — skips local DNG bursts",
    )
    ap.add_argument("--bursts", type=Path,
                    default=ROOT / "datasets/imx662_project/bursts")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    ap.add_argument("--gains", default=",".join(str(g) for g in DEFAULT_GAINS))
    ap.add_argument("--gt-frames", type=int, default=512)
    ap.add_argument(
        "--gt-mode", choices=("mean", "alpha_trim"), default="mean",
        help="GT for eval panels / optional gt_weight anchor",
    )
    ap.add_argument(
        "--gt-cache", type=Path, default=ROOT / "datasets/imx662_project/gt_alpha16",
    )
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--holdout-start", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--channels", type=int, default=64,
                    help="Pi-friendly width (student)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--temporal", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument(
        "--method", choices=("consistency", "regression_match"),
        default="consistency",
        help="consistency = CFM-CD 1-step (default); "
             "regression_match = old L1-to-teacher-samples ablation",
    )
    ap.add_argument(
        "--student-arch", choices=("naf", "lite"), default="naf",
        help="naf = ConsistencyStudent/_NAFBlock (default, SCA+GAP); "
             "lite = LiteDenoiseNet-style 3x3/ReLU/stride/nearest "
             "(Hailo-friendly; see docs/hailo_student_op_checklist.md)",
    )
    ap.add_argument("--teacher-steps", type=int, default=16,
                    help="Euler steps for regression_match targets / "
                         "consistency integrate budget")
    ap.add_argument("--integrate-steps", type=int, default=4,
                    help="Teacher ODE steps per CFM-CD target (consistency)")
    ap.add_argument("--cd-weight", type=float, default=0.5,
                    help="Classic EMA consistency-distillation weight "
                         "(0 disables)")
    ap.add_argument("--cd-intervals", type=int, default=16,
                    help="Discrete time grid size for classic CD")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-heun", action="store_true",
                    help="Use plain Euler for teacher ODE (default Heun)")
    ap.add_argument("--gt-weight", type=float, default=0.0,
                    help="low-pass L1 to GT (colour lock); 0 = pure CFM-CD "
                         "(default 0 — matches cfm_l1; low-pass can smear texture)")
    ap.add_argument(
        "--gt-grad-weight", type=float, default=0.0,
        help="spatial-∇ L1 of deploy boundary vs GT (pixelwise; can blur)",
    )
    ap.add_argument(
        "--gt-grad-energy-weight", type=float, default=0.0,
        help="|mean|∇|pred|/mean|∇|GT − 1| — scale-only push to grad_ratio≈1",
    )
    ap.add_argument(
        "--gt-hf-weight", type=float, default=0.0,
        help="high-pass L1 of deploy boundary vs GT (structure/texture, not grain)",
    )
    ap.add_argument(
        "--gt-lowfreq-weight", type=float, default=0.0,
        help="multi-scale low-freq L1 vs GT — kills flat-region blotch / colour "
             "drift the eye sees but L1 misses (try 0.5–2.0; regression path)",
    )
    ap.add_argument(
        "--sample-loss",
        choices=(
            "l1", "l2", "mse", "inv_luma", "l1_hf", "l1_grad",
            "l1_lpips", "l1_lpips05",
        ),
        default="l1",
        help="teacher-endpoint match: l1 (default), l1_lpips, l1_lpips05 "
             "(softer LPIPS), l1_hf, l1_grad, l2/mse, or inv_luma",
    )
    ap.add_argument(
        "--dump-dir", type=Path, default=None,
        help="Offline teacher ODE dumps (Stage A straighten); skips live teacher",
    )
    ap.add_argument("--inv-luma", action="store_true",
                    help="alias for --sample-loss inv_luma")
    ap.add_argument(
        "--restore-best", action="store_true",
        help="Ship the mid-train best (PSNR+grad_ratio≈1 score) instead of "
             "final weights — use with l1_grad / texture runs",
    )
    ap.add_argument(
        "--early-abort-soft", action="store_true",
        help="Dump distill: stop early if best peaks by step 100 and probe "
             "grad_ratio softens (<0.97) for 2 panels — skip likely rejects",
    )
    ap.add_argument(
        "--early-abort-after", type=int, default=200,
        help="Min steps before --early-abort-soft can fire (default 200)",
    )
    ap.add_argument(
        "--init-student", type=Path, default=None,
        help="Warm-start ConsistencyStudent from an existing cfm_student.pt",
    )
    ap.add_argument("--panel-every", type=int, default=400)
    ap.add_argument("--panel-dir", type=Path, default=ROOT / "outputs/cfm_student_panels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument(
        "--gain-channel", action="store_true",
        help="Student in_ch=4T+1: append log2(gain/128) constant map; "
             "peeled for FiLM at deploy (auto-on if teacher has gain_film)",
    )
    ap.add_argument(
        "--no-gain-channel", action="store_true",
        help="Force-disable gain channel even if teacher has gain_film",
    )
    args = ap.parse_args()

    if args.dump_dir is not None and not args.dump_dir.is_dir():
        print(f"Dump directory missing: {args.dump_dir}", file=sys.stderr)
        return 1
    # dump-dir + consistency is OK when a live teacher is present (synth_pairs
    # become (noisy, gt) crops; teacher still runs the ODE for CFM-CD).
    # dump-only / no teacher still forces regression_match (offline targets).
    if (
        args.dump_dir is not None
        and args.method != "regression_match"
        and not args.teacher.is_file()
    ):
        print("WARN: --dump-dir without --teacher forces regression_match",
              flush=True)
        args.method = "regression_match"

    if args.dump_dir is None and not args.teacher.is_file():
        print(f"Teacher checkpoint missing: {args.teacher}", file=sys.stderr)
        return 1
    if (
        args.method == "consistency"
        and args.dump_dir is not None
        and not args.teacher.is_file()
    ):
        print("consistency + --dump-dir requires a live --teacher",
              file=sys.stderr)
        return 1

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    gains = tuple(int(x) for x in args.gains.split(",") if x.strip())
    temporal = max(1, int(args.temporal))
    gt_frames = int(args.gt_frames)
    if args.gt_mode == "alpha_trim" and gt_frames == 512:
        gt_frames = 16
    stream_ch = 4 * temporal
    out_ch = 4
    dev = _device()
    heun = not args.no_heun
    recipe = (
        "consistency_flow_matching"
        if args.method == "consistency"
        else "cfm_distilled_1step_regression_match"
    )
    print(
        f"Device {dev}  recipe: {recipe}  "
        f"({'Consistency Flow Matching / CD → 1-step Pi' if args.method == 'consistency' else 'regression match ablation'})",
        flush=True,
    )

    teacher = None
    tblob: dict = {}
    teacher_gain = False
    tmeta: dict = {}
    # Load live teacher for consistency (and for regression_match when no dumps).
    need_teacher = (
        args.method == "consistency"
        or (args.dump_dir is None and args.method == "regression_match")
    )
    if need_teacher and args.teacher.is_file():
        teacher, tblob = _load_teacher(args.teacher, dev)
        tmeta = tblob.get("model", {})
        teacher_gain = bool(tmeta.get("gain_film", tblob.get("gain_film", False)))
        if int(tmeta.get("temporal", temporal)) != temporal:
            print(f"WARN: teacher temporal={tmeta.get('temporal')} vs "
                  f"--temporal {temporal}", flush=True)
        print(f"Teacher: {args.teacher}  "
              f"({tmeta.get('base_channels')}ch×{tmeta.get('block_depth')})"
              + ("  [gain FiLM]" if teacher_gain else ""), flush=True)
    if args.dump_dir is not None:
        print(f"Stage A: offline dumps at {args.dump_dir}"
              + ("  (+ live teacher CFM-CD)" if teacher is not None else ""),
              flush=True)

    # Prefer explicit flags; else match --init-student I/O; else teacher FiLM.
    gain_channel = False if args.no_gain_channel else bool(args.gain_channel)
    if not args.no_gain_channel and not args.gain_channel and args.init_student is not None:
        try:
            iblob = torch.load(args.init_student, map_location="cpu",
                               weights_only=False)
            istate = iblob.get("state_dict", iblob)
            if any(k.startswith("student.") for k in istate):
                istate = {k[len("student."):]: v for k, v in istate.items()
                          if k.startswith("student.")}
            hw = istate.get("head.weight")
            if hw is not None and getattr(hw, "ndim", 0) == 4:
                init_in = int(hw.shape[1])
                if init_in == stream_ch + 1:
                    gain_channel = True
                elif init_in == stream_ch:
                    gain_channel = False
                else:
                    print(f"WARN: init head in_ch={init_in} vs stream_ch={stream_ch}; "
                          f"falling back to teacher gain_film={teacher_gain}",
                          flush=True)
                    gain_channel = bool(teacher_gain)
            elif "gain_channel" in iblob:
                gain_channel = bool(iblob["gain_channel"])
            else:
                gain_channel = bool(teacher_gain)
        except Exception as exc:  # noqa: BLE001 — warm-start is best-effort
            print(f"WARN: could not probe --init-student for gain_channel ({exc}); "
                  f"using teacher gain_film={teacher_gain}", flush=True)
            gain_channel = bool(teacher_gain)
    elif not args.no_gain_channel and not args.gain_channel:
        gain_channel = bool(teacher_gain)
    in_ch = stream_ch + (1 if gain_channel else 0)

    dump_pair_gains: list[int] | None = None
    if args.pack_dir is not None:
        pairs, evals, meta = load_cloud_pack(args.pack_dir)
        pair_gains = meta.get("pair_gains")
    elif args.dump_dir is not None and args.method == "consistency":
        # Live CFM-CD on synth_pairs / dump shards (same loader as teacher).
        pairs, evals, pair_gains, meta = _load_dump_pairs(args.dump_dir)
        meta = {**meta, "pair_gains": pair_gains}
    elif args.dump_dir is not None and args.method == "regression_match":
        # Train on dump/synth shards; build real-burst holdouts for panels/eval.
        pairs, evals, meta = build_pairs(
            args.bursts, scenes, gains,
            gt_frames=gt_frames, stride=args.stride,
            holdout_start=args.holdout_start, temporal=temporal,
            gt_mode=args.gt_mode, gt_cache_root=args.gt_cache)
        pair_gains = meta.get("pair_gains")
        # Placeholder count — real dump size printed when tensors load.
        if not pairs and not evals:
            print("No eval pairs from bursts — panels will be limited",
                  flush=True)
    else:
        pairs, evals, meta = build_pairs(
            args.bursts, scenes, gains,
            gt_frames=gt_frames, stride=args.stride,
            holdout_start=args.holdout_start, temporal=temporal,
            gt_mode=args.gt_mode, gt_cache_root=args.gt_cache)
        pair_gains = meta.get("pair_gains")
    if not pairs and args.dump_dir is None:
        print("No training pairs — check bursts/, --pack-dir, or --dump-dir",
              file=sys.stderr)
        return 1
    print(f"Total train pairs: {len(pairs)}  in_ch={in_ch}"
          + (" (4T+gain)" if gain_channel else ""), flush=True)

    loss_name = "inv_luma" if args.inv_luma else args.sample_loss
    sample_loss, sample_loss_name = _make_sample_loss(name=loss_name)
    args.out.mkdir(parents=True, exist_ok=True)  # best_path saves mid-training

    if args.method == "consistency":
        if teacher is None:
            print("consistency method requires --teacher (not dump-only)",
                  file=sys.stderr)
            return 1
        student = build_cfm_student(
            args.student_arch,
            cond_ch=in_ch, out_ch=out_ch,
            base_channels=args.channels, block_depth=args.depth,
            gain_channel=gain_channel, consistency=True)
        if args.init_student is not None:
            if not args.init_student.is_file():
                print(f"Missing --init-student {args.init_student}",
                      file=sys.stderr)
                return 1
            iblob = torch.load(args.init_student, map_location=dev,
                               weights_only=False)
            istate = iblob["state_dict"]
            if any(k.startswith("student.") for k in istate):
                istate = {k[len("student."):]: v for k, v in istate.items()
                          if k.startswith("student.")}
            missing, unexpected = student.load_state_dict(istate, strict=False)
            print(f"Warm-start from {args.init_student} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})",
                  flush=True)
        n_params = sum(p.numel() for p in student.parameters())
        arch_name = type(student).__name__
        print(f"Student {arch_name} [{args.student_arch}] {in_ch}→{out_ch}  "
              f"{n_params:,} params  ({args.channels}ch × {args.depth})  "
              f"boundary x0=noisy,t=0"
              + ("  [gain channel FiLM]" if gain_channel else ""),
              flush=True)
        student = _train_consistency(
            student, teacher, pairs, args.steps,
            crop=args.crop, batch=args.batch, lr=args.lr, device=dev,
            panel_every=args.panel_every, panel_dir=args.panel_dir, evals=evals,
            integrate_steps=args.integrate_steps, cd_weight=args.cd_weight,
            cd_intervals=args.cd_intervals, ema_decay=args.ema_decay,
            gt_weight=args.gt_weight, heun=heun,
            sample_loss=sample_loss, sample_loss_name=sample_loss_name,
            best_path=args.out / "cfm_student_best.pt",
            pair_gains=pair_gains if (gain_channel or teacher_gain) else None,
            restore_best=bool(args.restore_best),
            gt_grad_weight=float(args.gt_grad_weight),
            gt_grad_energy_weight=float(args.gt_grad_energy_weight),
            gt_hf_weight=float(args.gt_hf_weight))
        deploy = wrap_deploy(student).to(dev).eval()
        family = (
            "cfm_lite_consistency_1step" if args.student_arch == "lite"
            else "cfm_consistency_1step"
        )
    else:
        if args.student_arch == "lite":
            student = LiteCfmStudent(
                cond_ch=in_ch, out_ch=out_ch,
                base_channels=args.channels,
                gain_channel=gain_channel)
        else:
            student = RawDenoiser(
                base_channels=args.channels, block_depth=args.depth,
                in_ch=in_ch, out_ch=out_ch)
            # Deploy / eval helpers look for this flag (RawDenoiser has no FiLM).
            student.gain_channel = bool(gain_channel)  # type: ignore[attr-defined]
            student.stream_ch = stream_ch  # type: ignore[attr-defined]
        if args.init_student is not None:
            if not args.init_student.is_file():
                print(f"Missing --init-student {args.init_student}",
                      file=sys.stderr)
                return 1
            iblob = torch.load(args.init_student, map_location=dev,
                               weights_only=False)
            istate = iblob["state_dict"] if isinstance(iblob, dict) and "state_dict" in iblob else iblob
            if any(k.startswith("student.") for k in istate):
                istate = {k[len("student."):]: v for k, v in istate.items()
                          if k.startswith("student.")}
            # Drop CFM-only keys; expand head when adding gain channel.
            istate = _warm_start_expand_head(student, istate)
            missing, unexpected = student.load_state_dict(istate, strict=False)
            print(f"Warm-start RawDenoiser from {args.init_student} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})",
                  flush=True)
        n_params = sum(p.numel() for p in student.parameters())
        print(f"Student {type(student).__name__} [{args.student_arch}] "
              f"{in_ch}→{out_ch}  {n_params:,} params  "
              f"({args.channels}ch × {args.depth})"
              + ("  [gain channel]" if gain_channel else ""), flush=True)
        if args.dump_dir is not None:
            dump_tensors, dump_pair_gains = _load_dump_tensors(args.dump_dir)
            student = _train_regression_from_dumps(
                student, dump_tensors, args.steps,
                crop=args.crop, batch=args.batch, lr=args.lr, device=dev,
                panel_every=args.panel_every, panel_dir=args.panel_dir,
                evals=evals, sample_loss=sample_loss,
                sample_loss_name=sample_loss_name,
                best_path=args.out / "cfm_student_best.pt",
                early_abort_soft=bool(args.early_abort_soft),
                early_abort_after=int(args.early_abort_after),
                pair_gains=dump_pair_gains if gain_channel else None,
                gain_channel=gain_channel,
                gt_hf_weight=float(args.gt_hf_weight),
                gt_grad_energy_weight=float(args.gt_grad_energy_weight),
                gt_lowfreq_weight=float(args.gt_lowfreq_weight))
        else:
            if teacher is None:
                print("regression_match without --dump-dir needs --teacher",
                      file=sys.stderr)
                return 1
            student = _train_regression_match(
                student, teacher, pairs, args.steps,
                crop=args.crop, batch=args.batch, lr=args.lr, device=dev,
                panel_every=args.panel_every, panel_dir=args.panel_dir,
                evals=evals,
                teacher_steps=args.teacher_steps, gt_weight=args.gt_weight,
                sample_loss=sample_loss, sample_loss_name=sample_loss_name,
                best_path=args.out / "cfm_student_best.pt",
                pair_gains=pair_gains if gain_channel else None,
                gain_channel=gain_channel)
        deploy = student
        family = (
            "cfm_lite_student" if args.student_arch == "lite"
            else "raw_denoiser_stream"
        )

    rows = evaluate(deploy, evals, dev, gain_channel=gain_channel)
    mean_in = float(np.mean([r["psnr_in"] for r in rows]))
    mean_out = float(np.mean([r["psnr_out"] for r in rows]))
    mean_gr = float(np.mean([r["grad_ratio"] for r in rows]))
    print(f"Held-out mean PSNR {mean_in:.2f} → {mean_out:.2f} dB  "
          f"grad_ratio={mean_gr:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "cfm_student.pt"
    state = (
        student.state_dict()
        if args.method == "consistency"
        else deploy.state_dict()
    )
    torch.save({
        "state_dict": state,
        "model": {
            "family": family,
            "student_arch": args.student_arch,
            "base_channels": args.channels,
            "block_depth": args.depth,
            "in_ch": in_ch,
            "out_ch": out_ch,
            "cond_ch": in_ch,
            "temporal": temporal,
            "gain_channel": bool(gain_channel),
            "gain_encoding": "log2(gain/128)" if gain_channel else None,
            "deploy_boundary": "x0_noisy_t0" if args.method == "consistency" else None,
        },
        "recipe": recipe,
        "method": args.method,
        "sample_loss": sample_loss_name,
        "teacher": str(args.teacher),
        "teacher_steps": args.teacher_steps,
        "integrate_steps": args.integrate_steps,
        "cd_weight": args.cd_weight,
        "gt_weight": args.gt_weight,
        "gt_frames": gt_frames,
        "gt_mode": args.gt_mode,
        "dump_dir": str(args.dump_dir) if args.dump_dir else None,
        "gains": list(gains),
        "scenes": list(scenes),
        "temporal": temporal,
        "gain_channel": bool(gain_channel),
        "psnr_in": mean_in,
        "psnr_out": mean_out,
        "grad_ratio": mean_gr,
        "eval": rows,
    }, ckpt)
    # Also write under the deploy name so pi_stream_denoise defaults Just Work
    deploy_ckpt = args.out / "stream_to_gt_cfm.pt"
    shutil.copy2(ckpt, deploy_ckpt)
    meta_json = {k: v for k, v in meta.items() if k != "pair_gains"}
    summary = {
        "recipe": recipe,
        "method": args.method,
        "sample_loss": sample_loss_name,
        "gain_channel": bool(gain_channel),
        "psnr_in": mean_in, "psnr_out": mean_out,
        "grad_ratio": mean_gr,
        "params": n_params, "pairs": len(pairs),
        "channels": args.channels, "depth": args.depth,
        "steps": args.steps,
        "integrate_steps": args.integrate_steps,
        "teacher_steps": args.teacher_steps,
        "cd_weight": args.cd_weight,
        "temporal": temporal, "in_ch": in_ch,
        "teacher": str(args.teacher),
        "gains": list(gains), "eval": rows, **meta_json,
    }
    (args.out / "cfm_student_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Checkpoint: {ckpt}", flush=True)
    print(f"Deploy copy: {deploy_ckpt}", flush=True)

    if evals:
        _save_eval_panel(
            deploy, evals[0], dev, args.panel_dir, args.steps,
            gain_channel=gain_channel)
        shutil.copy2(args.panel_dir / f"step_{args.steps:05d}.png",
                     args.out / "cfm_student_panel.png")

    if not args.no_onnx:
        onnx_path = args.out / "cfm_student.onnx"
        _export_onnx(deploy, in_ch, onnx_path)
        shutil.copy2(onnx_path, args.out / "stream_to_gt_cfm.onnx")
        print(f"ONNX: {onnx_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
