"""Rectified flow (image-to-image CFM) + consistency distillation for packed-RAW.

Regression (L1 / Charbonnier) learns the conditional mean → oil-paint blur at
ag512. We instead learn a *velocity field* along the rectified-flow path from
the live noisy frame to the sharp clean sample, conditioned on the full noisy
temporal stack (InDI-style image-to-image flow), so textures stay crisp:

    x₀ = noisy frame (cond[:, :4], + tiny gaussian jitter for stochasticity)
    x₁ = clean GT
    x_t = (1-t)·x₀ + t·x₁,   target v = x₁ - x₀

Because x₀ is already a good draft of x₁, the ODE is short — 4-8 Euler steps.

Optional analogue-gain FiLM: g = log2(gain/128) → MLP → (γ, β) fused with
time FiLM so h' = (1+γ)⊙h + β (identity at init: γ=β=0). Teacher takes a
(B,) gain tensor; the Pi student prefers a constant gain channel on cond
(in_ch = 4T+1) peeled into the same FiLM — single-tensor ONNX.

Teacher: multi-step Euler/Heun ODE on the velocity network (noisy→clean path).

Student (consistency distillation for edge 1-step inference):
    A time-conditioned ConsistencyStudent maps (x_t, t, cond) → clean x̂₁.
    Distillation matches the teacher's integrated displacement t→1 (CFM-CD),
    optionally with classic EMA consistency between neighbouring times.
    At inference the Pi evaluates the boundary only:
        x₀ = noisy frame, t = 0  →  one forward from the live frame.
    ``BoundaryConsistencyWrapper`` exposes RawDenoiser I/O (cond 4T[+1] → packed 4)
    for ONNX / ``pi_stream_denoise.py``.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import _NAFBlock

# Analog-gain FiLM: g = log2(gain / GAIN_REF) → 128→0, 256→1, 512→2.
GAIN_REF = 128.0


def encode_analog_gain(gain: torch.Tensor) -> torch.Tensor:
    """Map analogue gain → scalar condition. ``gain`` shape (B,) → (B,)."""
    return torch.log2(gain.float().clamp(min=1.0) / GAIN_REF)


def append_gain_channel(
    cond: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    """Broadcast ``encode_analog_gain(gain)`` as a constant HxW channel on ``cond``.

    Deploy / ONNX path: student ``in_ch = 4T+1``; Pi fills this channel from
    ``--gain`` (no second ONNX input).
    """
    b, _, h, w = cond.shape
    g = encode_analog_gain(gain).to(dtype=cond.dtype, device=cond.device)
    ch = g.view(b, 1, 1, 1).expand(b, 1, h, w)
    return torch.cat([cond, ch], dim=1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous t in [0, 1]. ``t`` shape (B,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def sample_ot_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal-transport / rectified-flow path: x_t = (1-t) x0 + t x1, v = x1-x0."""
    # t: (B,) → (B,1,1,1)
    tw = t.view(-1, 1, 1, 1).to(dtype=x0.dtype)
    x_t = (1.0 - tw) * x0 + tw * x1
    v = x1 - x0
    return x_t, v


def flow_x0(
    cond: torch.Tensor,
    out_ch: int = 4,
    jitter: float = 0.02,
) -> torch.Tensor:
    """Flow start point x₀ = live noisy frame ``cond[:, :out_ch]`` (+ jitter).

    ``jitter`` adds tiny gaussian noise (σ relative to the [0,1] data scale)
    for path stochasticity; 0 gives the deterministic Pi boundary.
    """
    x0 = cond[:, :out_ch]
    if jitter > 0:
        x0 = x0 + jitter * torch.randn_like(x0)
    return x0


class FlowVelocityNet(nn.Module):
    """NAF velocity field v_θ(x_t, t, cond) → velocity in packed-RAW space (4ch).

    ``cond`` is the noisy stream stack (4T channels). ``x_t`` is the flow state
    (4ch). Time is injected via FiLM after the head.

    EDM preconditioning (``edm_precond=True``, Karras et al. 2022 §5)
    ------------------------------------------------------------------
    Our rectified flow x_t = (1−t)·x₀ + t·x₁ with x₀ = x₁ + n (noisy frame =
    clean + frame noise + jitter) means x_t = x₁ + (1−t)·n, i.e. the exact EDM
    setting "clean signal + gaussian-ish noise" with an effective noise level

        σ(t) = (1−t)·σ_flow,      σ_flow = std(x₀ − x₁)  (measured from data,
                                            jitter added in quadrature)

    and signal scale σ_data = std(x₁) (measured from clean GT; the packed RAW
    is very dark so this is ~0.02-0.08, NOT the 0.5 of natural images).
    The network core F is wrapped exactly like EDM's denoiser
    D(x;σ) = c_skip·x + c_out·F(c_in·x, c_noise):

        c_in(σ)    = 1/√(σ² + σ_data²)              (unit-variance input)
        c_skip(σ)  = σ_data²/(σ² + σ_data²)
        c_out(σ)   = σ·σ_data/√(σ² + σ_data²)       (unit-variance target)
        c_noise(σ) = ln(σ)/4                        (log-warped noise label)
        cond scale = 1/√(σ_flow² + σ_data²)         (constant; cond carries
                                                     full frame noise)

    The external velocity API is unchanged: since v = x₁ − x₀ = (x₁ − x_t)/(1−t),
    the returned velocity is v = (D(x_t;σ(t)) − x_t)/(1−t), computed in the
    singularity-free form

        v = −(1−t)·σ_flow²/(σ² + σ_data²) · x_t
            + σ_flow·σ_data/√(σ² + σ_data²) · F(...)

    which is smooth at t=1 (both factors stay finite; v(t=1) → σ_flow·F).
    With zero-init tail the initial model is the EDM-optimal linear shrinkage
    D = c_skip·x_t rather than the identity.
    """

    def __init__(
        self,
        cond_ch: int,
        out_ch: int = 4,
        base_channels: int = 128,
        block_depth: int = 8,
        time_dim: int = 128,
        edm_precond: bool = False,
        sigma_data: float = 0.05,
        sigma_flow: float = 0.06,
        gain_film: bool = False,
    ):
        super().__init__()
        self.cond_ch = cond_ch
        self.out_ch = out_ch
        self.base_channels = base_channels
        self.block_depth = block_depth
        self.time_dim = time_dim
        self.edm_precond = bool(edm_precond)
        self.gain_film = bool(gain_film)
        if self.edm_precond:
            # Buffers only in EDM mode so legacy checkpoints still load strict.
            self.register_buffer("sigma_data", torch.tensor(float(sigma_data)))
            self.register_buffer("sigma_flow", torch.tensor(float(sigma_flow)))
        c = base_channels
        self.head = nn.Conv2d(out_ch + cond_ch, c, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, c * 2),
            nn.SiLU(),
            nn.Linear(c * 2, c * 2),
        )
        if self.gain_film:
            # Maps g=log2(gain/128) → (γ,β) fused with time FiLM.
            # Zero-init last layer ⇒ γ=0,β=0 ⇒ h'=h at load (identity).
            self.gain_mlp = nn.Sequential(
                nn.Linear(1, c * 2),
                nn.SiLU(),
                nn.Linear(c * 2, c * 2),
            )
            nn.init.zeros_(self.gain_mlp[-1].weight)
            nn.init.zeros_(self.gain_mlp[-1].bias)
        else:
            self.gain_mlp = None
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(block_depth)])
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)
        # Zero-init tail so early training ≈ identity velocity ≈ 0
        # (EDM mode: ≈ optimal linear shrinkage, see class docstring)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def _film_body(
        self,
        x_in: torch.Tensor,
        t_label: torch.Tensor,
        gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.head(x_in)
        emb = self.time_mlp(timestep_embedding(t_label, self.time_dim))
        if self.gain_mlp is not None:
            if gain is None:
                g = torch.zeros(h.shape[0], 1, device=h.device, dtype=emb.dtype)
            else:
                g = encode_analog_gain(gain).to(device=h.device, dtype=emb.dtype)
                g = g.view(-1, 1)
            emb = emb + self.gain_mlp(g)
        scale, shift = emb.chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        return self.tail(self.body(h))

    def edm_scales(
        self, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(c_in, c_noise, a, b) at flow time ``t`` — v = a·x_t + b·F.

        a = (c_skip−1)/(1−t), b = c_out/(1−t) in closed form (both finite ∀t).
        """
        sd = self.sigma_data
        s0 = self.sigma_flow
        t = t.view(-1).to(sd.dtype)
        sigma_t = (1.0 - t) * s0
        s2 = sigma_t * sigma_t + sd * sd
        c_in = 1.0 / torch.sqrt(s2)
        c_noise = 0.25 * torch.log(sigma_t.clamp(min=1e-6))
        a = -(1.0 - t) * s0 * s0 / s2
        b = s0 * sd / torch.sqrt(s2)
        return c_in, c_noise, a, b

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.edm_precond:
            return self._film_body(torch.cat([x_t, cond], dim=1), t, gain)
        c_in, c_noise, a, b = self.edm_scales(t)
        c_cond = 1.0 / torch.sqrt(
            self.sigma_flow * self.sigma_flow + self.sigma_data * self.sigma_data)
        x_in = torch.cat(
            [x_t * c_in.view(-1, 1, 1, 1), cond * c_cond], dim=1)
        f_raw = self._film_body(x_in, c_noise, gain)
        return a.view(-1, 1, 1, 1) * x_t + b.view(-1, 1, 1, 1) * f_raw


class ConsistencyStudent(nn.Module):
    """Predicts clean x̂₁ from (x_t, t, cond) — Consistency Flow Matching student.

    Residual is anchored on the live noisy frame ``cond[:, :out_ch]`` (same
    contract as RawDenoiser). At the Pi boundary (x_t = noisy frame, t = 0)
    this is one forward that recovers a sharp sample conditioned on the stream.

    When ``gain_channel=True``, ``cond`` is ``[stream 4T | gain_norm]``
    (``in_ch = 4T+1``). The gain map is peeled for FiLM (same ``log2(gain/128)``
    encoding as the teacher); the stream stack alone feeds the head / residual
    base. ONNX stays a single tensor input — Pi fills the constant channel.
    """

    def __init__(
        self,
        cond_ch: int,
        out_ch: int = 4,
        base_channels: int = 64,
        block_depth: int = 6,
        time_dim: int = 64,
        gain_channel: bool = False,
    ):
        super().__init__()
        self.cond_ch = cond_ch
        self.out_ch = out_ch
        self.in_ch = cond_ch  # Boundary wrapper / ONNX I/O alias
        self.base_channels = base_channels
        self.block_depth = block_depth
        self.time_dim = time_dim
        self.gain_channel = bool(gain_channel)
        self.stream_ch = cond_ch - (1 if self.gain_channel else 0)
        c = base_channels
        self.head = nn.Conv2d(out_ch + self.stream_ch, c, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, c * 2),
            nn.SiLU(),
            nn.Linear(c * 2, c * 2),
        )
        if self.gain_channel:
            self.gain_mlp = nn.Sequential(
                nn.Linear(1, c * 2),
                nn.SiLU(),
                nn.Linear(c * 2, c * 2),
            )
            nn.init.zeros_(self.gain_mlp[-1].weight)
            nn.init.zeros_(self.gain_mlp[-1].bias)
        else:
            self.gain_mlp = None
        self.body = nn.Sequential(*[_NAFBlock(c) for _ in range(block_depth)])
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def _split_cond(
        self, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(stream, gain_scalar_or_None). Gain already encoded as log2(gain/128)."""
        if not self.gain_channel:
            return cond, None
        stream = cond[:, : self.stream_ch]
        # Constant map → per-sample scalar (mean over spatial for safety).
        g = cond[:, -1:].mean(dim=(2, 3))  # (B, 1)
        return stream, g

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        stream, g = self._split_cond(cond)
        h = self.head(torch.cat([x_t, stream], dim=1))
        emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if self.gain_mlp is not None:
            if g is None:
                g = torch.zeros(h.shape[0], 1, device=h.device, dtype=emb.dtype)
            emb = emb + self.gain_mlp(g.to(dtype=emb.dtype))
        scale, shift = emb.chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        residual = self.tail(self.body(h))
        base = stream[:, : self.out_ch]
        return torch.clamp(base + residual, 0.0, 1.0)


class BoundaryConsistencyWrapper(nn.Module):
    """Deploy wrapper: RawDenoiser I/O — ``forward(cond) → clean``.

    Evaluates the consistency student at the fixed Pi boundary:
    x₀ = live noisy frame ``cond[:, :out_ch]`` (deterministic, no jitter),
    t = 0. No time tensor is exposed to ONNX.
    """

    def __init__(self, student: ConsistencyStudent):
        super().__init__()
        self.student = student
        self.in_ch = student.cond_ch
        self.out_ch = student.out_ch

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        x0 = cond[:, : self.out_ch]
        t = torch.zeros(b, device=cond.device, dtype=cond.dtype)
        return self.student(x0, t, cond)


def cfm_loss(
    model: FlowVelocityNet,
    clean: torch.Tensor,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    x0_jitter: float = 0.02,
    p_mean: float = 0.0,
    p_std: float = 1.0,
) -> torch.Tensor:
    """Rectified-flow matching loss (noisy frame → clean | cond [, gain]).

    x₀ = cond[:, :4] (+ ``x0_jitter`` gaussian), x₁ = clean;
    x_t = (1-t)·x₀ + t·x₁, target v = x₁ - x₀.

    ``clean`` / network output: B×4×H×W. ``cond``: B×C×H×W noisy stream stack.
    ``gain``: optional (B,) analogue gains for FiLM (neutral zeros if omitted).

    If ``model.edm_precond`` (EDM training, Karras et al. 2022 §5):
      - t ~ logit-normal(p_mean, p_std) instead of Uniform(0,1) — the
        flow-matching analog of EDM's log-normal σ sampling, concentrating
        capacity on informative middle noise levels;
      - per-sample weight w(t) = 1/c_out'(t)² (c_out' = c_out/(1−t), the
        velocity-space output scale) so every t contributes a unit-variance
        gradient on the raw network F — the loss value IS the F-space MSE,
        O(1) at init regardless of how dark the data is.
    """
    b = clean.shape[0]
    device = clean.device
    x0 = flow_x0(cond, clean.shape[1], x0_jitter)
    if getattr(model, "edm_precond", False):
        t = torch.sigmoid(p_mean + p_std * torch.randn(b, device=device))
        x_t, v_target = sample_ot_path(x0, clean, t)
        v_pred = model(x_t, t, cond, gain)
        _, _, _, b_out = model.edm_scales(t)
        w = 1.0 / (b_out * b_out).clamp(min=1e-12)
        per = (v_pred - v_target).pow(2).mean(dim=(1, 2, 3))
        return (w.to(per.dtype) * per).mean()
    t = torch.rand(b, device=device)
    x_t, v_target = sample_ot_path(x0, clean, t)
    v_pred = model(x_t, t, cond, gain)
    return F.mse_loss(v_pred, v_target)


def rho_alphas(
    steps: int, rho: float = 7.0, sigma_min_frac: float = 0.05,
) -> list[float]:
    """Karras-style step fractions α_i ∈ [0,1], dense near the clean end.

    EDM (eq. 5): σ_i^{1/ρ} linear from σ_max^{1/ρ} to σ_min^{1/ρ} over N
    nodes, with σ_N = 0 exactly. In flow time σ(t) ∝ (1−t), so with
    s_i = σ_i/σ_max: α_i = 1 − s_i, s_min = ``sigma_min_frac``.
    ρ ≤ 1 recovers the legacy uniform grid.
    """
    n = max(1, int(steps))
    r = float(rho)
    if r <= 1.0:
        return [i / n for i in range(n + 1)]
    smin = max(1e-6, float(sigma_min_frac)) ** (1.0 / r)
    alphas = [
        1.0 - (1.0 + (i / max(1, n - 1)) * (smin - 1.0)) ** r
        for i in range(n)
    ]
    alphas.append(1.0)
    return alphas


@torch.no_grad()
def ode_integrate(
    model: FlowVelocityNet,
    x: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    steps: int = 4,
    heun: bool = False,
    rho: float = 1.0,
) -> torch.Tensor:
    """Integrate teacher ODE dx/dt = v_θ from ``t_start`` → ``t_end`` (per-batch).

    ``t_start`` / ``t_end`` are shape (B,). ``rho`` > 1 warps the step grid
    EDM-style (dense near ``t_end``, the clean end); 1.0 = uniform. ``heun``
    adds a 2nd-order correction on every step except the last (which lands at
    σ≈0, matching EDM's convention).
    """
    model.eval()
    n = max(1, int(steps))
    b = x.shape[0]
    dtype = x.dtype
    alphas = rho_alphas(n, rho)
    span = t_end - t_start
    for i in range(n):
        t0 = t_start + span * alphas[i]
        t1 = t_start + span * alphas[i + 1]
        dt = (t1 - t0).view(b, 1, 1, 1).to(dtype=dtype)
        v = model(x, t0.to(dtype=dtype), cond, gain)
        if heun and i < n - 1:
            x_euler = x + dt * v
            v2 = model(x_euler, t1.to(dtype=dtype), cond, gain)
            x = x + 0.5 * dt * (v + v2)
        else:
            x = x + dt * v
    return x


@torch.no_grad()
def teacher_integrate_to_one(
    model: FlowVelocityNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    steps: int = 4,
    heun: bool = False,
    rho: float = 1.0,
) -> torch.Tensor:
    """Integrate teacher from current t to t=1 (clean endpoint), then clamp."""
    t_end = torch.ones_like(t)
    # Skip work when already at / past the endpoint
    out = ode_integrate(
        model, x_t, t, t_end, cond, gain=gain, steps=steps, heun=heun, rho=rho)
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def euler_sample(
    model: FlowVelocityNet,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    steps: int = 8,
    x0: torch.Tensor | None = None,
    x0_jitter: float = 0.02,
    heun: bool | None = None,
    rho: float | None = None,
) -> torch.Tensor:
    """Integrate ODE dx/dt = v_θ from t=0 (noisy frame) to t=1 (clean sample).

    Default x₀ = cond[:, :4] + ``x0_jitter`` gaussian (rectified flow starts at
    the live noisy frame, so few steps suffice).

    ``heun`` / ``rho`` default per-model: EDM-preconditioned models use the
    EDM sampler (Heun 2nd order, ρ=7 polynomial spacing toward the clean end);
    legacy models keep plain uniform Euler.
    """
    model.eval()
    is_edm = getattr(model, "edm_precond", False)
    if heun is None:
        heun = bool(is_edm)
    if rho is None:
        rho = 7.0 if is_edm else 1.0
    b = cond.shape[0]
    device = cond.device
    dtype = cond.dtype
    x = flow_x0(cond, model.out_ch, x0_jitter) if x0 is None else x0
    t0 = torch.zeros(b, device=device, dtype=dtype)
    t1 = torch.ones(b, device=device, dtype=dtype)
    return ode_integrate(
        model, x, t0, t1, cond, gain=gain, steps=steps, heun=heun, rho=rho,
    ).clamp(0.0, 1.0)


@torch.no_grad()
def one_step_sample(
    model: FlowVelocityNet,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single Euler step (t=0→1). Useful as a weak baseline / distill warm-start."""
    return euler_sample(model, cond, gain=gain, steps=1, x0=x0, heun=False)


def _teacher_cond_and_gain(
    student: ConsistencyStudent,
    cond: torch.Tensor,
    gain: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Teacher sees stream-only cond; gain is a separate FiLM scalar.

    Student may carry gain as the last channel (``gain_channel``); peel it so
    teacher ``cond_ch`` stays 4T.
    """
    if getattr(student, "gain_channel", False) and cond.shape[1] > student.stream_ch:
        stream = cond[:, : student.stream_ch]
        if gain is None:
            # Channel already holds encode_analog_gain; recover raw-ish by
            # passing encoded value and letting teacher encode again would
            # double-encode — so pass encoded via a synthetic raw gain that
            # encode(g)=channel. Prefer explicit ``gain`` from the dataloader.
            enc = cond[:, -1:].mean(dim=(2, 3)).view(-1)
            # Inverse of encode: gain = 128 * 2^enc
            gain = GAIN_REF * torch.pow(
                torch.tensor(2.0, device=cond.device, dtype=enc.dtype), enc)
        return stream, gain
    return cond, gain


def consistency_flow_matching_loss(
    student: ConsistencyStudent,
    teacher: FlowVelocityNet,
    clean: torch.Tensor,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    integrate_steps: int = 4,
    heun: bool = True,
    rho: float | None = None,
    t_eps: float = 1e-3,
    fixed_noise: bool = True,
    x0_jitter: float = 0.02,
    sample_loss=None,
) -> torch.Tensor:
    """Consistency Flow Matching distill: student jumps to x₁ matching teacher.

    Path: rectified flow between x₀ = noisy frame and clean | cond.
      - ``fixed_noise=True`` (default): x₀ = cond[:, :4] exactly — matches the
        deterministic Pi boundary.
      - else: x₀ = cond[:, :4] + ``x0_jitter`` gaussian for broader coverage.

    L = || f_θ(x_t, t, cond) − sg(teacher_integrate(x_t, t→1, cond)) ||²
    ``sample_loss(pred, target)`` overrides the default MSE endpoint match
    (e.g. charbonnier+swtrel to shape high-frequency texture).
    """
    b = clean.shape[0]
    device = clean.device
    if rho is None:
        rho = 7.0 if getattr(teacher, "edm_precond", False) else 1.0
    cond_t, gain_t = _teacher_cond_and_gain(student, cond, gain)
    x0 = flow_x0(cond_t, clean.shape[1], 0.0 if fixed_noise else x0_jitter)
    t = torch.rand(b, device=device) * (1.0 - t_eps)
    x_t, _ = sample_ot_path(x0, clean, t)
    with torch.no_grad():
        target = teacher_integrate_to_one(
            teacher, x_t, t, cond_t, gain=gain_t,
            steps=integrate_steps, heun=heun, rho=rho)
    pred = student(x_t, t, cond)
    if sample_loss is not None:
        return sample_loss(pred, target)
    return F.mse_loss(pred, target)


def consistency_distillation_loss(
    student: ConsistencyStudent,
    student_ema: ConsistencyStudent,
    teacher: FlowVelocityNet,
    clean: torch.Tensor,
    cond: torch.Tensor,
    *,
    gain: torch.Tensor | None = None,
    num_intervals: int = 16,
    ode_steps: int = 1,
    heun: bool = True,
    fixed_noise: bool = True,
    x0_jitter: float = 0.02,
) -> torch.Tensor:
    """Classic consistency distillation with EMA target network (FM time).

    Rectified-flow path: t=0 noisy frame → t=1 clean (``fixed_noise=True``
    uses x₀ = cond[:, :4] exactly; else adds ``x0_jitter`` gaussian). Sample
    t_early < t_late on a uniform grid; teacher advances x_early → x_late
    (stop-grad). Enforce
      f_θ(x_late, t_late) ≈ sg(f_θ⁻(x_early, t_early)).
    """
    b = clean.shape[0]
    device = clean.device
    dtype = clean.dtype
    n = max(2, int(num_intervals))
    # Indices: late ∈ {1..n}, early ∈ {0..late-1} so Δt > 0 toward clean
    late = torch.randint(1, n + 1, (b,), device=device)
    early = torch.floor(torch.rand(b, device=device) * late.float()).long()
    t_late = (late.float() / n).to(dtype=dtype)
    t_early = (early.float() / n).to(dtype=dtype)
    cond_t, gain_t = _teacher_cond_and_gain(student, cond, gain)
    x0 = flow_x0(cond_t, clean.shape[1], 0.0 if fixed_noise else x0_jitter)
    x_early, _ = sample_ot_path(x0, clean, t_early)
    with torch.no_grad():
        x_late = ode_integrate(
            teacher, x_early, t_early, t_late, cond_t, gain=gain_t,
            steps=max(1, int(ode_steps)), heun=heun)
        target = student_ema(x_early, t_early, cond)
    pred = student(x_late.detach(), t_late, cond)
    return F.mse_loss(pred, target.detach())


@torch.no_grad()
def update_ema(
    ema: nn.Module,
    model: nn.Module,
    decay: float = 0.999,
) -> None:
    """EMA: ema ← decay * ema + (1 - decay) * model."""
    d = float(decay)
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def make_ema(model: nn.Module) -> nn.Module:
    """Deep-copy ``model`` as a frozen EMA teacher-of-student."""
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def grad_energy(rgb: torch.Tensor) -> torch.Tensor:
    """Mean gradient magnitude — proxy for texture sharpness (higher = sharper)."""
    # rgb: B×C×H×W or 1×3×H×W
    dx = rgb[..., :, 1:] - rgb[..., :, :-1]
    dy = rgb[..., 1:, :] - rgb[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def grad_ratio(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred_grad / target_grad — ~1.0 means texture energy matches GT."""
    gp = float(grad_energy(pred.detach()).item())
    gt = float(grad_energy(target.detach()).item())
    if gt < 1e-8:
        return 0.0
    return gp / gt
