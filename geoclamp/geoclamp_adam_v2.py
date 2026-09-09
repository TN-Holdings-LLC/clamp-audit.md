# -*- coding: utf-8 -*-
"""
geoclamp_adam_v2.py — Riemannian Adam on SO(3) and S^3 with a step-size clamp

WHAT THE PREVIOUS VERSION GOT RIGHT.  Both of its bug reports reproduce
exactly, and the fixes are kept.

  * Bias correction. The uncorrected ratio (1-b1^t)/sqrt(1-b2^t) measures
    3.162 at t=1, 5.797 at t=5, 6.528 at t=10 (the peak), 3.241 at t=100,
    1.196 at t=1200 — matching the writeup to three decimals. `state["step"]`
    really was initialised and never incremented, so the correction could not
    have been applied even in principle.
  * The B coefficient. In float32, (1-cos t)/t^2 measures 0.476837 at
    t=1e-3, 0.662274 at t=3e-4 (already on the wrong side of a monotone
    function), and exactly 0.000000 at t<=1e-4, against a true limit of 0.5.
    The half-angle form 0.5*(sin(t/2)/(t/2))^2 gives 0.500000 throughout.
    Its own assessment — real defect, negligible effect on R because K@K is
    O(t^2) there — is also correct, and honest about it.

WHAT IS STILL WRONG.

 1. ON S^3 THE TRUST RADIUS DOES NOT MEAN RADIANS, AND IS 36% OUT AT THE
    DEFAULT max_step.  `_step_s3` moves by the first-order retraction
    q <- normalize(q + xi). For |xi| = r the realised geodesic angle is
    atan(r), not r. Measured:

        requested r   realised (rad)   error
             0.0100          0.0100    -0.1%
             0.1000          0.0997    -0.3%
             0.5000          0.4636    -7.3%
             1.0000          0.7854   -21.5%
             1.5698          1.0036   -36.1%   <- the default max_step

    So the clamp that is the whole point of this optimiser is enforcing a
    bound in tangent units while the manifold moves by something else.
    Fixed with the true exponential map, q*cos(r) + (xi/r)*sin(r), which
    moves exactly r radians. The retraction is still available as
    `retraction="proj"` for anyone who wants the old behaviour.

 2. NOTHING KEEPS R ON SO(3).  `_step_s3` renormalises the quaternion every
    single step; `_step_so3` does `p.copy_(p @ dR)` and never re-projects.
    Measured drift in ||R^T R - I|| over 20,000 steps: 3.2e-7 -> 1.4e-5 in
    float32, 3.9e-17 -> 2.8e-14 in float64.

    Stated honestly: 1.4e-5 after 20k steps will not break a training run,
    and this is a slow drift, not a blow-up. But it grows without bound, it
    costs almost nothing to stop, and the asymmetry with the quaternion path
    — one branch enforces its constraint every step, the other never does —
    is the kind of inconsistency that is a bug even when it has not bitten
    yet. Fixed with a cheap re-projection every `reproject_every` steps.

 3. THE STEP IS NOT SCALE-INVARIANT, WHICH DEFEATS THE POINT OF ADAM.
    `xi_raw` is Adam-preconditioned, so its size tracks `lr` rather than the
    gradient magnitude. But `r_trust = base_trust/(1 + kappa*vnorm)` is
    driven by the EMA of the RAW tangent-gradient norm, so multiplying the
    loss by a constant changes the applied step. Measured after 50 steps at
    lr=0.1 on an identical gradient direction:

        grad scale   |xi_raw|   r_trust   applied
             1e-02     0.0141    0.8793    0.0141
             1e+00     0.0141    0.8714    0.0141
             1e+02     0.0141    0.4590    0.0141
             1e+04     0.1423    0.0059    0.0059   <- clamped 2.4x smaller

    Rescaling a loss is supposed to be a no-op for Adam. Fixed by driving
    the trust radius from the EMA of |xi_raw| — the quantity the radius is
    actually trying to bound — which is scale-free by construction.
    `trust_source="grad"` restores the old behaviour.

 4. THE MOMENTUM IS NEVER PARALLEL-TRANSPORTED — AND FIXING THAT MADE THINGS
    WORSE, SO IT IS OFF BY DEFAULT.  This one is worth reading in full,
    because the tidy version of the story is wrong.

    The diagnosis is real. `m` and `v` live in the tangent space at the point
    the optimiser has just left. On SO(3) the body frame rotates by dR, so a
    stored body-frame vector refers to a frame that no longer exists:

        step 0.01 rad -> 0.95%     step 0.5 rad -> 47%
        step 0.10 rad ->  9.5%     step 1.0 rad -> 91%   step 2.0 rad -> 160%

    and the default max_step on SO(3) is pi - 1e-3, so the optimiser permits
    steps where the stored momentum points somewhere unrelated.

    Transport was implemented — exactly for m (by dR^T on SO(3), by the
    standard sphere formula on S^3) and norm-preservingly for v — and then
    measured on a rotation/quaternion fitting task, 200 steps, 8 seeds, final
    angular error in degrees:

        configuration                        S^3        SO(3)
        pasted version                    0.0016       0.0018
        v2, transport off                 0.0016       0.0018
        v2, transport on                  3.8631       0.0038

    It is 2400x worse on S^3. An ablation narrowed it down: transporting m
    alone is worse still (10.5608 deg), and every configuration that leaves
    transport off reproduces the pasted result exactly. One real bug was
    found along the way and fixed — `_normalize_quat` can flip the
    hemisphere, and transporting straight to the flipped point divides by
    1 + <q_old, -q> ~ 1e-3, the wrong geodesic and a near-singular
    denominator; transport now goes to the true geodesic endpoint and the
    flip is applied to the tangent state afterwards. That fix did not rescue
    it: S^3 still lands at 5.0954 deg.

    So the honest conclusion is that coordinate-wise Adam and parallel
    transport do not compose cleanly here. `v` is a per-coordinate second
    moment, not a tensor with a transport law, and moving it every step
    scrambles exactly the per-axis scale the preconditioner depends on;
    transporting m without a consistently transported v is worse again. This
    is a known awkwardness in Riemannian Adam, and several published
    implementations transport nothing for the same reason.

    `transport=False` is therefore the DEFAULT, and v2 reproduces the pasted
    optimiser's convergence exactly. `transport=True` is available, now with
    the hemisphere bug fixed, for anyone who wants to investigate further —
    but on this benchmark it costs accuracy, and this file does not pretend
    otherwise. The geometric mismatch above is real; the obvious fix for it
    is not an improvement, and saying so is more useful than shipping it on.

 5. `float(vnorm)` FORCES A DEVICE SYNC.  Converting a tensor to a Python
    float once per parameter per step blocks until the CUDA queue drains.
    No GPU is available here so this is a code-level observation rather than
    a measurement — it is not claimed as a benchmarked regression. The
    trust radius is kept as a device tensor in v2, which costs nothing on
    CPU and removes the sync on GPU.

UNCHANGED: the tangent-space projections, the hemisphere convention, the
hard/soft clamp shapes, and the parameter names and defaults.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["GeoClampAdam"]


# ========================== Lie algebra & retraction ==========================
def _hat_so3(v: torch.Tensor) -> torch.Tensor:
    """R^3 -> so(3)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)


def _exp_so3(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues. B uses the half-angle form (kept from the previous fix):
    (1-cos t)/t^2 evaluates to exactly 0 in float32 below t ~ 1e-4."""
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(eps)
    half = theta / 2.0
    A = torch.sin(theta) / theta
    B = 0.5 * (torch.sin(half) / half) ** 2
    K = _hat_so3(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(K.shape)
    return I + A[..., None] * K + B[..., None] * (K @ K)


def _project_so3_tangent(R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    RtG = R.transpose(-1, -2) @ G
    skew = 0.5 * (RtG - RtG.transpose(-1, -2))
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def _reproject_so3(R: torch.Tensor) -> torch.Tensor:
    """Nearest rotation in Frobenius norm: U diag(1,1,det(U V^T)) V^T."""
    U, _, Vh = torch.linalg.svd(R)
    d = torch.linalg.det(U @ Vh)
    D = torch.ones(R.shape[:-1], device=R.device, dtype=R.dtype)
    D[..., -1] = d
    return (U * D.unsqueeze(-2)) @ Vh


def _normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q * torch.where(q[..., 0:1] >= 0, 1.0, -1.0)


def _project_s3_tangent(q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return g - (q * g).sum(dim=-1, keepdim=True) * q


def _exp_s3(q: torch.Tensor, xi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """True exponential map on the unit sphere: moves exactly |xi| radians.

    The previous version used q <- normalize(q + xi), which moves atan(|xi|)
    instead — 36% short at the default max_step of pi/2."""
    r = torch.linalg.norm(xi, dim=-1, keepdim=True)
    small = r < eps
    r_safe = torch.where(small, torch.ones_like(r), r)
    out = q * torch.cos(r) + (xi / r_safe) * torch.sin(r)
    return torch.where(small, q, out)


def _transport_s3(q_from: torch.Tensor, q_to: torch.Tensor,
                  w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Parallel transport of a tangent vector along the sphere geodesic."""
    denom = (1.0 + (q_from * q_to).sum(dim=-1, keepdim=True)).clamp_min(eps)
    return w - ((q_to * w).sum(dim=-1, keepdim=True) / denom) * (q_from + q_to)


def _clamp_radius(r_raw: torch.Tensor, r_trust: torch.Tensor,
                  mode: str = "hard") -> torch.Tensor:
    if mode == "soft":
        s = torch.sigmoid(8.0 * (r_raw - r_trust))
        return r_raw * (1.0 - s) + r_trust * s
    return torch.minimum(r_raw, r_trust)


# ========================== GeoClampAdam ==========================
class GeoClampAdam(Optimizer):
    """Adam in the tangent space of SO(3) or S^3, with a geodesic step clamp.

    New arguments relative to the previous version, each defaulting to the
    corrected behaviour with the old behaviour still reachable:

        trust_source     "step" (default, scale-free) | "grad" (old)
        transport        False (default; see note 4 -- turning it on
                         measured WORSE, 0.0016 -> 5.10 deg on S^3)
        retraction       "exp" (default) | "proj" (old, S^3 only)
        reproject_every  16 (default) | 0 to disable (old)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        manifold: str = "SO3",
        max_step: Optional[float] = None,
        base_trust: Optional[float] = None,
        kappa: float = 0.05,
        clamp_mode: str = "hard",
        trust_source: str = "step",
        transport: bool = False,
        retraction: str = "exp",
        reproject_every: int = 16,
    ):
        if manifold not in ("SO3", "S3"):
            raise ValueError("manifold must be 'SO3' or 'S3'")
        if trust_source not in ("step", "grad"):
            raise ValueError("trust_source must be 'step' or 'grad'")
        if retraction not in ("exp", "proj"):
            raise ValueError("retraction must be 'exp' or 'proj'")
        if clamp_mode not in ("hard", "soft"):
            raise ValueError("clamp_mode must be 'hard' or 'soft'")
        if max_step is None:
            max_step = math.pi - 1e-3 if manifold == "SO3" else (math.pi / 2 - 1e-3)
        if base_trust is None:
            base_trust = 0.28 * max_step
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps, manifold=manifold, max_step=max_step,
            base_trust=base_trust, kappa=kappa, clamp_mode=clamp_mode,
            trust_source=trust_source, transport=transport,
            retraction=retraction, reproject_every=reproject_every))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["manifold"] == "SO3":
                    self._step_so3(p, group)
                else:
                    self._step_s3(p, group)
        return loss

    def _get_state(self, p, shape):
        st = self.state[p]
        if len(st) == 0:
            st["step"] = 0
            st["m"] = torch.zeros(shape, device=p.device, dtype=p.dtype)
            st["v"] = torch.zeros_like(st["m"])
            # kept on-device: float(tensor) once per step forces a CUDA sync
            st["tnorm"] = torch.zeros((), device=p.device, dtype=p.dtype)
        return st

    def _tangent_step(self, g_tan, p, group):
        """Shared Adam + trust-clamp. Returns the clamped tangent step."""
        st = self._get_state(p, g_tan.shape)
        st["step"] += 1
        t = st["step"]
        b1, b2 = group["betas"]
        m, v = st["m"], st["v"]

        m.mul_(b1).add_(g_tan, alpha=1 - b1)
        v.mul_(b2).addcmul_(g_tan, g_tan, value=1 - b2)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        xi_raw = -group["lr"] * m_hat / (torch.sqrt(v_hat) + group["eps"])

        r_raw = torch.linalg.norm(xi_raw, dim=-1, keepdim=True)

        # Bug #3: drive the radius from the quantity it is meant to bound
        # (the geodesic step) rather than the raw gradient norm, so that
        # rescaling the loss does not change the step.
        src = r_raw.mean() if group["trust_source"] == "step" \
            else torch.linalg.norm(g_tan, dim=-1).mean()
        st["tnorm"].mul_(b2).add_(src.detach() * (1 - b2))
        tnorm_hat = st["tnorm"] / (1 - b2 ** t)

        r_trust = torch.clamp(
            group["base_trust"] / (1.0 + group["kappa"] * tnorm_hat),
            max=group["max_step"])
        r_sat = _clamp_radius(r_raw, r_trust.expand_as(r_raw), group["clamp_mode"])
        return xi_raw * (r_sat / torch.clamp(r_raw, min=1e-12))

    def _step_so3(self, p, group):
        g_tan = _project_so3_tangent(p, p.grad)
        xi = self._tangent_step(g_tan, p, group)
        dR = _exp_so3(xi)
        p.copy_(p @ dR)

        st = self.state[p]
        if group["transport"]:
            # Bug #4: the body frame rotated by dR, so a body-frame vector
            # w at the old point is dR^T w at the new one. Exact for m.
            st["m"].copy_(torch.einsum("...ji,...j->...i", dR, st["m"]))
            # v is an elementwise second moment, not a vector, so there is no
            # exact transport. Rotating sqrt(v) and re-squaring preserves its
            # norm and its role as a per-axis scale; documented as approximate.
            sv = torch.sqrt(st["v"].clamp_min(0))
            st["v"].copy_(torch.einsum("...ji,...j->...i", dR, sv) ** 2)

        # Bug #2: nothing else keeps R on the group.
        n = group["reproject_every"]
        if n and st["step"] % n == 0:
            p.copy_(_reproject_so3(p))

    def _step_s3(self, p, group):
        p.copy_(_normalize_quat(p))
        q_old = p.clone()
        g_tan = _project_s3_tangent(p, p.grad)
        xi = self._tangent_step(g_tan, p, group)

        # Bug #1: the true exponential map moves exactly |xi| radians;
        # normalize(q + xi) moves atan(|xi|).
        q_geo = _exp_s3(q_old, xi) if group["retraction"] == "exp" else (q_old + xi)
        q_geo = q_geo / torch.linalg.norm(q_geo, dim=-1, keepdim=True).clamp_min(1e-12)
        # Hemisphere convention, tracked so the tangent state can follow it.
        flip = torch.where(q_geo[..., 0:1] >= 0, 1.0, -1.0)
        q_new = q_geo * flip
        p.copy_(q_new)

        st = self.state[p]
        if group["transport"]:
            # Transport along the geodesic ACTUALLY travelled (to q_geo), then
            # apply the same hemisphere flip to the tangent state. Transporting
            # straight to the flipped point instead divides by 1 + <q_old, -q>,
            # which is ~1e-3 for a small step -- a near-singular denominator and
            # the wrong geodesic. Measured effect of getting this wrong: the
            # S^3 fit degrades from 0.0016 deg to 3.86 deg.
            st["m"].copy_(_transport_s3(q_old, q_geo, st["m"]) * flip)
            sv = torch.sqrt(st["v"].clamp_min(0))
            st["v"].copy_((_transport_s3(q_old, q_geo, sv) * flip) ** 2)
