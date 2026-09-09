# The pasted "corrected" optimizer, verbatim.
from __future__ import annotations
import math
from typing import Iterable, Tuple, Optional
import torch
from torch.optim.optimizer import Optimizer

def _hat_so3(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack([torch.stack([O,-z,y],dim=-1),
                        torch.stack([z,O,-x],dim=-1),
                        torch.stack([-y,x,O],dim=-1)], dim=-2)

def _exp_so3(phi, eps=1e-8):
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(eps)
    half = theta/2.0
    A = torch.sin(theta)/theta
    B = 0.5*(torch.sin(half)/half)**2
    K = _hat_so3(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(K.shape)
    return I + A[...,None]*K + B[...,None]*(K@K)

def _exp_so3_pasted_B(phi, eps=1e-8):
    """The PRE-fix B, for comparison."""
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(eps)
    A = torch.sin(theta)/theta
    B = (1.0-torch.cos(theta))/theta**2
    K = _hat_so3(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(K.shape)
    return I + A[...,None]*K + B[...,None]*(K@K)

def _project_so3_tangent(R, G):
    RtG = R.transpose(-1,-2) @ G
    skew = 0.5*(RtG - RtG.transpose(-1,-2))
    return torch.stack([skew[...,2,1], skew[...,0,2], skew[...,1,0]], dim=-1)

def _normalize_quat(q, eps=1e-12):
    q = q/torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(q[...,0:1] >= 0, 1.0, -1.0)
    return q*sign

def _project_s3_tangent(q, g):
    return g - (q*g).sum(dim=-1, keepdim=True)*q

def _clamp_radius(r_raw, r_trust, mode="hard"):
    if mode == "soft":
        s = torch.sigmoid(8.0*(r_raw-r_trust))
        return r_raw*(1.0-s) + r_trust*s
    return torch.minimum(r_raw, r_trust)

class GeoClampAdam(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9,0.999), eps=1e-8,
                 manifold="SO3", max_step=None, base_trust=None,
                 kappa=0.05, clamp_mode="hard", bias_correction=True):
        if manifold not in ("SO3","S3"): raise ValueError("manifold")
        if max_step is None:
            max_step = math.pi-1e-3 if manifold=="SO3" else (math.pi/2-1e-3)
        if base_trust is None: base_trust = 0.28*max_step
        super().__init__(params, dict(lr=lr,betas=betas,eps=eps,manifold=manifold,
            max_step=max_step,base_trust=base_trust,kappa=kappa,
            clamp_mode=clamp_mode,bias_correction=bias_correction))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                (self._step_so3 if group["manifold"]=="SO3" else self._step_s3)(p, group)
        return loss

    def _get_state(self, p, shape):
        st = self.state[p]
        if len(st)==0:
            st["step"]=0
            st["m"]=torch.zeros(shape, device=p.device, dtype=p.dtype)
            st["v"]=torch.zeros_like(st["m"])
            st["vnorm"]=torch.zeros(1, device=p.device, dtype=p.dtype)
        return st

    def _core(self, g_tan, p, group):
        st = self._get_state(p, g_tan.shape)
        st["step"] += 1; t = st["step"]
        b1,b2 = group["betas"]; m,v,vnorm = st["m"],st["v"],st["vnorm"]
        m.mul_(b1).add_(g_tan, alpha=1-b1)
        v.mul_(b2).addcmul_(g_tan, g_tan, value=1-b2)
        if group["bias_correction"]:
            m_hat = m/(1-b1**t); v_hat = v/(1-b2**t)
        else:
            m_hat, v_hat = m, v
        xi_raw = -group["lr"]*m_hat*(1.0/(torch.sqrt(v_hat)+group["eps"]))
        cur = torch.linalg.norm(g_tan, dim=-1).mean()
        vnorm.mul_(b2).add_(cur.detach()*(1-b2))
        vh = float(vnorm)/(1-b2**t) if group["bias_correction"] else float(vnorm)
        r_trust = min(group["max_step"], group["base_trust"]/(1.0+group["kappa"]*vh))
        r_raw = torch.linalg.norm(xi_raw, dim=-1, keepdim=True)
        r_sat = _clamp_radius(r_raw, torch.full_like(r_raw, r_trust), group["clamp_mode"])
        xi = xi_raw*(r_sat/torch.clamp(r_raw, min=1e-12))
        return xi, float(r_raw.mean()), float(r_sat.mean()), r_trust

    def _step_so3(self, p, group):
        g_tan = _project_so3_tangent(p, p.grad)
        xi,_,_,_ = self._core(g_tan, p, group)
        p.copy_(p @ _exp_so3(xi))

    def _step_s3(self, p, group):
        p.copy_(_normalize_quat(p))
        g_tan = _project_s3_tangent(p, p.grad)
        xi,_,_,_ = self._core(g_tan, p, group)
        p.copy_(_normalize_quat(p + xi))
