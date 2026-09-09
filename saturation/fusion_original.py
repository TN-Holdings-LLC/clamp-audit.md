"""The pasted 'corrected' fusion simulator, verbatim (minus plotting)."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple


def q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dtype=float)


def q_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-14 else np.array([1.0, 0.0, 0.0, 0.0])


def q_from_axis_angle(axis, angle):
    a = np.asarray(axis, dtype=float)
    na = np.linalg.norm(a)
    if na < 1e-14 or abs(angle) < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / na
    h = 0.5 * angle
    s = np.sin(h)
    return q_norm(np.array([np.cos(h), s*a[0], s*a[1], s*a[2]]))


def clamp_delta(delta, sigma=1.0):
    return delta / np.sqrt(sigma**2 + delta**2)


def eit(zbar, phi, lam=0.12):
    return (1.0 - lam) * zbar + lam * complex(np.cos(phi), np.sin(phi))


@dataclass
class PSFZeroCfg:
    lam: float = 0.12
    sigma: float = 0.9
    max_phase_jump: float = np.deg2rad(55)
    enabled: bool = True


@dataclass
class PSFResult:
    q_next: np.ndarray
    dtheta_applied: float
    zbar_next: complex
    action: str


def psfzero_step(q, phi, zbar, axis, raw_dtheta, cfg):
    dtheta = clamp_delta(raw_dtheta, cfg.sigma) if cfg.enabled else raw_dtheta
    zbar_new = eit(zbar, phi, cfg.lam)
    axis_n = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis_n)
    axis_u = axis_n / n if n > 1e-14 else np.array([0., 0., 1.])
    dq = q_from_axis_angle(axis_u, dtheta)
    q_new = q_norm(q_mul(dq, q))
    action = "ABSTAIN" if abs(raw_dtheta) > cfg.max_phase_jump else "CONTINUE"
    return PSFResult(q_new, dtheta, zbar_new, action)


@dataclass
class FusionCfg:
    dt: float = 0.001
    omega_phi: float = 2.0
    coil_gain: float = 1.0
    dtheta_hw_limit: float = np.deg2rad(70)
    stimulus_gain: float = 0.85
    psf: PSFZeroCfg = field(default_factory=PSFZeroCfg)


class FusionAdapter:
    def __init__(self, cfg=None):
        self.cfg = cfg or FusionCfg()
        self.q = q_from_axis_angle(np.array([1., 0., 0.]), np.deg2rad(15))
        self.phi = 0.0
        self.zbar = complex(1.0, 0.0)
        self.t = 0.0
        self.coil_cmd = 0.0

    def sensors_to_stimulus(self, sensors):
        rad = sensors.get("rad", 0.0)
        bdot = sensors.get("bdot", 0.0)
        eci = sensors.get("eci", 0.0)
        v = np.array([0.65*rad + 0.25*bdot - 0.1*eci,
                      -0.15*rad + 0.70*bdot + 0.35*eci])
        mag = float(np.linalg.norm(v))
        direction = v / mag if mag > 1e-14 else np.array([0., 0.])
        return direction, mag

    def step(self, sensors):
        direction, mag = self.sensors_to_stimulus(sensors)
        axis = np.append(direction, 0.25)
        raw_dtheta = np.clip(self.cfg.stimulus_gain * mag, -np.pi, np.pi)
        res = psfzero_step(self.q, self.phi, self.zbar, axis, raw_dtheta, self.cfg.psf)
        self.phi = (self.phi + self.cfg.omega_phi * self.cfg.dt) % (2 * np.pi)
        dtheta_hw = np.clip(res.dtheta_applied, -self.cfg.dtheta_hw_limit,
                            self.cfg.dtheta_hw_limit)
        if res.action == "CONTINUE":
            self.coil_cmd += self.cfg.coil_gain * dtheta_hw
        self.q = res.q_next
        self.zbar = res.zbar_next
        self.t += self.cfg.dt
        return {"t": self.t, "coil_cmd": float(self.coil_cmd),
                "raw_dtheta": float(raw_dtheta),
                "applied_dtheta": float(res.dtheta_applied),
                "action": res.action, "hw_clipped": bool(
                    abs(res.dtheta_applied) > self.cfg.dtheta_hw_limit)}


def synthetic_sensors(t, rng):
    rad = 0.75*np.sin(2.3*t) + 0.3*np.cos(0.8*t)
    bdot = 0.65*np.cos(1.7*t) + 0.45*np.sin(1.2*t)
    eci = 0.55*np.sin(1.0*t + 0.5)
    if int(t*15) % 47 == 0:
        rad += 3.2
    if int(t*12) % 39 == 0:
        bdot += 2.4
    if int(t*18) % 61 == 0:
        eci -= 2.1
    rad += 0.08*rng.normal()
    bdot += 0.08*rng.normal()
    eci += 0.06*rng.normal()
    return {"rad": float(rad), "bdot": float(bdot), "eci": float(eci)}


def run_simulation(seconds=12.0, seed=42):
    dt = 0.001
    steps = int(seconds / dt)
    rng = np.random.default_rng(seed)
    a_on = FusionAdapter(FusionCfg(dt=dt))
    d_on = [a_on.step(synthetic_sensors(a_on.t, rng)) for _ in range(steps)]
    rng2 = np.random.default_rng(seed)
    a_off = FusionAdapter(FusionCfg(dt=dt, psf=PSFZeroCfg(enabled=False)))
    d_off = [a_off.step(synthetic_sensors(a_off.t, rng2)) for _ in range(steps)]
    return pd.DataFrame(d_on), pd.DataFrame(d_off)
