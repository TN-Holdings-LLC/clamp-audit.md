# -*- coding: utf-8 -*-
"""
PSF-Zero Head: /0 projection + EIT + S^3 shortest-arc (Quaternion)

A pre-processing head across 4 domains (Quantum / Robotics / PLL / Affective Cone)
with an A/B test harness comparing ON/OFF under identical code paths.

Dependencies: numpy (matplotlib for plotting only, skipped if missing)

    $ python psfzero_engine.py                 # Run A/B + output logs/ + plot
    $ python psfzero_engine.py --selftest      # Run regression tests only
    $ python psfzero_engine.py --glitch-rate 0 # Compare without external glitches
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np


# =========================================================
# Quaternion Utilities (w, x, y, z) — SU(2) ≅ S^3
# =========================================================
def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def q_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def q_norm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q if n == 0.0 else q / n


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / n
    h = 0.5 * angle
    s = math.sin(h)
    return q_norm(np.array([math.cos(h), *(s * a)], dtype=float))


def rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    vq = np.concatenate([[0.0], v])
    return q_mul(q_mul(q, vq), q_conj(q))[1:]


def quat_to_bloch_angles(q: np.ndarray) -> tuple[float, float]:
    """Rotates the Z-axis vector to obtain Bloch angles (theta, phi) for visualization."""
    x, y, z = rotate_vec_by_quat(np.array([0.0, 0.0, 1.0]), q_norm(q))
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)
    return theta, phi


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# =========================================================
# PSF-Zero Head
# =========================================================
def clamp_delta(dtheta: float, sigma: float = 1.0) -> float:
    """/0 Projection: sigma * Δ / sqrt(sigma^2 + Δ^2). Saturates to ±sigma as |Δ| → ∞."""
    return sigma * dtheta / math.sqrt(sigma * sigma + dtheta * dtheta)


def eit(zbar: complex, phi: float, lam: float = 0.1) -> complex:
    """EIT: Exponential moving average phase tracker on S^1."""
    return (1.0 - lam) * zbar + lam * complex(math.cos(phi), math.sin(phi))


def su2_update_minimal_arc(q: np.ndarray, axis: np.ndarray, dtheta: float) -> np.ndarray:
    """S^3 Shortest Arc Update: dq * q"""
    return q_norm(q_mul(q_from_axis_angle(axis, dtheta), q))


@dataclass
class HeadConfig:
    """Configuration allowing independent ON/OFF toggling of head mechanisms.

    enabled=False acts as a complete bypass mode. Avoids hacks like inflating sigma
    to disable clamping, which causes unintended alternate forms of suppression.
    """
    enabled: bool = True
    clamp: bool = True
    sigma: float = 1.0
    eit: bool = True
    lam: float = 0.10
    guard: bool = True
    max_phase_jump: float = math.radians(60)
    abstain_policy: str = "hold"       # hold | scale | apply
    latency_guard: bool = False
    max_latency_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.abstain_policy not in ("hold", "scale", "apply"):
            raise ValueError(f"Unknown abstain_policy: {self.abstain_policy}")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be within [0, 1]")

    @classmethod
    def bypass(cls) -> "HeadConfig":
        return cls(enabled=False)


@dataclass
class Metrics:
    steps: int = 0
    quatmul_calls: int = 0
    time_ms: float = 0.0
    abstain: int = 0
    fallback: int = 0
    saturated: int = 0
    sum_abs_applied: float = 0.0
    max_abs_applied: float = 0.0
    max_abs_raw: float = 0.0

    def as_summary(self) -> dict:
        n = max(self.steps, 1)
        return {
            "steps": self.steps,
            "quatmul_calls": self.quatmul_calls,
            "mean_latency_ms": self.time_ms / n,
            "abstain_rate": self.abstain / n,
            "fallback_rate": self.fallback / n,
            "saturation_rate": self.saturated / n,
            "mean_abs_applied": self.sum_abs_applied / n,
            "max_abs_applied": self.max_abs_applied,
            "max_abs_raw": self.max_abs_raw,
        }


@dataclass
class StepResult:
    applied: float
    zbar: complex
    action: str
    reason: str
    saturated: bool
    latency_ms: float


class PSFZeroHead:
    """Consolidated head managing clamping, EIT, and safety guards.

    Both quaternion channels (quat_step) and scalar phase channels (scalar_step)
    pass through this class, ensuring shared verification logic.
    """

    SAT_RATIO = 0.98  # Saturation threshold: |applied| >= 0.98 * sigma

    def __init__(self, cfg: HeadConfig | None = None):
        self.cfg = cfg if cfg is not None else HeadConfig()
        self.metrics = Metrics()

    # --- Common processing path: Raw command -> Applied value + Decision ---
    def _process(self, raw: float, phi: float, zbar: complex) -> StepResult:
        c = self.cfg
        t0 = time.perf_counter()

        action, reason, saturated = "CONTINUE", "", False

        if not c.enabled:
            applied = raw
        else:
            # The safety guard inspects the RAW command before clamping.
            # Post-clamping values are bounded by ±sigma by definition and would never trigger.
            if c.guard and abs(raw) > c.max_phase_jump:
                action, reason = "ABSTAIN", "excess-phase-jump"

            applied = clamp_delta(raw, c.sigma) if c.clamp else raw
            if c.clamp:
                saturated = abs(applied) >= self.SAT_RATIO * c.sigma

            if action == "ABSTAIN":
                if c.abstain_policy == "hold":
                    applied = 0.0
                elif c.abstain_policy == "scale":
                    applied = math.copysign(min(abs(applied), c.max_phase_jump), raw)
                # "apply" keeps applied as-is (legacy replication mode)

            if c.eit:
                zbar = eit(zbar, phi, c.lam)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if c.enabled and c.latency_guard and latency_ms > c.max_latency_ms:
            action, reason = "FALLBACK", "latency-exceeded"
            applied = 0.0

        m = self.metrics
        m.steps += 1
        m.time_ms += latency_ms
        m.sum_abs_applied += abs(applied)
        m.max_abs_applied = max(m.max_abs_applied, abs(applied))
        m.max_abs_raw = max(m.max_abs_raw, abs(raw))
        if action == "ABSTAIN":
            m.abstain += 1
        elif action == "FALLBACK":
            m.fallback += 1
        if saturated:
            m.saturated += 1

        return StepResult(applied, zbar, action, reason, saturated, latency_ms)

    # --- Quaternion channel ---
    def quat_step(self, q: np.ndarray, phi: float, zbar: complex,
                  axis: np.ndarray, raw_dtheta: float) -> tuple[np.ndarray, StepResult]:
        res = self._process(raw_dtheta, phi, zbar)
        # Execute quaternion product even if applied=0 to maintain equal computational budgets between ON/OFF.
        q_new = su2_update_minimal_arc(q, axis, res.applied)
        self.metrics.quatmul_calls += 1
        return q_new, res

    # --- Scalar phase channel (for PLL) ---
    def scalar_step(self, raw: float, phi: float, zbar: complex) -> StepResult:
        return self._process(raw, phi, zbar)


def eit_readout(zbar: complex) -> tuple[float, float]:
    """Observables from EIT tracker: (tracked phase [0, 2π), coherence |zbar|)."""
    return (math.atan2(zbar.imag, zbar.real) + 2 * math.pi) % (2 * math.pi), abs(zbar)


# =========================================================
# Domain 1: Quantum Bloch Precession
# =========================================================
@dataclass
class QuantumCfg:
    omega: float = 2.0
    dt: float = 0.02
    head: HeadConfig = field(default_factory=HeadConfig)


class QuantumEngine:
    name = "quantum"

    def __init__(self, cfg: QuantumCfg | None = None):
        self.cfg = cfg if cfg is not None else QuantumCfg()
        self.head = PSFZeroHead(self.cfg.head)
        self.reset()

    @property
    def metrics(self) -> Metrics:
        return self.head.metrics

    def reset(self, theta0: float = math.radians(20.0), phi0: float = 0.0) -> None:
        self.q = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), theta0)
        self.theta, self.phi = theta0, phi0
        self.zbar = complex(math.cos(phi0), math.sin(phi0))
        self.t = 0.0

    def step(self, stimulus: np.ndarray | None = None) -> dict:
        v = np.zeros(2) if stimulus is None else np.asarray(stimulus, float)
        axis = np.array([v[0], v[1], 0.2])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
        raw = float(min(0.5 * float(np.linalg.norm(v)), math.pi))

        q_new, res = self.head.quat_step(self.q, self.phi, self.zbar, axis, raw)
        self.q = q_new
        self.phi = (self.phi + self.cfg.omega * self.cfg.dt) % (2 * math.pi)
        self.theta, _ = quat_to_bloch_angles(q_new)
        self.zbar = res.zbar
        self.t += self.cfg.dt

        eit_phi, eit_coh = eit_readout(self.zbar)
        return {"t": self.t, "theta": self.theta, "phi": self.phi,
                "raw_dtheta": raw, "applied_dtheta": res.applied,
                "saturated": res.saturated, "action": res.action, "reason": res.reason,
                "eit_phase": eit_phi, "eit_coherence": eit_coh}


# =========================================================
# Domain 2: Robotics (IMU Attitude Filter Prototype)
# =========================================================
@dataclass
class RoboCfg:
    dt: float = 0.01
    head: HeadConfig = field(default_factory=HeadConfig)


class RoboEngine:
    name = "robo"

    def __init__(self, cfg: RoboCfg | None = None):
        self.cfg = cfg if cfg is not None else RoboCfg()
        self.head = PSFZeroHead(self.cfg.head)
        self.reset()

    @property
    def metrics(self) -> Metrics:
        return self.head.metrics

    def reset(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.t = 0.0

    def step(self, gyro: np.ndarray) -> dict:
        g = np.asarray(gyro, float)
        n = float(np.linalg.norm(g))
        axis = g / n if n > 0 else np.array([0.0, 0.0, 1.0])
        raw = n * self.cfg.dt

        q_new, res = self.head.quat_step(self.q, 0.0, 1 + 0j, axis, raw)
        self.q = q_new
        self.t += self.cfg.dt
        theta, phi = quat_to_bloch_angles(q_new)

        return {"t": self.t, "theta": theta, "phi": phi,
                "raw_dtheta": raw, "applied_dtheta": res.applied,
                "saturated": res.saturated, "action": res.action, "reason": res.reason,
                "eit_phase": 0.0, "eit_coherence": 1.0}


# =========================================================
# Domain 3: PLL Phase Synchronization
# =========================================================
@dataclass
class PLLCfg:
    dt: float = 0.01
    kp: float = 8.0                  # Loop gain [1/s]
    feedforward: bool = True         # Utilize omega_ref
    eit_feedback: bool = False       # Use EIT for self-phase estimation (default off)
    head: HeadConfig = field(default_factory=HeadConfig)


class PLLEngine:
    name = "pll"

    def __init__(self, cfg: PLLCfg | None = None):
        self.cfg = cfg if cfg is not None else PLLCfg()
        self.head = PSFZeroHead(self.cfg.head)
        self.reset()

    @property
    def metrics(self) -> Metrics:
        return self.head.metrics

    def reset(self, phi0: float = 0.0) -> None:
        self.phi = phi0
        self.zbar = complex(math.cos(phi0), math.sin(phi0))
        self.t = 0.0

    def step(self, phi_ref: float, omega_ref: float) -> dict:
        c = self.cfg
        phi_est = self.phi
        if c.eit_feedback and abs(self.zbar) > 1e-12:
            phi_est = math.atan2(self.zbar.imag, self.zbar.real)
        err = wrap_pi(phi_ref - phi_est)

        res = self.head.scalar_step(err, self.phi, self.zbar)
        ff = omega_ref * c.dt if c.feedforward else 0.0
        dphi = ff + c.kp * res.applied * c.dt

        self.phi = (self.phi + dphi) % (2 * math.pi)
        self.zbar = res.zbar
        self.t += c.dt

        eit_phi, eit_coh = eit_readout(self.zbar)
        return {"t": self.t, "phi": self.phi, "phi_ref": phi_ref,
                "phase_error": wrap_pi(phi_ref - self.phi),
                "raw_dtheta": err, "applied_dtheta": res.applied, "applied_dphi": dphi,
                "saturated": res.saturated, "action": res.action, "reason": res.reason,
                "eit_phase": eit_phi, "eit_coherence": eit_coh}


# =========================================================
# Domain 4: Affective Cone (S^2 Solid Angle Dynamics)
# =========================================================
@dataclass
class AffectCfg:
    alpha: float = 1.0
    beta: float = 1.0
    omega: float = 2.0
    dt: float = 0.02
    head: HeadConfig = field(default_factory=HeadConfig)


class AffectEngine:
    name = "affect"

    def __init__(self, cfg: AffectCfg | None = None):
        self.cfg = cfg if cfg is not None else AffectCfg()
        self.head = PSFZeroHead(self.cfg.head)
        self.reset()

    @property
    def metrics(self) -> Metrics:
        return self.head.metrics

    def reset(self, theta0: float = math.radians(15), phi0: float = 0.0) -> None:
        self.q = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), theta0)
        self.theta, self.phi = theta0, phi0
        self.zbar = complex(math.cos(phi0), math.sin(phi0))
        self.t = 0.0

    def step(self, stimulus: np.ndarray | None = None) -> dict:
        c = self.cfg
        v = np.zeros(2) if stimulus is None else np.asarray(stimulus, float)
        axis = np.array([v[0], v[1], 0.2])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
        raw = float(min(0.5 * float(np.linalg.norm(v)), math.pi))

        q_new, res = self.head.quat_step(self.q, self.phi, self.zbar, axis, raw)
        self.q = q_new
        self.phi = (self.phi + c.omega * c.dt) % (2 * math.pi)
        self.theta, _ = quat_to_bloch_angles(q_new)
        self.zbar = res.zbar
        self.t += c.dt

        solid_angle = 2 * math.pi * (1 - math.cos(self.theta))
        e_val = c.alpha * solid_angle + c.beta * (abs(c.omega) * math.sin(self.theta))
        eit_phi, eit_coh = eit_readout(self.zbar)
        return {"t": self.t, "theta": self.theta, "phi": self.phi,
                "E_component": e_val,
                "raw_dtheta": raw, "applied_dtheta": res.applied,
                "saturated": res.saturated, "action": res.action, "reason": res.reason,
                "eit_phase": eit_phi, "eit_coherence": eit_coh}


# =========================================================
# Stimuli (Shared identical sequences across arms)
# =========================================================
def glitch_schedule(steps: int, rate: float, seed: int) -> np.ndarray:
    """Reproducible sensor anomaly schedule shared by both arms."""
    rng = np.random.default_rng(seed)
    hit = rng.random(steps) < rate
    scale = np.where(hit, rng.uniform(8.0, 25.0, steps), 1.0)
    return scale


def quantum_stim(t: float) -> np.ndarray:
    return np.array([0.8 * math.sin(0.7 * t), 0.6 * math.cos(0.5 * t)])


def robo_gyro(t: float) -> np.ndarray:
    return np.array([2.0 * math.sin(1.1 * t), 1.8 * math.cos(0.9 * t), 0.6 * math.sin(0.7 * t)])


def pll_ref(t: float) -> tuple[float, float]:
    return (2.5 * t + 0.5 * math.sin(1.7 * t)) % (2 * math.pi), 2.5 + 0.85 * math.cos(1.7 * t)


def affect_stim(t: float) -> np.ndarray:
    return np.array([0.9 * math.sin(0.6 * t), 0.7 * math.cos(0.4 * t)])


# =========================================================
# A/B Test Harness
# =========================================================
def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def run_arm(make_engine: Callable[[HeadConfig], object],
            drive: Callable[[object, int, float], dict],
            head_cfg: HeadConfig, steps: int, scale: np.ndarray) -> tuple[list[dict], Metrics]:
    eng = make_engine(head_cfg)
    rows = [drive(eng, i, float(scale[i])) for i in range(steps)]
    return rows, eng.metrics


def build_domains(seconds: float):
    """Returns a list of (name, engine_factory, drive_fn, dt)."""
    def quantum_factory(h): return QuantumEngine(QuantumCfg(head=h))
    def quantum_drive(e, i, s): return e.step(quantum_stim(e.t) * s)

    def robo_factory(h): return RoboEngine(RoboCfg(head=h))
    def robo_drive(e, i, s): return e.step(robo_gyro(e.t) * s)

    def pll_factory(h): return PLLEngine(PLLCfg(head=h))
    def pll_drive(e, i, s):
        clean_ref, omega_ref = pll_ref(e.t)
        phi_ref = clean_ref
        if s != 1.0:
            phi_ref = (clean_ref + math.pi * (s / 25.0 + 0.5)) % (2 * math.pi)
        row = e.step(phi_ref, omega_ref)
        row["phi_ref_clean"] = clean_ref
        row["phase_error_clean"] = wrap_pi(clean_ref - row["phi"])
        return row

    def affect_factory(h): return AffectEngine(AffectCfg(head=h))
    def affect_drive(e, i, s): return e.step(affect_stim(e.t) * s)

    return [
        ("quantum", quantum_factory, quantum_drive, 0.02),
        ("robo", robo_factory, robo_drive, 0.01),
        ("pll", pll_factory, pll_drive, 0.01),
        ("affect", affect_factory, affect_drive, 0.02),
    ]


def run_all(seconds: float, outdir: str, glitch_rate: float, seed: int,
            abstain_policy: str = "hold") -> list[dict]:
    summaries: list[dict] = []
    logs: dict[str, dict[str, list[dict]]] = {}

    for name, factory, drive, dt in build_domains(seconds):
        steps = int(seconds / dt)
        scale = glitch_schedule(steps, glitch_rate, seed)
        logs[name] = {}

        for arm, head_cfg in (("on", HeadConfig(abstain_policy=abstain_policy)),
                              ("off", HeadConfig.bypass())):
            rows, metrics = run_arm(factory, drive, head_cfg, steps, scale)
            write_csv(os.path.join(outdir, f"{name}_{arm}.csv"), rows)
            logs[name][arm] = rows

            summary = {"domain": name, "arm": arm, **metrics.as_summary()}
            if name == "pll":
                err = np.array([r["phase_error_clean"] for r in rows])
                summary["rms_phase_error_clean"] = float(np.sqrt(np.mean(err ** 2)))
                summary["max_abs_phase_error_clean"] = float(np.max(np.abs(err)))
            summaries.append(summary)
            print(f"  {name:<8} {arm:<3} steps={metrics.steps:<5} "
                  f"abstain={metrics.abstain:<4} sat={metrics.saturated:<5} "
                  f"max|applied|={metrics.max_abs_applied:.4f} rad")

    write_csv(os.path.join(outdir, "summary.csv"), summaries)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    return summaries


def plot_summary(outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Skipping plot generation.")
        return

    panels = [("quantum", "applied_dtheta", "Quantum: applied rotation [rad]"),
              ("robo", "applied_dtheta", "Robotics: applied rotation [rad]"),
              ("pll", "phase_error_clean", "PLL: phase error vs clean reference [rad]"),
              ("affect", "E_component", "Affect: E component")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    for ax, (name, col, title) in zip(axes.ravel(), panels):
        for arm, color in (("off", "#B04A3A"), ("on", "#1F5C7A")):
            path = os.path.join(outdir, f"{name}_{arm}.csv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            t = [float(r["t"]) for r in rows]
            y = [float(r[col]) for r in rows]
            ax.plot(t, y, lw=1.2, color=color, label=f"head {arm}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("PSF-Zero head A/B (identical stimulus, identical code path)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure to: {path}")


# =========================================================
# Regression Tests
# =========================================================
def selftest() -> None:
    print("selftest:")

    # 1) Clamp properties: identity when Δ << sigma, saturated to ±sigma when Δ >> sigma
    assert abs(clamp_delta(1e-3, 1.0) - 1e-3) < 1e-8
    assert abs(clamp_delta(1e6, 2.5) - 2.5) < 1e-3
    assert abs(clamp_delta(0.5, 1e9) - 0.5) < 1e-6, "Large sigma should be transparent/identity"
    print("  [ok] clamp_delta identity / saturation check")

    # 2) Bypass mode == raw quaternion calculation (verifying ON/OFF shares identical code path)
    eng = QuantumEngine(QuantumCfg(head=HeadConfig.bypass()))
    ref_q = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(20.0))
    ref_t = 0.0
    for _ in range(200):
        v = quantum_stim(ref_t)
        out = eng.step(v)
        axis = np.array([v[0], v[1], 0.2])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
        raw = min(0.5 * float(np.linalg.norm(v)), math.pi)
        ref_q = q_norm(q_mul(q_from_axis_angle(axis, raw), ref_q))
        ref_theta, _ = quat_to_bloch_angles(ref_q)
        assert abs(out["theta"] - ref_theta) < 1e-12
        ref_t += 0.02
    print("  [ok] bypass == legacy off arm manual calculation (200 steps match)")

    # 3) Guard triggers on raw commands, and 'hold' halts movement effectively
    head = PSFZeroHead(HeadConfig(abstain_policy="hold"))
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1, res = head.quat_step(q0, 0.0, 1 + 0j, np.array([0.0, 0.0, 1.0]), 30.0)
    assert res.action == "ABSTAIN", res.action
    assert res.applied == 0.0
    assert np.allclose(q1, q0), "Pose must remain unchanged during ABSTAIN"
    print("  [ok] raw=30rad → ABSTAIN with constant pose")

    head_apply = PSFZeroHead(HeadConfig(abstain_policy="apply"))
    _, res_a = head_apply.quat_step(q0, 0.0, 1 + 0j, np.array([0.0, 0.0, 1.0]), 30.0)
    assert res_a.action == "ABSTAIN" and abs(res_a.applied - clamp_delta(30.0, 1.0)) < 1e-12
    print("  [ok] abstain_policy='apply' replicates legacy behavior")

    # 4) EIT is observable (changing lam modifies outputs)
    def coh(lam: float) -> float:
        e = QuantumEngine(QuantumCfg(head=HeadConfig(lam=lam)))
        for _ in range(100):
            out = e.step(quantum_stim(e.t))
        return out["eit_coherence"]
    assert abs(coh(0.9) - coh(0.05)) > 1e-6, "lam does not impact output"
    print("  [ok] lam reflects on output (eit_coherence)")

    # 5) PLL utilizes omega_ref
    e1 = PLLEngine(PLLCfg())
    e2 = PLLEngine(PLLCfg(feedforward=False))
    for _ in range(50):
        pr, om = pll_ref(e1.t)
        e1.step(pr, om)
        pr2, om2 = pll_ref(e2.t)
        e2.step(pr2, om2)
    assert abs(e1.phi - e2.phi) > 1e-6, "omega_ref is being ignored"
    print("  [ok] omega_ref functions effectively as feedforward")

    # 6) write_csv does not fail with paths lacking directories
    tmp = "._selftest_rows.csv"
    write_csv(tmp, [{"a": 1, "b": 2}])
    os.remove(tmp)
    print("  [ok] write_csv('bare.csv') raises no exception")

    print("selftest: all passed")


# =========================================================
# CLI
# =========================================================
def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PSF-Zero head 4-domain A/B harness")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--outdir", default="logs")
    p.add_argument("--glitch-rate", type=float, default=0.02, help="Sensor anomaly frequency rate")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--abstain-policy", choices=["hold", "scale", "apply"], default="hold")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        selftest()
        return

    ensure_dir(args.outdir)
    print(f"PSF-Zero A/B: seconds={args.seconds}, glitch_rate={args.glitch_rate}, "
          f"seed={args.seed}, abstain_policy={args.abstain_policy}")
    run_all(args.seconds, args.outdir, args.glitch_rate, args.seed, args.abstain_policy)
    if not args.no_plot:
        plot_summary(args.outdir)
    print(f"Complete. Output directory: {args.outdir}/")


if __name__ == "__main__":
    main()