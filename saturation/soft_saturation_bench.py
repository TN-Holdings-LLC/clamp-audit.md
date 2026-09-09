#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft vs. hard actuator saturation: a closed-loop benchmark
==========================================================

This replaces the "Love-OS PSF-Zero Fusion Plasma Control Simulator". The
name change is part of the fix: nothing in that file was a plasma model,
and nothing here is either. What it does contain is the one question the
original was reaching for and could not answer -- does a SMOOTH actuator
saturation behave better than a HARD clip at the same limit, inside a real
feedback loop?

WHY THE PREVIOUS VERSION COULD NOT ANSWER IT.  Its own three fixes were
real and reproduce: `raw_dtheta` now ranges 0.0064-3.1223 rad instead of
being frozen at 0.85, and the ~6.2% peak / ~5.8% RMS numbers come out as
stated. But the measurement underneath them does not support any claim
about saturation, for five measured reasons:

 1. `coil_cmd` NEVER DECREASES.  `raw_dtheta = clip(gain * ||v||, -pi, pi)`
    is a norm, so it is non-negative, so every increment is >= 0 --
    measured 0 negative steps out of 11,999. `coil_cmd` is therefore a
    monotone ramp (final value 5521 baseline / 5181 saturated), "peak"
    is just its endpoint, and "mean |coil_cmd|" is the mean of a ramp.
    Neither metric measures excursion suppression; there are no
    excursions, only a slope.

 2. THE 6.2% IS A CLOSED FORM, NOT A SIMULATION RESULT.  It equals
    1 - sum(clamp(raw)) / sum(raw) over the non-abstained steps, to the
    last digit: predicted 6.17%, simulated 6.17%. It is the average
    compression of the saturation curve over the input distribution, and
    can be computed in two lines without any dynamics.

 3. THE QUATERNION, EIT AND PHASE SUBSYSTEMS ARE DEAD CODE.  Re-running
    with a completely different initial quaternion, `zbar` and `phi`
    produces a bit-identical `coil_cmd` trace. `q` is integrated and never
    read; `eit()`'s output is stored and never used (the docstring admits
    this); `phi` only feeds `eit()`. That is most of the file.

 4. THE ABSTAIN GATE REMOVES EXACTLY THE EVENTS UNDER TEST.  It fires at
    raw > 55 deg (0.96 rad) for both arms identically -- 501/12000 steps.
    Measured raw range on CONTINUE steps: 0.01-0.96 rad; on ABSTAIN steps:
    0.96-3.12 rad. So every large excursion is sat out by both arms, and
    the saturation is only ever exercised on inputs too small to need it.
    The 70 deg "shared hardware limit" the docstring highlights is
    likewise unreachable: it only ever binds on steps that are abstained
    anyway.

 5. THE "MHD SPIKES" ARE 67 ms PLATEAUS.  `int(t*15) % 47 == 0` latches
    for the whole 1/15 s that `int(t*15)` holds its value -- measured 4
    runs of 66-67 consecutive steps, not impulses. And the 5-seed
    agreement (6.23-6.27%) is not evidence of robustness: the seed only
    perturbs noise of sd 0.06-0.08 riding on deterministic sinusoids of
    amplitude 0.55-0.75, so the input is ~90% identical by construction.

 6. `clamp_delta(d, sigma)` AMPLIFIES SMALL SIGNALS.  Its small-signal
    gain is 1/sigma, so at the configured sigma=0.9 every small command is
    multiplied by 1.111 before any saturation happens, and it saturates at
    1.0 regardless of sigma -- sigma sets the knee, not the limit. A
    saturation whose passband gain is not 1 is not comparable to a clip at
    the same limit. Fixed here: `soft_algebraic` is normalised to unit
    small-signal gain and an explicit limit.

WHAT THIS FILE DOES INSTEAD.  A second-order plant under PI control with
an actuator that saturates, driven by genuinely impulsive disturbances.
Four actuator arms at a MATCHED limit -- none / hard clip / soft algebraic
/ soft tanh -- plus an anti-windup switch, because integrator windup is
the actual mechanism by which saturation hurts a PI loop and is the thing
a smooth saturation might or might not help with. Metrics are tracking
error (true RMS, not mean-abs), peak excursion, recovery time after a
disturbance, control effort, and actuator chatter.

WHAT IT MEASURES (run the file to reproduce; 5 seeds per point).

There is a genuine crossover, and it goes both ways -- RMS tracking error
against actuator limit, anti-windup on:

    limit    hard   soft_alg  soft_tanh   winner
      0.4   0.2988    0.3354     0.3137   hard
      0.6   0.1853    0.2624     0.2355   hard
      0.8   0.2087    0.2130     0.1902   soft_tanh
      1.0   0.2566    0.1843     0.1730   soft_tanh
      1.4   0.4051    0.1747     0.1926   soft_algebraic
      3.0   0.8848    0.3423     0.4131   soft_algebraic

At a tight limit the loop needs every bit of authority it can get and the
soft curve throws some away, so the hard clip wins. As the limit loosens
the hard clip does nothing until the command reaches it, so it converges
toward the unlimited case -- which is unstable here (RMS 22.3, peak 103.4)
-- while the soft curve keeps compressing large commands wherever the
limit sits. That is the mechanism, and it is why the ratio reaches 2.6x at
limit 3.0.

DO NOT QUOTE THAT 2.6x.  Each arm has its own best limit, and comparing at
one shared limit mostly measures which arm that limit happens to suit.
Tuned per arm:

    hard             best limit 0.6   RMS 0.1854
    soft_algebraic   best limit 1.3   RMS 0.1723
    soft_tanh        best limit 1.1   RMS 0.1724

    -> soft saturation wins by 7.6%, not by 2.6x.

The larger and more practical difference is robustness to getting the
limit wrong -- worst/best RMS across limits 0.4-3.0:

    hard            0.1854 .. 0.8845   (4.8x)
    soft_algebraic  0.1746 .. 0.3420   (2.0x)
    soft_tanh       0.1729 .. 0.4129   (2.4x)

So the defensible claim is: soft saturation is worth about 7.6% at its own
optimum, and roughly halves how badly you are punished for mis-setting the
actuator limit. It is not worth 75%, and it is not worth 2.6x.

SCOPE, restated because the original file's title invited the opposite
reading: one second-order plant, one PI controller, one disturbance shape,
synthetic throughout. No plasma, no MHD, no tokamak. A different plant or
controller can move the crossover, and nothing here has been checked
against hardware.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["Saturation", "SATURATIONS", "Plant", "PIController",
           "SimConfig", "run_trial", "compare"]


# ====================== Actuator saturation models ======================
@dataclass(frozen=True)
class Saturation:
    """An actuator nonlinearity with an explicit limit.

    Every model here has unit small-signal gain (s(u) ~ u as u -> 0) and
    saturates at +/- `limit`, so the arms differ ONLY in how they bend
    between those two regimes. That is what makes the comparison fair --
    and is what the original `clamp_delta(d, sigma=0.9)` did not do, since
    its small-signal gain was 1/0.9 = 1.111 and its limit was fixed at 1.0
    independently of sigma.
    """

    name: str
    fn: Callable[[np.ndarray, float], np.ndarray]

    def __call__(self, u, limit: float):
        return self.fn(np.asarray(u, dtype=float), limit)


SATURATIONS: Dict[str, Saturation] = {
    "none": Saturation("none", lambda u, L: u),
    "hard": Saturation("hard", lambda u, L: np.clip(u, -L, L)),
    # The project's "/0 projective" form, correctly normalised:
    # unit gain at u->0, asymptote at L.
    "soft_algebraic": Saturation(
        "soft_algebraic", lambda u, L: u / np.sqrt(1.0 + (u / L) ** 2)),
    "soft_tanh": Saturation("soft_tanh", lambda u, L: L * np.tanh(u / L)),
}


# ====================== Plant ======================
@dataclass
class Plant:
    """Second-order oscillatory plant, discretised (semi-implicit Euler):

        xdd + 2*zeta*wn*xd + wn^2 * x = wn^2 * u + d(t)

    Lightly damped on purpose: an actuator limit only matters when the
    loop actually needs authority it cannot always get.
    """

    wn: float = 6.0
    zeta: float = 0.10
    dt: float = 0.001
    x: float = 0.0
    xd: float = 0.0

    def step(self, u: float, dist: float = 0.0) -> float:
        xdd = (self.wn ** 2) * (u - self.x) - 2 * self.zeta * self.wn * self.xd + dist
        self.xd += xdd * self.dt
        self.x += self.xd * self.dt
        return self.x


# ====================== Controller ======================
@dataclass
class PIController:
    """PI with optional back-calculation anti-windup.

    `anti_windup` feeds the actuator's realised output back into the
    integrator, which is the standard remedy for the windup that any
    saturation causes. It is a switch here because whether a smooth
    saturation helps depends entirely on whether windup is already
    handled -- reporting one without the other would be cherry-picking.
    """

    kp: float = 2.5
    ki: float = 8.0
    dt: float = 0.001
    anti_windup: bool = True
    tt: float = 0.05          # back-calculation time constant
    _i: float = 0.0

    def reset(self) -> None:
        self._i = 0.0

    def __call__(self, err: float, u_prev_raw: float, u_prev_sat: float) -> float:
        if self.anti_windup:
            self._i += (self.ki * err + (u_prev_sat - u_prev_raw) / self.tt) * self.dt
        else:
            self._i += self.ki * err * self.dt
        return self.kp * err + self._i


# ====================== Simulation ======================
@dataclass
class SimConfig:
    seconds: float = 12.0
    dt: float = 0.001
    limit: float = 1.0
    setpoint_hz: float = 0.35
    setpoint_amp: float = 0.8
    noise_sd: float = 0.01
    # Genuine impulses: a few samples wide, not a 67 ms latched plateau.
    spike_times: tuple = (2.0, 4.5, 7.0, 9.5)
    spike_width_ms: float = 3.0
    spike_amp: float = 900.0
    anti_windup: bool = True
    abstain_deg: Optional[float] = None   # None = gate off (default)


def _disturbance(cfg: SimConfig, n: int) -> np.ndarray:
    d = np.zeros(n)
    w = max(1, int(cfg.spike_width_ms * 1e-3 / cfg.dt))
    for k, ts in enumerate(cfg.spike_times):
        i = int(ts / cfg.dt)
        if i + w <= n:
            d[i:i + w] += cfg.spike_amp * (-1.0 if k % 2 else 1.0)
    return d


def run_trial(sat_name: str, cfg: SimConfig = None, seed: int = 0) -> dict:
    cfg = cfg or SimConfig()
    sat = SATURATIONS[sat_name]
    n = int(cfg.seconds / cfg.dt)
    t = np.arange(n) * cfg.dt
    rng = np.random.default_rng(seed)

    r = cfg.setpoint_amp * np.sin(2 * np.pi * cfg.setpoint_hz * t)
    dist = _disturbance(cfg, n)
    meas_noise = rng.normal(0.0, cfg.noise_sd, n)

    plant = Plant(dt=cfg.dt)
    pi = PIController(dt=cfg.dt, anti_windup=cfg.anti_windup)
    thr = np.deg2rad(cfg.abstain_deg) if cfg.abstain_deg is not None else None

    x = np.zeros(n); u_raw = np.zeros(n); u_app = np.zeros(n)
    abstained = np.zeros(n, dtype=bool)
    pr, ps = 0.0, 0.0
    for i in range(n):
        y = plant.x + meas_noise[i]
        e = r[i] - y
        ur = pi(e, pr, ps)
        us = float(sat(ur, cfg.limit))
        # If an abstain gate is enabled it is judged on the RAW command for
        # every arm alike -- judging it on the post-saturation value would
        # let precisely the smoothest arm through during the worst events.
        if thr is not None and abs(ur) > thr:
            abstained[i] = True
            us = 0.0
        x[i] = plant.step(us, dist[i])
        u_raw[i], u_app[i] = ur, us
        pr, ps = ur, us

    err = r - x
    # Post-spike IAE over a fixed 1.5 s window after each impulse. A
    # settling-time criterion ("|err| under a tolerance for 100 ms") is
    # not usable here: the setpoint is a continuously moving sinusoid, so
    # the error never parks near zero and the criterion returned NaN for
    # every arm. Integrated absolute error over a fixed window is always
    # defined and is directly comparable between arms.
    w = int(1.5 / cfg.dt)
    iae = [np.abs(err[int(ts / cfg.dt): int(ts / cfg.dt) + w]).sum() * cfg.dt
           for ts in cfg.spike_times if int(ts / cfg.dt) + w <= n]

    return {
        "arm": sat_name, "t": t, "r": r, "x": x, "err": err,
        "u_raw": u_raw, "u_app": u_app, "abstained": abstained,
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
        "peak_err": float(np.max(np.abs(err))),
        "effort": float(np.sqrt(np.mean(u_app ** 2))),
        # Chatter: mean |du| per step, in limit units. A hard clip riding
        # its corner produces more of this than a smooth curve does.
        "chatter": float(np.mean(np.abs(np.diff(u_app))) / cfg.limit),
        # Fraction of time within 10% of the limit. NOTE this is not
        # symmetric between arms by construction: a soft curve approaches
        # its asymptote only for very large u (soft_algebraic needs
        # u = 7L to reach 0.99L), so a low number here means "rarely near
        # the asymptote", not "rarely limiting". Compare `duty` instead.
        "near_limit": float(np.mean(np.abs(u_app) > 0.9 * cfg.limit)),
        # Mean |u|/L -- how hard the actuator is worked, comparably.
        "duty": float(np.mean(np.abs(u_app)) / cfg.limit),
        "post_spike_iae": float(np.mean(iae)) if iae else np.nan,
        "n_abstain": int(abstained.sum()),
    }


METRICS = ("rms_err", "peak_err", "post_spike_iae", "effort", "chatter", "duty")


def compare(cfg: SimConfig = None, seeds=range(5), arms=None) -> dict:
    cfg = cfg or SimConfig()
    arms = arms or ["hard", "soft_algebraic", "soft_tanh"]
    out = {}
    for a in arms:
        rs = [run_trial(a, cfg, s) for s in seeds]
        out[a] = {k: float(np.nanmean([r[k] for r in rs])) for k in METRICS}
        out[a]["rms_err_sd"] = float(np.std([r["rms_err"] for r in rs]))
    return out


def _table(title: str, res: dict) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    print(f"  {'arm':16}{'RMS err':>11}{'+/-':>8}{'peak err':>10}"
          f"{'spike IAE':>11}{'effort':>9}{'chatter':>9}{'duty':>8}")
    best = min(res, key=lambda a: res[a]["rms_err"])
    for a, m in res.items():
        mark = "  <-- best" if a == best else ""
        print(f"  {a:16}{m['rms_err']:>11.4f}{m['rms_err_sd']:>8.4f}"
              f"{m['peak_err']:>10.4f}{m['post_spike_iae']:>11.4f}"
              f"{m['effort']:>9.4f}{m['chatter']:>9.5f}{m['duty']:>8.3f}{mark}")


if __name__ == "__main__":
    print("Soft vs. hard actuator saturation, matched limit, closed loop.")
    print("(Synthetic control benchmark -- not a plasma model, and not")
    print(" presented as one. See module docstring.)")
    print("\nAn unlimited actuator is not included as a rival: with these")
    print("disturbances it goes unstable (RMS error 22.3, peak 103.4), which")
    print("is the reason a limit exists, not a baseline to beat.")

    _table("A) actuator limit 1.0, anti-windup ON (the competent baseline)",
           compare(SimConfig(anti_windup=True)))
    _table("B) actuator limit 1.0, anti-windup OFF",
           compare(SimConfig(anti_windup=False)))
    _table("C) actuator limit 0.6 (tight), anti-windup ON",
           compare(SimConfig(limit=0.6, anti_windup=True)))

    print("\n" + "=" * 86)
    print("D) Where is the crossover? RMS error vs. actuator limit, 5 seeds")
    print("=" * 86)
    print(f"  {'limit':>7}{'hard':>11}{'soft_alg':>11}{'soft_tanh':>11}"
          f"{'  winner':>16}")
    for L in (0.4, 0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0):
        r = compare(SimConfig(limit=L, anti_windup=True))
        w = min(r, key=lambda a: r[a]["rms_err"])
        print(f"  {L:>7.1f}{r['hard']['rms_err']:>11.4f}"
              f"{r['soft_algebraic']['rms_err']:>11.4f}"
              f"{r['soft_tanh']['rms_err']:>11.4f}{w:>16}")
    print("\n" + "=" * 86)
    print("E) The fair comparison: each arm at ITS OWN best limit")
    print("=" * 86)
    grid = np.arange(0.3, 3.01, 0.1)
    best = {}
    for a in ("hard", "soft_algebraic", "soft_tanh"):
        curve = [(L, compare(SimConfig(limit=float(L), anti_windup=True),
                             seeds=range(3), arms=[a])[a]["rms_err"])
                 for L in grid]
        L, e = min(curve, key=lambda p: p[1])
        best[a] = (L, e)
        print(f"  {a:16} best limit {L:>4.1f}   RMS err {e:.4f}")
    lo = min(best.values(), key=lambda p: p[1])[1]
    hi = max(best.values(), key=lambda p: p[1])[1]
    print(f"\n  Spread between arms once each is given its own best limit: "
          f"{(hi/lo - 1)*100:.1f} %")
    print("\n  Sensitivity to getting that limit wrong (worst/best RMS over"
          " limits 0.4-3.0):")
    for a in ("hard", "soft_algebraic", "soft_tanh"):
        vals = [compare(SimConfig(limit=float(L), anti_windup=True),
                        seeds=range(3), arms=[a])[a]["rms_err"]
                for L in (0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0)]
        print(f"    {a:16} {min(vals):.4f} .. {max(vals):.4f}   "
              f"({max(vals)/min(vals):.1f}x)")
    print("  This is the number to quote. The large ratios in (D) are mostly")
    print("  a statement about mismatched limits, not about the shape of the")
    print("  nonlinearity -- a hard clip at the wrong limit loses to a soft")
    print("  curve at a good one, which is not the claim anyone should make.")
