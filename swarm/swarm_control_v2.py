# -*- coding: utf-8 -*-
"""
swarm_control_v2.py — saturating vs. clipped droop control in a low-inertia grid
================================================================================

Replaces "PSF-Zero Swarm Control Simulator". The rename is part of the fix:
the previous version's own docstring had already retracted the "legacy
diverges, PSF-Zero saves it" headline, and what is left is a narrow,
worthwhile question — does a SMOOTH saturation beat a HARD clip at the same
gain and the same physical limit? — which the old name oversold.

WHAT THE PREVIOUS VERSION GOT RIGHT.  Everything it claimed reproduces
exactly, and its self-criticism was accurate and unusually honest:

  * the `ro cof_max` SyntaxError is real (the file cannot be imported);
  * unclipped PSF-Zero really did peak at 0.6691 GW against a 0.60 GW fleet
    (+11.5%), while COMMANDING up to 84.76 GW;
  * the old settling-time metric really was always ~0.0 s;
  * and at the pasted defaults neither law diverges — measured Legacy
    osc=0.0666 / final=-0.102 against PSF-Zero osc=0.0755 / final=+0.0935,
    with nadir (-0.1582 Hz) and peak RoCoF (0.5 Hz/s) IDENTICAL between
    them, exactly as it said.

Two things it did not find.

 1. THE RATE LIMITER IS APPLIED TO THE WRONG VARIABLE, SO THE EFFECTIVE SLEW
    LIMIT IS PROPORTIONAL TO THE TIMESTEP.  The old loop does

        u_cmd = clip(u_cmd, u - r_max*dt, u + r_max*dt)
        u    += (u_cmd - u) * (dt/T)

    The clip bounds the *command* to within r_max*dt of u, and then u moves
    only dt/T of the way there, so the largest step u can take is
    r_max*dt * dt/T — an achieved slew rate of r_max*dt/T, not r_max.
    Measured, predicted against actual max|du/dt|, agreeing to 4 decimals:

        dt = 2.00 ms -> 0.96 GW/s      dt = 0.50 ms -> 0.24 GW/s
        dt = 1.00 ms -> 0.48 GW/s      dt = 0.10 ms -> 0.048 GW/s

    The nominal r_max_gw_s = 12 GW/s is never reached at any timestep. At
    the default dt = 1 ms the fleet is throttled to 0.48 GW/s — 25x slower
    than the parameter says.

    This makes the whole simulation UNCONVERGED, so none of its numbers
    mean anything physical. Refining the timestep throttles the controller
    further instead of resolving the dynamics, and the nadir gets
    monotonically worse:

        dt = 2.00 ms -> nadir -0.1313 Hz
        dt = 1.00 ms -> nadir -0.1582 Hz     (the default)
        dt = 0.50 ms -> nadir -0.2072 Hz
        dt = 0.25 ms -> nadir -0.2620 Hz
        dt = 0.10 ms -> nadir -0.3282 Hz     (2.5x the default's value)

    Fixed by rate-limiting a persistent setpoint state and then passing it
    through the inverter lag, so the achieved slew equals r_max at any dt
    and the solution converges as dt shrinks (verified below).

 2. THE TWO CONTROL LAWS DO NOT SHARE A GAIN, SO THE COMPARISON MEASURES
    THE WRONG THING.  With

        legacy    u = -Kf * df                       -> slope at 0 = Kf
        PSF-Zero  u = -Kf * df/sqrt(sigma^2 + df^2)  -> slope at 0 = Kf/sigma

    and sigma = 0.012, PSF-Zero's small-signal loop gain is 83x its Kf.
    At the pasted defaults that is 85/0.012 = 7083 GW/Hz against legacy's
    25 GW/Hz — a factor of 283. Even the writeup's "push Legacy to Kf=85
    too" fix still compares 85 against 7083. `sigma` is not a shape
    parameter in that form; it is a hidden gain multiplier.

    The consequence shows up directly in the steady state (there is no
    integral action, so a step disturbance leaves an offset
    df_ss = -dP/(D + gain)):

        legacy   Kf=25 -> -0.01357 Hz
        legacy   Kf=85 -> -0.00408 Hz
        PSF-Zero Kf=85 -> -0.0000494 Hz

    PSF-Zero's near-zero offset is 283x more loop gain, not geometry.

    Fixed by parameterising both laws the same way — a shared small-signal
    gain K and a shared saturation limit L:

        clipped:  u = clip(-K*df, -L, +L)
        smooth:   u = -K*df / sqrt(1 + (K*df/L)^2)

    Both have slope K at df = 0 and asymptote to L, so they differ ONLY in
    how they bend between the two. That is the comparison worth running,
    and it is the one this file now runs.

WHAT THE FAIR COMPARISON ACTUALLY SHOWS.  With the rate limiter fixed the
solution converges — nadir moves 1.4150 -> 1.4460 (2%) across a 20x range
of timesteps, against the old version's 150% — and the achieved slew is
11.2 GW/s against the nominal 12, independent of dt.

Matched K and matched limit, integrated absolute error after the
disturbance (lower is better):

    K [GW/Hz]   clipped IAE   smooth IAE   clipped sat%   smooth sat%
          5        1.0880       1.2260            0%            0%
         10        0.8366       0.6911           38%            0%
         25        1.4937       1.0631           78%            0%
         50        1.6726       1.4995           90%           43%
        200        1.7225       1.6911           97%           83%

So there is a crossover, and the smooth law is NOT uniformly better. At
K=5 nothing saturates and the smooth curve is 13% WORSE, because it starts
bending well before the limit and so gives away authority it did not need
to. From K=10 up it wins, by 17-29%, and the mechanism is visible in the
last two columns: the hard clip spends 38-97% of the run pinned against
the ceiling, chattering (oscillation std 0.041 vs 0.0008 at K=10), while
the smooth law rarely reaches it at all.

Raising the disturbance toward the ceiling at K=50 does not extend that
advantage monotonically either:

    dP = 350 MW (58% of capacity): clipped 1.6726 | smooth 1.4995
    dP = 500 MW (83%)            : clipped 1.3737 | smooth 0.5746
    dP = 580 MW (97%)            : clipped 0.5478 | smooth 0.5947

smooth is 2.4x better at 500 MW and slightly worse at 580 MW. Note also
that IAE FALLS as the disturbance grows for the clipped law — the error is
dominated by limit-cycle oscillation, not by the steady-state offset, and
a large enough disturbance parks the actuator in saturation and stops the
cycle. Any claim of the form "this control law is better" needs the
operating point attached to it.

AND THE OPERATING POINT IS FURTHER FROM "REGULATION" THAN ANY OF THIS
SUGGESTS.  With a 1 MW step, far below anything that saturates, the loop
settles for K <= 5 and does not for K >= 10:

    K =  1, 2, 3, 5  -> settles, oscillation std 0.000000
    K = 10           -> never settles, osc 0.0489
    K = 25           -> never settles, osc 0.0729

With a 150 ms delay this loop is simply unstable above K ~ 5. So every row
of the comparison at K >= 10 is measuring the amplitude of a LIMIT CYCLE
that the saturation is bounding — a legitimate thing to compare between two
saturation shapes, and arguably the thing that matters here, but not what
"settling time" or "oscillation" would ordinarily be taken to mean. The
previous file's defaults (legacy Kf=25, PSF-Zero effectively 7083) sat deep
inside this region for both arms, which is why neither ever settled and why
its "oscillation_std" numbers were limit-cycle amplitudes rather than
convergence rates.

KEPT FROM THE PREVIOUS VERSION: the capacity clip applied to both branches,
the corrected settling-time definition, and plot labels that state what was
configured rather than asserting an outcome. Plotting now uses the Agg
backend and never calls plt.show(), so the file runs headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["GridParams", "SwarmParams", "SimParams", "simulate", "dt_convergence"]


@dataclass
class GridParams:
    M: float = 0.70      # [GW*s/Hz] inertia constant (very low)
    D: float = 0.8       # [GW/Hz] load damping
    tau: float = 0.150   # [s] communication delay
    T: float = 0.025     # [s] inverter time constant


@dataclass
class SwarmParams:
    N: int = 100_000
    p_device_max_kw: float = 6.0
    r_max_gw_s: float = 12.0     # physical slew limit, now actually achieved

    @property
    def capacity_gw(self) -> float:
        return self.N * self.p_device_max_kw / 1e6


@dataclass
class SimParams:
    t_end: float = 20.0
    dt: float = 0.001
    disturbance_time: float = 2.0
    disturbance_mw: float = 350.0


def _control(df: float, K: float, L: float, law: str) -> float:
    """Both laws: slope -K at df=0, magnitude bounded by L. Nothing else differs."""
    if law == "clipped":
        return float(np.clip(-K * df, -L, L))
    if law == "smooth":
        x = K * df
        return float(-x / np.sqrt(1.0 + (x / L) ** 2))
    if law == "none":
        return 0.0
    raise ValueError(f"unknown law {law!r}")


def simulate(grid: GridParams, swarm: SwarmParams, sim: SimParams,
             K: float, law: str = "smooth",
             limit_gw: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, Dict]:
    """Simulate the delayed droop loop.

    K       small-signal gain [GW/Hz], identical in meaning for every law
    law     "clipped" | "smooth" | "none"
    limit_gw saturation limit; defaults to the fleet's physical capacity
    """
    L = swarm.capacity_gw if limit_gw is None else float(limit_gw)
    dt = sim.dt
    delay_steps = max(1, int(round(grid.tau / dt)))
    steps = int(round(sim.t_end / dt)) + 1

    df_hist = np.zeros(delay_steps)   # circular buffer, no per-step reallocation
    head = 0
    df = 0.0
    u = 0.0        # actuator output after the inverter lag
    u_set = 0.0    # rate-limited setpoint (the state the slew limit acts on)

    t_arr = np.empty(steps)
    df_arr = np.empty(steps)
    u_arr = np.empty(steps)
    sat_arr = np.zeros(steps, dtype=bool)

    du_max = swarm.r_max_gw_s * dt   # now bounds the setpoint itself
    for k in range(steps):
        t = k * dt
        dP = sim.disturbance_mw / 1000.0 if t >= sim.disturbance_time else 0.0

        df_delayed = df_hist[head]
        u_cmd = _control(df_delayed, K, L, law)
        u_cmd = float(np.clip(u_cmd, -L, L))          # physical capacity, both laws
        sat_arr[k] = abs(u_cmd) > 0.99 * L

        # True slew limit: bound the SETPOINT's own step, then lag toward it.
        # The old code bounded u_cmd relative to u and then moved only dt/T of
        # the way, which made the achieved rate r_max*dt/T.
        u_set += float(np.clip(u_cmd - u_set, -du_max, du_max))
        u += (u_set - u) * (dt / max(1e-9, grid.T))

        ddf = (-grid.D * df - dP + u) / max(1e-9, grid.M)
        df += ddf * dt

        df_hist[head] = df
        head = (head + 1) % delay_steps

        t_arr[k] = t
        df_arr[k] = df
        u_arr[k] = u

    tol = 0.01
    after = t_arr >= sim.disturbance_time
    outside = (np.abs(df_arr) > tol) & after
    if np.any(outside):
        li = int(np.flatnonzero(outside)[-1])
        settling = None if li >= steps - 1 else float(t_arr[li] - sim.disturbance_time)
    else:
        settling = 0.0

    post = df_arr[t_arr >= 5.0]
    metrics = {
        "law": law,
        "K_gw_per_hz": K,
        "limit_gw": L,
        "Nadir_Hz": round(float(np.min(df_arr)), 4),
        "Max_RoCoF_Hz_s": round(float(np.max(np.abs(np.diff(df_arr) / dt))), 4),
        "Settling_Time_s": (round(settling, 3) if settling is not None else None),
        "Oscillation_Std": round(float(np.std(post)), 5),
        "Final_df_Hz": round(float(df_arr[-1]), 5),
        # Integrated absolute error after the disturbance: one number that does
        # not depend on whether a tolerance band happens to be crossed.
        "IAE_Hz_s": round(float(np.abs(df_arr[after]).sum() * dt), 4),
        "Max_abs_u_GW": round(float(np.max(np.abs(u_arr))), 4),
        "Achieved_slew_GW_s": round(float(np.max(np.abs(np.diff(u_arr))) / dt), 4),
        "Saturated_fraction": round(float(sat_arr[after].mean()), 4),
        "capacity_GW": L,
    }
    return t_arr, df_arr, u_arr, metrics


def dt_convergence(grid: GridParams, swarm: SwarmParams, sim: SimParams,
                   K: float, law: str, dts=(2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4)) -> Dict:
    """Re-run at several timesteps. If these disagree, no other number here
    means anything -- which was the previous version's actual situation."""
    out = {}
    for dt in dts:
        sp = SimParams(t_end=sim.t_end, dt=dt,
                       disturbance_time=sim.disturbance_time,
                       disturbance_mw=sim.disturbance_mw)
        _, _, _, m = simulate(grid, swarm, sp, K, law)
        out[dt] = m
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")          # headless: the old file called plt.show()
    import matplotlib.pyplot as plt

    grid, swarm, sim = GridParams(), SwarmParams(), SimParams()
    L = swarm.capacity_gw
    print("Saturating vs. clipped droop control, matched gain and limit")
    print("=" * 78)
    print(f"fleet capacity {L:.2f} GW | delay {grid.tau*1000:.0f} ms | "
          f"M={grid.M} GW*s/Hz | disturbance +{sim.disturbance_mw:.0f} MW")

    print("\n--- timestep convergence (the old version had none) ---")
    print(f"  {'dt [ms]':>8} {'nadir':>10} {'IAE':>9} {'osc':>9} {'achieved slew':>15}")
    for dt, m in dt_convergence(grid, swarm, sim, 25.0, "smooth").items():
        print(f"  {dt*1000:>8.3f} {m['Nadir_Hz']:>10.4f} {m['IAE_Hz_s']:>9.4f} "
              f"{m['Oscillation_Std']:>9.5f} {m['Achieved_slew_GW_s']:>15.3f}")

    print("\n--- matched comparison: same K, same limit, only the bend differs ---")
    print(f"  {'K [GW/Hz]':>10} {'law':>9} {'nadir':>9} {'IAE':>9} {'osc':>9} "
          f"{'final df':>10} {'sat frac':>9}")
    for K in (5.0, 10.0, 25.0, 50.0, 100.0, 200.0):
        for law in ("clipped", "smooth"):
            _, _, _, m = simulate(grid, swarm, sim, K, law)
            print(f"  {K:>10.1f} {law:>9} {m['Nadir_Hz']:>9.4f} {m['IAE_Hz_s']:>9.4f} "
                  f"{m['Oscillation_Std']:>9.5f} {m['Final_df_Hz']:>10.5f} "
                  f"{m['Saturated_fraction']:>9.2%}")

    print("\n--- where saturation is actually exercised: bigger disturbance ---")
    for mw in (350.0, 500.0, 580.0):
        sp = SimParams(disturbance_mw=mw)
        row = []
        for law in ("clipped", "smooth"):
            _, _, _, m = simulate(grid, swarm, sp, 50.0, law)
            row.append(m)
        print(f"  dP={mw:>5.0f} MW ({mw/1000/L:>4.0%} of capacity): "
              f"clipped IAE={row[0]['IAE_Hz_s']:.4f} sat={row[0]['Saturated_fraction']:.0%} | "
              f"smooth IAE={row[1]['IAE_Hz_s']:.4f} sat={row[1]['Saturated_fraction']:.0%}")

    t1, d1, u1, m1 = simulate(grid, swarm, sim, 50.0, "clipped")
    t2, d2, u2, m2 = simulate(grid, swarm, sim, 50.0, "smooth")
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t1, d1, color="#c0392b", lw=1.6, label="hard clip (K=50)")
    axs[0].plot(t2, d2, color="#2471a3", lw=2.0, label="smooth saturation (K=50)")
    axs[0].axvline(sim.disturbance_time, color="k", ls="--", alpha=.6,
                   label=f"+{sim.disturbance_mw:.0f} MW")
    axs[0].axhline(0, color="k", lw=.6)
    axs[0].set_ylabel(r"$\Delta f$ [Hz]")
    axs[0].set_title("Matched gain and matched limit; the laws differ only in shape\n"
                     "(synthetic model — not a validated grid study)")
    axs[0].legend(loc="lower right"); axs[0].grid(alpha=.3)
    axs[1].plot(t1, u1, color="#c0392b", lw=1.6, label="hard clip")
    axs[1].plot(t2, u2, color="#2471a3", lw=2.0, label="smooth saturation")
    for s in (1, -1):
        axs[1].axhline(s * L, color="grey", ls=":", lw=1.2)
    axs[1].axhline(L, color="grey", ls=":", lw=1.2, label="fleet capacity")
    axs[1].set_ylabel("swarm output u [GW]"); axs[1].set_xlabel("time [s]")
    axs[1].legend(loc="lower right"); axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("swarm_control_v2.png", dpi=150, bbox_inches="tight")
    print("\nPlot written to swarm_control_v2.png")
