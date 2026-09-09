"""The pasted 'corrected' simulator, simulate() verbatim."""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass
class GridParams:
    M: float = 0.70; D: float = 0.8; tau: float = 0.150; T: float = 0.025
@dataclass
class SwarmParams:
    N: int = 100_000; p_device_max_kw: float = 6.0
    sigma: float = 0.012; r_max_gw_s: float = 12.0
@dataclass
class SimParams:
    t_end: float = 20.0; dt: float = 0.001
    disturbance_time: float = 2.0; disturbance_mw: float = 350.0

def simulate(grid, swarm, sim, Kf, use_psf_zero, clip_both=True, rate_limit=True):
    Pmax_total = swarm.N * swarm.p_device_max_kw / 1e6
    delay_steps = max(1, int(round(grid.tau / sim.dt)))
    df_buffer = np.zeros(delay_steps)
    df = 0.0; u = 0.0
    steps = int(sim.t_end / sim.dt) + 1
    t_arr = np.zeros(steps); df_arr = np.zeros(steps); u_arr = np.zeros(steps)
    ucmd_arr = np.zeros(steps)
    for k in range(steps):
        t = k * sim.dt
        dP = sim.disturbance_mw / 1000.0 if t >= sim.disturbance_time else 0.0
        df_delayed = df_buffer[0]
        if use_psf_zero:
            u_cmd = -Kf * (df_delayed / np.sqrt(swarm.sigma**2 + df_delayed**2))
        else:
            u_cmd = -Kf * df_delayed
        ucmd_arr[k] = u_cmd
        if clip_both or not use_psf_zero:
            u_cmd = np.clip(u_cmd, -Pmax_total, Pmax_total)
        if rate_limit:
            du_max = swarm.r_max_gw_s * sim.dt
            u_cmd = np.clip(u_cmd, u - du_max, u + du_max)
        u += (u_cmd - u) * (sim.dt / max(1e-9, grid.T))
        ddf = (-grid.D * df - dP + u) / max(1e-9, grid.M)
        df += ddf * sim.dt
        df_buffer = np.roll(df_buffer, -1); df_buffer[-1] = df
        t_arr[k] = t; df_arr[k] = df; u_arr[k] = u
    nadir = float(np.min(df_arr))
    rocof_max = float(np.max(np.abs(np.diff(df_arr) / sim.dt)))
    tol = 0.01
    after = t_arr >= sim.disturbance_time
    outside = (np.abs(df_arr) > tol) & after
    if np.any(outside):
        li = np.where(outside)[0][-1]
        settling = None if li >= steps - 1 else float(t_arr[li] - sim.disturbance_time)
    else:
        settling = 0.0
    metrics = {
        "Nadir_Hz": round(nadir,4),
        "Max_RoCoF_Hz_s": round(rocof_max,4),
        "Settling_Time_s": (round(settling,2) if settling is not None else None),
        "Oscillation_Std": round(float(np.std(df_arr[int(5/sim.dt):])),4),
        "Final_df": round(float(df_arr[-1]),4),
        "Max_abs_u_GW": round(float(np.max(np.abs(u_arr))),4),
        "Max_abs_ucmd_GW": round(float(np.max(np.abs(ucmd_arr))),4),
        "Pmax_total_GW": Pmax_total,
    }
    return t_arr, df_arr, u_arr, metrics
