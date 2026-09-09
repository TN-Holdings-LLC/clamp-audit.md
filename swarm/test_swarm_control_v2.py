# -*- coding: utf-8 -*-
"""Head-to-head: the pasted simulator vs swarm_control_v2."""
import numpy as np
import swarm_original as O
import swarm_control_v2 as V

FAIL = 0


def check(ok, what):
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        FAIL += 1


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


og, ow, os_ = O.GridParams(), O.SwarmParams(), O.SimParams()
vg, vw, vs = V.GridParams(), V.SwarmParams(), V.SimParams()

# ------------------------------------------------------------------ 1
sec("1. The pasted version's own claims reproduce")
_, _, _, ml = O.simulate(og, ow, os_, 25.0, False)
_, _, _, mp = O.simulate(og, ow, os_, 85.0, True)
print(f"  Legacy   Kf=25: osc={ml['Oscillation_Std']} final={ml['Final_df']} "
      f"nadir={ml['Nadir_Hz']}")
print(f"  PSF-Zero Kf=85: osc={mp['Oscillation_Std']} final={mp['Final_df']} "
      f"nadir={mp['Nadir_Hz']}")
check(abs(ml["Oscillation_Std"] - 0.0666) < 1e-3 and
      abs(mp["Oscillation_Std"] - 0.0755) < 1e-3, "claimed oscillation figures match")
check(ml["Nadir_Hz"] == mp["Nadir_Hz"] and
      ml["Max_RoCoF_Hz_s"] == mp["Max_RoCoF_Hz_s"],
      "nadir and RoCoF identical between the laws, as claimed")
_, _, _, mu = O.simulate(og, ow, os_, 85.0, True, clip_both=False)
print(f"  unclipped PSF-Zero peak output {mu['Max_abs_u_GW']} GW vs "
      f"{mu['Pmax_total_GW']} GW capacity; commanded {mu['Max_abs_ucmd_GW']} GW")
check(abs(mu["Max_abs_u_GW"] - 0.6691) < 1e-3, "the 0.6691 GW overshoot reproduces")

# ------------------------------------------------------------------ 2
sec("2. The rate limiter's effective slew scales with dt (old) or does not (v2)")
print(f"  {'dt [ms]':>8} {'old achieved':>14} {'predicted r*dt/T':>18} {'v2 achieved':>13}")
old_slews, new_slews = [], []
for dt in (2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4):
    _, _, uo, _ = O.simulate(og, ow, O.SimParams(dt=dt), 85.0, True)
    so = np.max(np.abs(np.diff(uo))) / dt
    _, _, _, mv = V.simulate(vg, vw, V.SimParams(dt=dt), 25.0, "smooth")
    sv = mv["Achieved_slew_GW_s"]
    old_slews.append(so); new_slews.append(sv)
    print(f"  {dt*1000:>8.3f} {so:>14.4f} {ow.r_max_gw_s*dt/og.T:>18.4f} {sv:>13.4f}")
check(max(old_slews) / min(old_slews) > 15,
      "old: achieved slew spans 20x across timesteps (should be constant)")
check(max(new_slews) / min(new_slews) < 1.02,
      "v2: achieved slew is timestep-independent")
check(min(new_slews) > 0.9 * vw.r_max_gw_s,
      f"v2 actually reaches the nominal {vw.r_max_gw_s} GW/s (old never did)")

# ------------------------------------------------------------------ 3
sec("3. Timestep convergence")
print(f"  {'dt [ms]':>8} {'old nadir':>11} {'v2 nadir':>11}")
on, vn = [], []
for dt in (2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4):
    _, _, _, a = O.simulate(og, ow, O.SimParams(dt=dt), 25.0, False)
    _, _, _, b = V.simulate(vg, vw, V.SimParams(dt=dt), 25.0, "smooth")
    on.append(a["Nadir_Hz"]); vn.append(b["Nadir_Hz"])
    print(f"  {dt*1000:>8.3f} {a['Nadir_Hz']:>11.4f} {b['Nadir_Hz']:>11.4f}")
so, sn = max(map(abs, on)) / min(map(abs, on)), max(map(abs, vn)) / min(map(abs, vn))
print(f"  magnitude spread: old {100*(so-1):.0f}%   v2 {100*(sn-1):.1f}%")
check(so > 2.0, "old version does not converge (>100% spread in |nadir|)")
check(sn < 1.05, "v2 converges (<5% spread)")

# ------------------------------------------------------------------ 4
sec("4. The two laws in the old file do not share a small-signal gain")
for Kf in (25.0, 85.0):
    print(f"  Kf={Kf:5.1f}: legacy slope={Kf:7.1f}  psf slope=Kf/sigma="
          f"{Kf/ow.sigma:9.1f} GW/Hz  ({1/ow.sigma:.0f}x)")
check(abs(1 / ow.sigma - 83.333) < 0.1,
      "sigma=0.012 is an 83x hidden gain multiplier, not a shape knob")
dP = 0.35
ss_leg = -dP / (og.D + 25.0)
ss_psf = -dP / (og.D + 85.0 / ow.sigma)
print(f"  steady-state offset: legacy Kf=25 {ss_leg:+.5f} Hz | "
      f"psf Kf=85 {ss_psf:+.7f} Hz  ({ss_leg/ss_psf:.0f}x smaller)")
check(abs(ss_leg / ss_psf) > 100,
      "PSF-Zero's near-zero offset follows from loop gain, not geometry")

# ------------------------------------------------------------------ 5
sec("5. v2 matched comparison: is the smooth law uniformly better?")
print(f"  {'K':>7} {'clipped IAE':>13} {'smooth IAE':>12} {'winner':>10} "
      f"{'clip sat%':>10} {'soft sat%':>10}")
wins = {"clipped": 0, "smooth": 0}
for K in (5.0, 10.0, 25.0, 50.0, 100.0, 200.0):
    _, _, _, a = V.simulate(vg, vw, vs, K, "clipped")
    _, _, _, b = V.simulate(vg, vw, vs, K, "smooth")
    w = "clipped" if a["IAE_Hz_s"] < b["IAE_Hz_s"] else "smooth"
    wins[w] += 1
    print(f"  {K:>7.1f} {a['IAE_Hz_s']:>13.4f} {b['IAE_Hz_s']:>12.4f} {w:>10} "
          f"{a['Saturated_fraction']:>9.0%} {b['Saturated_fraction']:>10.0%}")
print(f"  wins: {wins}")
check(wins["clipped"] >= 1 and wins["smooth"] >= 1,
      "there is a genuine crossover -- neither law wins everywhere")

print(f"\n  disturbance sweep at K=50:")
res = []
for mw in (350.0, 500.0, 580.0):
    sp = V.SimParams(disturbance_mw=mw)
    a = V.simulate(vg, vw, sp, 50.0, "clipped")[3]
    b = V.simulate(vg, vw, sp, 50.0, "smooth")[3]
    res.append((mw, a["IAE_Hz_s"], b["IAE_Hz_s"]))
    print(f"    dP={mw:>5.0f} MW: clipped {a['IAE_Hz_s']:.4f} | "
          f"smooth {b['IAE_Hz_s']:.4f}  -> "
          f"{'smooth' if b['IAE_Hz_s'] < a['IAE_Hz_s'] else 'clipped'}")
check(any(b < a for _, a, b in res) and any(b > a for _, a, b in res),
      "the advantage does not extend monotonically toward the ceiling either")

# ------------------------------------------------------------------ 6
sec("6. v2 keeps the fixes the previous version got right")
_, _, _, m = V.simulate(vg, vw, vs, 200.0, "smooth")
print(f"  peak output {m['Max_abs_u_GW']:.4f} GW vs capacity {m['capacity_GW']:.2f} GW")
check(m["Max_abs_u_GW"] <= m["capacity_GW"] * 1.001,
      "capacity limit respected by the saturating law too")
# K=3 is inside the loop's stable region (see section 7); at K=25 nothing
# settles because the loop is linearly unstable, which is a property of the
# plant, not of the metric.
_, _, _, ms = V.simulate(vg, vw, V.SimParams(disturbance_mw=1.0), 3.0, "smooth")
print(f"  stable gain K=3, 1 MW step: settling={ms['Settling_Time_s']}s "
      f"final={ms['Final_df_Hz']:+.6f} Hz")
check(ms["Settling_Time_s"] is not None and ms["Settling_Time_s"] < 5.0,
      "settling time is a real measurement, not always ~0.0")
_, _, _, mb = V.simulate(vg, vw, vs, 25.0, "smooth")
check(mb["Settling_Time_s"] is None,
      "and honestly reports None when it never settles")

# ------------------------------------------------------------------ 7
sec("7. What is actually being compared: regulation or a limit cycle?")
print("  1 MW step (far from any saturation), does it settle?")
print(f"  {'K':>7} {'settling [s]':>13} {'final df':>12} {'osc std':>11}")
stable_max = None
for K in (1.0, 2.0, 3.0, 5.0, 10.0, 25.0):
    m = V.simulate(vg, vw, V.SimParams(disturbance_mw=1.0, t_end=40.0), K, "smooth")[3]
    st = m["Settling_Time_s"]
    if st is not None:
        stable_max = K
    print(f"  {K:>7.1f} {str(st):>13} {m['Final_df_Hz']:>12.6f} "
          f"{m['Oscillation_Std']:>11.6f}")
print(f"  -> the loop is stable up to about K={stable_max:.0f} and limit-cycles above it.")
print("     So every row of section 5 with K>=10 is comparing LIMIT-CYCLE")
print("     amplitudes bounded by the two saturation shapes, not regulation.")
print("     That is a fair comparison, but it is not what 'settling' or")
print("     'oscillation' would normally be taken to mean, and the old file's")
print("     defaults (Kf=25 legacy, effective 7083 for PSF-Zero) sat deep in")
print("     this region for both arms.")
check(stable_max is not None and stable_max <= 5.0,
      "the operating point is identified rather than left implicit")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'SOME CHECKS FAILED'}  "
      f"({FAIL} failure{'' if FAIL == 1 else 's'})")
raise SystemExit(0 if FAIL == 0 else 1)
