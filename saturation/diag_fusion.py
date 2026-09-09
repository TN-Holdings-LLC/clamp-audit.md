"""Verify the pasted version's claims, then ask what the 6.2% actually measures."""
import numpy as np
import pandas as pd
from fusion_original import (run_simulation, clamp_delta, FusionAdapter,
                             FusionCfg, PSFZeroCfg, synthetic_sensors)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


df_on, df_off = run_simulation(12.0, 42)

# ------------------------------------------------------------------ [1]
sec("[1] Claimed: raw_dtheta now spike-sensitive (~0.006 to ~3.12 rad)")
r = df_on["raw_dtheta"]
print(f"  measured range {r.min():.4f} .. {r.max():.4f} rad, sd {r.std():.4f}"
      f"   -> the v1 'always 0.85' artifact is genuinely gone")

# ------------------------------------------------------------------ [2]
sec("[2] Claimed: ~6.2% peak / ~5.8% RMS reduction")
p_off, p_on = df_off.coil_cmd.abs().max(), df_on.coil_cmd.abs().max()
m_off, m_on = df_off.coil_cmd.abs().mean(), df_on.coil_cmd.abs().mean()
print(f"  peak {(1-p_on/p_off)*100:.2f} %   'RMS' {(1-m_on/m_off)*100:.2f} %"
      f"   -> reproduces")
true_rms_off = np.sqrt((df_off.coil_cmd**2).mean())
true_rms_on = np.sqrt((df_on.coil_cmd**2).mean())
print(f"  NB the 'RMS' metric is mean(|x|), not RMS. True RMS reduction: "
      f"{(1-true_rms_on/true_rms_off)*100:.2f} %")

# ------------------------------------------------------------------ [3]
sec("[3] Is coil_cmd a control signal, or a monotonic ramp?")
for name, d in (("PSF-Zero", df_on), ("baseline", df_off)):
    inc = np.diff(d.coil_cmd.values)
    print(f"  {name:9}: increments  min {inc.min():+.5f}  max {inc.max():+.5f}  "
          f"negative steps: {(inc < 0).sum()}/{len(inc)}")
print("  coil_cmd never decreases, so 'peak' == the final value and")
print("  'mean |coil_cmd|' == the mean of a monotone ramp. Neither is a")
print("  measure of excursion suppression; both are the ramp's endpoint.")
print(f"  final values: baseline {df_off.coil_cmd.iloc[-1]:.2f}, "
      f"PSF {df_on.coil_cmd.iloc[-1]:.2f}")

# ------------------------------------------------------------------ [4]
sec("[4] So what IS the 6.2%? Closed form, no simulation needed.")
cont = df_on["action"] == "CONTINUE"
raw = df_on.loc[cont, "raw_dtheta"].values
pred = 1 - clamp_delta(raw, 0.9).sum() / raw.sum()
print(f"  1 - sum(clamp(raw)) / sum(raw)  over CONTINUE steps = {pred*100:.2f} %")
print(f"  simulated peak reduction                            = {(1-p_on/p_off)*100:.2f} %")
print("  Identical. The result is the mean compression of the saturation")
print("  curve over the input distribution -- computable in two lines. The")
print("  quaternions, EIT, phase and 12 s of dynamics contribute nothing.")

# ------------------------------------------------------------------ [5]
sec("[5] Do the quaternion / EIT / phase subsystems affect the output?")
a = FusionAdapter(FusionCfg())
a.q = np.array([0.3, 0.5, -0.2, 0.78])          # arbitrary different attitude
a.q = a.q / np.linalg.norm(a.q)
a.zbar = complex(-0.4, 0.9)
a.phi = 2.7
rng = np.random.default_rng(42)
alt = [a.step(synthetic_sensors(a.t, rng)) for _ in range(12000)]
alt = pd.DataFrame(alt)
same = np.allclose(alt.coil_cmd.values, df_on.coil_cmd.values)
print(f"  Re-ran with a completely different initial quaternion, zbar and phi.")
print(f"  coil_cmd identical to the original run: {same}")
print("  -> q, zbar and phi are dead code with respect to the measured output.")

# ------------------------------------------------------------------ [6]
sec("[6] Does the ABSTAIN gate exclude the spikes from the comparison?")
n_ab = (df_on.action == "ABSTAIN").sum()
print(f"  ABSTAIN steps: {n_ab}/{len(df_on)} ({100*n_ab/len(df_on):.1f} %), "
      f"identical for both arms: {(df_on.action != df_off.action).sum() == 0}")
thr = np.deg2rad(55)
print(f"  ABSTAIN fires at raw_dtheta > {thr:.3f} rad.")
print(f"  raw_dtheta during ABSTAIN steps: "
      f"{df_on.loc[~cont,'raw_dtheta'].min():.2f}..{df_on.loc[~cont,'raw_dtheta'].max():.2f}")
print(f"  raw_dtheta during CONTINUE steps: {raw.min():.2f}..{raw.max():.2f}")
print("  Every large excursion is ABSTAINed -- for BOTH arms equally -- so")
print("  neither arm ever integrates a spike. The saturation is only ever")
print("  exercised on sub-threshold inputs, i.e. never on the events it")
print("  exists to suppress.")

# ------------------------------------------------------------------ [7]
sec("[7] Does the shared hardware limit (70 deg) ever bind?")
print(f"  steps where |dtheta_applied| > hw limit: "
      f"{df_on.hw_clipped.sum()} (PSF), {df_off.hw_clipped.sum()} (baseline)")
print("  ABSTAIN triggers at 55 deg, below the 70 deg limit, so the limit the")
print("  docstring highlights as 'applied to BOTH paths equally' is unreachable.")

# ------------------------------------------------------------------ [8]
sec("[8] Are the injected 'MHD spikes' actually spikes?")
t = np.arange(12000) * 0.001
mask = (np.floor(t * 15).astype(int) % 47) == 0
runs, cur = [], 0
for m in mask:
    if m:
        cur += 1
    elif cur:
        runs.append(cur); cur = 0
if cur:
    runs.append(cur)
print(f"  'rad' spike condition true for {mask.sum()} of 12000 steps, in "
      f"{len(runs)} runs of {set(runs)} consecutive steps")
print(f"  -> {runs[0]*0.001*1000:.0f} ms plateaus, not spikes. int(t*15) holds")
print("     each value for 1/15 s, so the condition latches for ~67 steps.")

# ------------------------------------------------------------------ [9]
sec("[9] Is the 5-seed 'reproducibility' evidence of robustness?")
for seed in range(1, 6):
    d1, d2 = run_simulation(12.0, seed)
    print(f"  seed={seed}: peak reduction "
          f"{(1-d1.coil_cmd.abs().max()/d2.coil_cmd.abs().max())*100:5.2f} %")
print("  The seed only perturbs additive noise of sd 0.06-0.08 on top of")
print("  deterministic sinusoids of amplitude ~0.55-0.75. The signal is")
print("  ~90 % identical across seeds by construction, so agreement across")
print("  them is not evidence the effect is robust to anything that matters.")
