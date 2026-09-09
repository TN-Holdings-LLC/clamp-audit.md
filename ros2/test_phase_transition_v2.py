# -*- coding: utf-8 -*-
"""Verifies phase_transition_proof_v2.py against the pasted behaviour."""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd
import phase_transition_proof_v2 as V

FAIL = 0
def check(ok, what):
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok: FAIL += 1
def sec(t): print("\n"+"="*76+"\n"+t+"\n"+"="*76)

true = lambda x: 10.0*np.exp(-1.3*x)

def write(path, t, phi_eit, extra_nan_bools=False):
    n = len(t)
    df = pd.DataFrame({
        "t": 1000.0 + np.asarray(t, float),
        "phi_inj_ms": true(np.asarray(t, float)) + 0.1,
        "phi_eit_ms": phi_eit,
        "TNow_event": np.zeros(n),
        "settled": np.ones(n),
    })
    if extra_nan_bools:
        df.loc[1, "TNow_event"] = np.nan
        df.loc[2, "settled"] = np.nan
    df.to_csv(path, index=False)
    return path

tmp = tempfile.mkdtemp()

# ---------------------------------------------------------------- 1
sec("1. The interpolation fix reproduces (and does not regress on even grids)")
t = np.array([0.0, 1.0, 2.1, 5.1, 5.2, 6.0]); v = true(t).copy(); v[3] = np.nan
d = pd.DataFrame({"t_rel": t, "phi": v})
pos = d["phi"].interpolate().iloc[3]
idx = d.set_index("t_rel")["phi"].interpolate(method="index").iloc[3]
print(f"  position {pos:.6f} | time-based {idx:.6f} | truth {true(5.1):.6f}")
check(abs(idx-true(5.1)) < abs(pos-true(5.1)),
      "time-based interpolation is closer to the truth on an uneven gap")
te = np.arange(0,6,0.1); ve = true(te).copy(); ve[20]=np.nan
de = pd.DataFrame({"t_rel": te, "phi": ve})
check(np.isclose(de["phi"].interpolate().iloc[20],
                 de.set_index("t_rel")["phi"].interpolate(method="index").iloc[20],
                 atol=1e-12), "identical on an even grid (no regression)")

# ---------------------------------------------------------------- 2
sec("2. Trailing gap is left as a gap, not carried forward")
t2 = np.array([0.,1.,2.,3.,4.]); v2 = true(t2).copy(); v2[0]=np.nan; v2[-1]=np.nan
old = pd.Series(v2, index=t2).interpolate(method="index")
df2, info2 = V.prepare(pd.read_csv(write(os.path.join(tmp,"a.csv"), t2, v2)))
print(f"  pasted behaviour : {np.array2string(old.to_numpy(), precision=4)}")
print(f"  v2 behaviour     : {np.array2string(df2['phi_eit_interp'].to_numpy(), precision=4)}")
check(not np.isnan(old.iloc[-1]), "pasted version really does carry the last value forward")
check(np.isnan(df2["phi_eit_interp"].iloc[-1]),
      "v2 leaves the trailing gap as NaN so the plot shows a break")
check(np.isnan(df2["phi_eit_interp"].iloc[0]), "leading gap also left as a gap")
check(info2["nan_out"] == 2 and info2["interpolated"] == 0,
      "the counts reported to the user match what happened")

# ---------------------------------------------------------------- 3
sec("3. Duplicate timestamps (dup_prob in the injector creates these)")
t3 = np.array([0.,1.,2.,2.,3.,4.]); v3 = true(t3).copy(); v3[3]=np.nan
df3, info3 = V.prepare(pd.read_csv(write(os.path.join(tmp,"b.csv"), t3, v3)))
print(f"  rows in 6 -> out {len(df3)}, collapsed {info3['duplicates_collapsed']}")
check(info3["duplicates_collapsed"] == 1, "the duplicate timestamp is collapsed")
check(df3["t_rel"].is_unique, "the resulting time index is unique")
check(df3["t_rel"].is_monotonic_increasing, "and monotonically increasing")

# ---------------------------------------------------------------- 4
sec("4. Missing columns fail up front with a useful message")
bad = os.path.join(tmp, "bad.csv")
pd.DataFrame({"t":[0,1,2], "phi_eit_ms":[1.,2.,3.]}).to_csv(bad, index=False)
r = subprocess.run([sys.executable, "phase_transition_proof_v2.py", "--csv", bad],
                   capture_output=True, text=True)
print(f"  exit {r.returncode}: {(r.stdout+r.stderr).strip().splitlines()[0][:72]}")
check(r.returncode != 0, "non-zero exit instead of a mid-plot KeyError")
check("phi_inj_ms" in (r.stdout+r.stderr), "the message names the missing column")

# ---------------------------------------------------------------- 5
sec("5. Missing booleans are not silently turned into False")
t5 = np.arange(6.0); v5 = true(t5)
p5 = write(os.path.join(tmp,"c.csv"), t5, v5, extra_nan_bools=True)
raw = pd.read_csv(p5)
print(f"  input has {int(raw['TNow_event'].isna().sum())} NaN in TNow_event, "
      f"{int(raw['settled'].isna().sum())} in settled")
check(raw["settled"].fillna(0).iloc[2] == 0.0,
      "the pasted .fillna(0) would render a dropped sample as 'not settled'")
d5, _ = V.prepare(raw)
check(bool(d5["settled"].isna().iloc[2]), "v2 keeps it missing rather than False")

# ---------------------------------------------------------------- 6
sec("6. End-to-end run is headless and reports what it reconstructed")
t6 = np.array([0.,1.,2.1,5.1,5.2,6.0]); v6 = true(t6).copy(); v6[3]=np.nan
p6 = write(os.path.join(tmp,"d.csv"), t6, v6)
png = os.path.join(tmp, "out.png")
env = dict(os.environ); env.pop("DISPLAY", None)
r = subprocess.run([sys.executable, "phase_transition_proof_v2.py",
                    "--csv", p6, "--out", png],
                   capture_output=True, text=True, env=env, timeout=180)
print("  " + "\n  ".join(r.stdout.strip().splitlines()))
check(r.returncode == 0, "runs to completion with no display")
check(os.path.exists(png) and os.path.getsize(png) > 10000, "a PNG was written")
check("interpolated: 1" in r.stdout, "reports how many samples were reconstructed")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'SOME CHECKS FAILED'}  "
      f"({FAIL} failure{'' if FAIL==1 else 's'})")
raise SystemExit(0 if FAIL==0 else 1)
