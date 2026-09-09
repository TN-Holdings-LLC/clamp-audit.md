# -*- coding: utf-8 -*-
"""Head-to-head: the pasted module vs psd_to_rms_phase_v2, same measurements."""
import numpy as np
from scipy import signal

import psd_original as O
import psd_to_rms_phase_v2 as V

FAIL = 0


def check(ok, what):
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        FAIL += 1


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


rng = np.random.default_rng(0)

# ---------------------------------------------------------------------- 1
sec("1. The previous version's own claims still hold")
FS, N = 2000.0, 65536
t = np.arange(N) / FS
phi = rng.normal(0, 2.0, N)
cfg = O.PhaseNoiseConfig(fmin=1.0, fmax=FS / 2)
old_clamped, _, _ = O._welch_band_rms(
    O.zero_clamp(signal.detrend(phi, type="linear"), 1.0), FS, cfg)
print(f"  blanket zero_clamp on 2.0 rad noise : {old_clamped:.4f} rad "
      f"({100*(old_clamped/2-1):+.1f}%)")
check(abs(old_clamped / 2 - 1 + 0.625) < 0.03, "the -62.5% bias reproduces")
for w, ns, exp in ((11, 5.0, 25.0), (41, 6.0, 0.0)):
    tone = 3.0 * np.sin(2 * np.pi * 50 * t)
    _, m = O.hampel_despike(signal.detrend(tone, type="linear"), w, ns)
    print(f"  Hampel({w},{ns}) false-flags on a 50 Hz tone: {100*m.mean():.1f}%")
    check(abs(100 * m.mean() - exp) < 1.0, f"claimed {exp}% reproduces")

# ---------------------------------------------------------------------- 2
sec("2. fmin is honoured (the previous version's headline number moved 45%)")
FS2, N2 = 1e6, 65536
t2 = np.arange(N2) / FS2
walk = np.cumsum(rng.normal(0, 1.0, N2)) / FS2 ** 0.5 * 300.0
print("  old module, 1/f^2 noise at fs=1 MHz, fmin=1.0 requested:")
for nps in (2 ** 14, 2 ** 16):
    r = O.rms_from_array(t2, walk, O.PhaseNoiseConfig(
        fmin=1.0, fmax=FS2 / 2, nperseg=min(nps, N2), despike=False))
    print(f"    nperseg={nps:<6} df={FS2/min(nps,N2):8.2f} Hz -> {r['rms_rad']:.4f} rad")
print("  v2 refuses to answer a question the record cannot support:")
try:
    V.rms_phase_from_timeseries("", V.PhaseNoiseConfig(fmin=1.0, fmax=FS2 / 2,
                                                       despike=False),
                                _arrays=(t2, walk))
    check(False, "should have raised: 65536 samples at 1 MHz cannot resolve 1 Hz")
except ValueError as e:
    print(f"    ValueError: {str(e)[:96]}...")
    check(True, "raises instead of silently starting the integral at fs/nperseg")

print("  With an achievable band (fmin=100 Hz) it reports what it actually did:")
r = V.rms_phase_from_timeseries("", V.PhaseNoiseConfig(fmin=100.0, fmax=FS2 / 2,
                                                       despike=False),
                                _arrays=(t2, walk))
print(f"    rms={r['rms_rad']:.4f} rad  nperseg={r['nperseg']}  "
      f"df={r['f_resolution']:.3f} Hz  band={r['band_actual']}")
check(r["f_resolution"] <= 100.0, "resolution is fine enough for the stated fmin")

# ---------------------------------------------------------------------- 3
sec("3. Log-spaced PSD integration")
A, f1, f2 = 1.0, 1.0, 1e5
exact = A * (1 / f1 - 1 / f2)
print(f"  S(f)=A/f^2, exact integral over [1, 1e5] = {exact:.6f}")
print(f"  {'pts/decade':>11} {'linear trapz':>16} {'err':>9} {'power-law':>14} {'err':>9}")
worst_new = 0.0
for ppd in (3, 5, 10, 20):
    fg = np.logspace(0, 5, ppd * 5 + 1)
    S = A * fg ** -2.0
    lin = V.integrate_psd(fg, S, f1, f2, loglog=False)
    log = V.integrate_psd(fg, S, f1, f2, loglog=True)
    worst_new = max(worst_new, abs(log / exact - 1))
    print(f"  {ppd:>11} {lin:>16.6f} {100*(lin/exact-1):>8.2f}% "
          f"{log:>14.6f} {100*(log/exact-1):>8.4f}%")
check(worst_new < 1e-6, "power-law integration is exact for a power law")

# a shape that is NOT a pure power law, so the fallback is exercised honestly
fg = np.logspace(0, 5, 51)
S = 1e-3 / fg ** 2 + 1e-9
num = V.integrate_psd(fg, S, 1.0, 1e5, loglog=True)
fine = np.logspace(0, 5, 200001)
ref = np.trapezoid(1e-3 / fine ** 2 + 1e-9, fine)
print(f"  mixed 1/f^2 + white, 10 pts/decade: {num:.8e} vs dense reference "
      f"{ref:.8e} ({100*(num/ref-1):+.3f}%)")
check(abs(num / ref - 1) < 0.02, "stays accurate on a non-power-law shape")

# ---------------------------------------------------------------------- 4
sec("4. dBc/Hz input")
fg = np.logspace(0, 5, 201)
L = -80 - 20 * np.log10(fg)
Slin = 2 * 10 ** (L / 10)
cfgp = V.PhaseNoiseConfig(fmin=1.0, fmax=1e5)
import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore")
    old_bad = O.rms_from_psd_arrays(fg, L, O.PhaseNoiseConfig(fmin=1.0, fmax=1e5))
print(f"  old module, raw dBc/Hz fed in : {old_bad}   (status was 'OK')")
check(np.isnan(old_bad), "old path really did return nan silently")
try:
    V.rms_phase_from_psd("", cfgp, units="rad2/Hz", _arrays=(fg, L))
    check(False, "v2 should refuse all-negative data labelled linear")
except ValueError as e:
    print(f"  v2 units='rad2/Hz'            : ValueError: {str(e)[:70]}...")
    check(True, "v2 refuses instead of returning nan")
rd = V.rms_phase_from_psd("", cfgp, units="dBc/Hz", _arrays=(fg, L))
rl = V.rms_phase_from_psd("", cfgp, units="rad2/Hz", _arrays=(fg, Slin))
print(f"  v2 units='dBc/Hz'             : {rd['rms_rad']:.6e} rad")
print(f"  v2 same data, pre-converted   : {rl['rms_rad']:.6e} rad")
check(abs(rd["rms_rad"] / rl["rms_rad"] - 1) < 1e-12,
      "dBc/Hz conversion matches the hand-converted table")

# ---------------------------------------------------------------------- 5
sec("5. Non-uniform timestamps, broken timestamps, NaN")
phi3 = rng.normal(0, 2.0, 8192)
t3 = np.arange(8192) / 2000.0
tj = np.sort(rng.uniform(0, 8192 / 2000.0, 8192))
c = V.PhaseNoiseConfig(fmin=1.0, fmax=1000.0)
old_j = O.rms_from_array(tj, phi3, O.PhaseNoiseConfig(fmin=1.0, fmax=1000.0))
print(f"  old, jittered t : fs={old_j['fs']:.1f} Hz  rms={old_j['rms_rad']:.4f} rad "
      f"(true 2.0)  status 'OK'")
for label, tt in (("jittered", tj), ("all-zero", np.zeros(8192))):
    try:
        V.rms_phase_from_timeseries("", c, _arrays=(tt, phi3))
        check(False, f"v2 should reject {label} timestamps")
    except ValueError as e:
        print(f"  v2, {label:9} t : ValueError: {str(e)[:64]}...")
        check(True, f"v2 rejects {label} timestamps")
pn = phi3.copy(); pn[123] = np.nan
try:
    V.rms_phase_from_timeseries("", c, _arrays=(t3, pn))
    check(False, "should raise on NaN by default")
except ValueError as e:
    print(f"  v2, one NaN     : {str(e)[:72]}...")
    check("index 123" in str(e), "error names the offending index")
c2 = V.PhaseNoiseConfig(fmin=1.0, fmax=1000.0, on_nonfinite="interpolate")
ri = V.rms_phase_from_timeseries("", c2, _arrays=(t3, pn))
print(f"  v2, interpolate : rms={ri['rms_rad']:.4f} rad  warnings={ri['warnings']}")
check(ri["status"] == "OK_WITH_WARNINGS", "status reflects that something happened")

# ---------------------------------------------------------------------- 6
sec("6. Accuracy on clean data is unchanged (no regression)")
r_old = O.rms_from_array(t, phi, O.PhaseNoiseConfig(fmin=1.0, fmax=FS / 2))
r_new = V.rms_phase_from_timeseries("", V.PhaseNoiseConfig(fmin=1.0, fmax=FS / 2),
                                    _arrays=(t, phi))
print(f"  true 2.0 rad  |  old {r_old['rms_rad']:.4f}  |  v2 {r_new['rms_rad']:.4f} "
      f"(nperseg={r_new['nperseg']}, {r_new['n_outliers_removed']} flagged)")
check(abs(r_new["rms_rad"] - 2.0) < 0.05, "v2 recovers the homogeneous-noise RMS")

spiked = phi.copy()
spiked[rng.choice(N, 20, replace=False)] += 40.0
r_sp = V.rms_phase_from_timeseries("", V.PhaseNoiseConfig(fmin=1.0, fmax=FS / 2),
                                   _arrays=(t, spiked))
print(f"  +20 spikes of 40 rad: rms={r_sp['rms_rad']:.4f} rad, "
      f"{r_sp['n_outliers_removed']} flagged")
check(abs(r_sp["rms_rad"] - 2.0) < 0.15 and r_sp["n_outliers_removed"] >= 20,
      "spikes are caught and the underlying level is preserved")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'SOME CHECKS FAILED'}  "
      f"({FAIL} failure{'' if FAIL == 1 else 's'})")
raise SystemExit(0 if FAIL == 0 else 1)
