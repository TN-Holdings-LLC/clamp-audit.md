"""Verify the pasted version's own claims, then probe where it breaks."""
import numpy as np
from eit_original import EITDetector, EITAccumulator, psf_zero_synchrony

OK, BAD = "as claimed", "DIFFERS"


def sec(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def noise(seed, T=8000, K=12):
    r = np.random.default_rng(seed)
    return r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))


def with_burst(seed, K=12, T=8000, amp=0.8, center=4.2, chans=None, fs=1000.0):
    r = np.random.default_rng(seed)
    t = np.arange(T) / fs
    z = r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))
    b = np.exp(1j * 2 * np.pi * 45 * t) * np.exp(-((t - center) / 0.4) ** 2)
    sel = range(K) if chans is None else chans
    for k in sel:
        z[:, k] += amp * b * (1 + 0.3 * r.standard_normal(T))
    return z


# ------------------------------------------------------------------ [1]
sec("[1] Claimed: pure-noise PLV baseline 0.068 +/- 0.0004")
vals = []
for s in range(30):
    d = EITDetector()
    _, sync = psf_zero_synchrony(d.eit.filter(noise(s)), win=d.sync_win,
                                 step=d.sync_step, tau=d.sync_tau)
    vals.append(sync.mean())
vals = np.array(vals)
print(f"  measured baseline: {vals.mean():.4f} +/- {vals.std():.4f}   "
      f"({OK if abs(vals.mean()-0.068) < 0.01 else BAD})")

# ------------------------------------------------------------------ [2]
sec("[2] Claimed: 0 false alarms on pure noise / burst detected ~3.8-4.2s")
fa = sum(EITDetector().detect(noise(s))["alarm_time"] is not None for s in range(30))
print(f"  false alarms, 30 pure-noise trials: {fa}/30")
ts = [EITDetector().detect(with_burst(s))["alarm_time"] for s in range(10)]
hit = [x for x in ts if x is not None]
print(f"  burst detected in {len(hit)}/10 trials"
      + (f", alarm times {min(hit):.2f}-{max(hit):.2f}s" if hit else ""))

# ------------------------------------------------------------------ [3]
sec("[3] Is the calibration tied to sync_win? (kappa/eta are constants)")
print(f"  {'win':>6} {'noise baseline':>16} {'kappa':>8} {'false alarms':>14} "
      f"{'burst found':>13}")
for win in (128, 192, 256, 384, 512, 1024):
    d = EITDetector(sync_win=win)
    base = np.mean([psf_zero_synchrony(d.eit.filter(noise(s)), win=win,
                                       step=d.sync_step, tau=d.sync_tau)[1].mean()
                    for s in range(6)])
    f = sum(EITDetector(sync_win=win).detect(noise(s))["alarm_time"] is not None
            for s in range(6))
    h = sum(EITDetector(sync_win=win).detect(with_burst(s))["alarm_time"] is not None
            for s in range(6))
    print(f"  {win:>6} {base:>16.4f} {d.cusum_kappa:>8.3f} {f:>10}/6     {h:>9}/6")

# ------------------------------------------------------------------ [4]
sec("[4] Is it tied to eit_alpha?")
print(f"  {'alpha':>7} {'noise baseline':>16} {'false alarms':>14} {'burst found':>13}")
for a in (0.2, 0.4, 0.75, 1.5, 3.0):
    d = EITDetector(eit_alpha=a)
    base = np.mean([psf_zero_synchrony(d.eit.filter(noise(s)), win=d.sync_win,
                                       step=d.sync_step, tau=d.sync_tau)[1].mean()
                    for s in range(6)])
    f = sum(EITDetector(eit_alpha=a).detect(noise(s))["alarm_time"] is not None
            for s in range(6))
    h = sum(EITDetector(eit_alpha=a).detect(with_burst(s))["alarm_time"] is not None
            for s in range(6))
    print(f"  {a:>7} {base:>16.4f} {f:>10}/6     {h:>9}/6")

# ------------------------------------------------------------------ [5]
sec("[5] Partial synchrony: only some of the 12 channels lock together")
print("  (the demo synchronises ALL 12, which is the easiest possible case)")
print(f"  {'channels locked':>17} {'max synchrony':>15} {'detected':>10}")
for n in (12, 8, 6, 4, 3, 2):
    res = [EITDetector().detect(with_burst(s, chans=range(n))) for s in range(6)]
    det = sum(r["alarm_time"] is not None for r in res)
    print(f"  {n:>10} of 12 {np.mean([r['max_synchrony'] for r in res]):>15.4f} "
          f"{det:>8}/6")

# ------------------------------------------------------------------ [6]
sec("[6] Real-valued input (the common case for sensor/EEG data)")
r = np.random.default_rng(0)
zr = r.standard_normal((8000, 12))
try:
    res = EITDetector().detect(zr.astype(float))
    print(f"  runs silently. max synchrony = {res['max_synchrony']:.4f}, "
          f"alarm = {res['alarm_time']}")
    print("  np.angle() of a real signal is only ever 0 or pi, so 'phase")
    print("  synchrony' here is measuring sign agreement, not phase. No warning.")
except Exception as e:
    print(f"  raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ [7]
sec("[7] Reported alarm time vs. when the alarm could actually be known")
d = EITDetector()
res = d.detect(with_burst(0))
if res["alarm_index"] is not None:
    i = res["alarm_index"]
    centre = res["times"][i] / d.fs
    end = (res["times"][i] + d.sync_win // 2) / d.fs
    print(f"  reported alarm_time : {centre:.3f} s  (window CENTRE)")
    print(f"  earliest knowable   : {end:.3f} s  (window END - all {d.sync_win}")
    print(f"                        samples must be in hand to compute the PLV)")
    print(f"  latency understated by {(end-centre)*1000:.0f} ms")

# ------------------------------------------------------------------ [8]
sec("[8] Cost of the EIT Python loop")
import time
for T in (8000, 32000):
    z = noise(0, T=T)
    e = EITAccumulator(alpha=0.75)
    t0 = time.perf_counter()
    e.filter(z)
    print(f"  T={T:>6}: {(time.perf_counter()-t0)*1e3:>8.1f} ms "
          f"({T} sequential Python iterations)")
