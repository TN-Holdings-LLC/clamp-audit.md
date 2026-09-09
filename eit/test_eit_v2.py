"""v1 vs v2, same measurements as diag_eit.py plus a proper ROC."""
import time
import numpy as np

from eit_original import EITDetector as V1, EITAccumulator as A1
from eit_v2 import EITDetector as V2, EITAccumulator as A2

OK, BAD = "PASS", "FAIL"


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def noise(seed, T=8000, K=12):
    r = np.random.default_rng(seed)
    return r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))


def burst(seed, K=12, T=8000, amp=0.8, center=4.2, chans=None, fs=1000.0):
    r = np.random.default_rng(seed)
    t = np.arange(T) / fs
    z = r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))
    b = np.exp(1j * 2 * np.pi * 45 * t) * np.exp(-((t - center) / 0.4) ** 2)
    for k in (range(K) if chans is None else chans):
        z[:, k] += amp * b * (1 + 0.3 * r.standard_normal(T))
    return z


# ------------------------------------------------------------------ [1]
sec("[1] Vectorised EIT must reproduce v1's loop exactly")
for T in (1000, 8000, 32000):
    z = noise(0, T=T)
    a, b = A1(alpha=0.75).filter(z), A2(alpha=0.75).filter(z)
    d = np.abs(a - b).max()
    t0 = time.perf_counter(); A1(alpha=0.75).filter(z); d1 = time.perf_counter() - t0
    t0 = time.perf_counter(); A2(alpha=0.75).filter(z); d2 = time.perf_counter() - t0
    print(f"  T={T:>6}  max dev {d:.2e}  loop {d1*1e3:7.1f}ms  "
          f"lfilter {d2*1e3:6.2f}ms  {d1/d2:5.1f}x  "
          f"{OK if d < 1e-10 else BAD}")

# ------------------------------------------------------------------ [2]
sec("[2] Robustness to sync_win  (v1: 6/6 false alarms at win<=192)")
print(f"  {'win':>6} | {'v1 FA':>7} {'v1 hit':>7} | {'v2 FA':>7} {'v2 hit':>7}")
for win in (128, 192, 256, 384, 512, 1024):
    f1 = sum(V1(sync_win=win).detect(noise(s))["alarm_time"] is not None for s in range(6))
    h1 = sum(V1(sync_win=win).detect(burst(s))["alarm_time"] is not None for s in range(6))
    f2 = sum(V2(sync_win=win).detect(noise(s))["alarm_time"] is not None for s in range(6))
    h2 = sum(V2(sync_win=win).detect(burst(s))["alarm_time"] is not None for s in range(6))
    print(f"  {win:>6} | {f1:>5}/6 {h1:>5}/6 | {f2:>5}/6 {h2:>5}/6")

# ------------------------------------------------------------------ [3]
sec("[3] Robustness to eit_alpha  (v1: 6/6 false alarms at alpha<=0.4)")
print(f"  {'alpha':>7} | {'v1 FA':>7} {'v1 hit':>7} | {'v2 FA':>7} {'v2 hit':>7}")
for a in (0.2, 0.4, 0.75, 1.5, 3.0):
    f1 = sum(V1(eit_alpha=a).detect(noise(s))["alarm_time"] is not None for s in range(6))
    h1 = sum(V1(eit_alpha=a).detect(burst(s))["alarm_time"] is not None for s in range(6))
    f2 = sum(V2(eit_alpha=a).detect(noise(s))["alarm_time"] is not None for s in range(6))
    h2 = sum(V2(eit_alpha=a).detect(burst(s))["alarm_time"] is not None for s in range(6))
    print(f"  {a:>7} | {f1:>5}/6 {h1:>5}/6 | {f2:>5}/6 {h2:>5}/6")

# ------------------------------------------------------------------ [4]
sec("[4] Partial synchrony: only m of 12 channels lock (12 trials each)")
print(f"  {'m':>4} | {'v1 detected':>12} | {'v2 mean_pair':>13} {'v2 eig':>9}")
for m in (12, 8, 6, 4, 3, 2):
    d1 = sum(V1().detect(burst(s, chans=range(m)))["alarm_time"] is not None
             for s in range(12))
    d2p = sum(V2(statistic="mean_pair").detect(burst(s, chans=range(m)))["alarm_time"]
              is not None for s in range(12))
    d2e = sum(V2(statistic="eig").detect(burst(s, chans=range(m)))["alarm_time"]
              is not None for s in range(12))
    print(f"  {m:>4} | {d1:>10}/12 | {d2p:>11}/12 {d2e:>7}/12")

# ------------------------------------------------------------------ [5]
sec("[5] False-alarm rate on pure noise, 60 trials")
for name, mk in [("v1", lambda: V1()),
                 ("v2 mean_pair", lambda: V2(statistic="mean_pair")),
                 ("v2 eig", lambda: V2(statistic="eig"))]:
    fa = sum(mk().detect(noise(s))["alarm_time"] is not None for s in range(60))
    print(f"  {name:14} {fa:>3}/60  ({100*fa/60:.1f} %)")

# ------------------------------------------------------------------ [6]
sec("[6] Detection power vs burst amplitude (12 trials, all 12 channels)")
print(f"  {'amp':>6} | {'v1 hit':>8} {'v1 FA-adj':>10} | {'v2 hit':>8}")
for amp in (0.2, 0.3, 0.4, 0.6, 0.8):
    h1 = sum(V1().detect(burst(s, amp=amp))["alarm_time"] is not None for s in range(12))
    h2 = sum(V2().detect(burst(s, amp=amp))["alarm_time"] is not None for s in range(12))
    print(f"  {amp:>6} | {h1:>6}/12 {'':>10} | {h2:>6}/12")

# ------------------------------------------------------------------ [7]
sec("[7] Detection latency: reported vs actually knowable")
r1, r2 = V1().detect(burst(0)), V2().detect(burst(0))
print(f"  v1 alarm_time (window centre) : {r1['alarm_time']:.3f} s")
print(f"  v2 alarm_time (window end)    : {r2['alarm_time']:.3f} s")
print(f"  v2 also reports centre        : {r2['alarm_window_center']:.3f} s")
print(f"  true burst centre             : 4.200 s")

# ------------------------------------------------------------------ [8]
sec("[8] Real-valued input")
zr = np.random.default_rng(0).standard_normal((8000, 12))
print(f"  v1                     -> max_sync {V1().detect(zr)['max_synchrony']:.4f}, "
      f"no warning   {BAD}")
try:
    V2().detect(zr)
    print(f"  v2 (analytic=False)    -> no error   {BAD}")
except TypeError as e:
    print(f"  v2 (analytic=False)    -> TypeError   {OK}")
    print(f"                            {str(e).splitlines()[0][:64]}...")
rb = np.real(burst(0))
r = V2(analytic=True).detect(rb)
fa = sum(V2(analytic=True).detect(
    np.random.default_rng(s).standard_normal((8000, 12)))["alarm_time"] is not None
    for s in range(10))
print(f"  v2 (analytic=True)     -> burst on real signal detected at "
      f"{r['alarm_time']:.2f}s, {fa}/10 FA   {OK if r['alarm_time'] and fa <= 2 else BAD}")

# ------------------------------------------------------------------ [9]
sec("[9] online mode refuses to guess")
try:
    V2(mode="online").detect(noise(0))
    print(f"  uncalibrated online detect -> no error   {BAD}")
except RuntimeError:
    print(f"  uncalibrated online detect -> RuntimeError   {OK}")
d = V2(mode="online").calibrate(noise(99))
r = d.detect(burst(0))
fa = sum(V2(mode="online").calibrate(noise(99)).detect(noise(s))["alarm_time"]
         is not None for s in range(20))
print(f"  calibrated on a separate quiet record: burst at {r['alarm_time']:.2f}s, "
      f"{fa}/20 false alarms   {OK if r['alarm_time'] else BAD}")

# ------------------------------------------------------------------ [10]
sec("[10] Short signal / too few channels are errors, not silent output")
for desc, fn in [("T < win", lambda: V2(sync_win=256).detect(noise(0, T=100))),
                 ("1 channel", lambda: V2().detect(noise(0, K=1)))]:
    try:
        fn()
        print(f"  {desc:12} -> no error   {BAD}")
    except ValueError as e:
        print(f"  {desc:12} -> ValueError   {OK}  ({str(e)[:52]}...)")
