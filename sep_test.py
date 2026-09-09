"""Is the eigenvalue statistic genuinely better, or does it just ride a
different threshold? Measure separation (d') of each statistic itself,
independent of any CUSUM threshold. 20 trials per point."""
import numpy as np
from eit_v2 import EITDetector

K, T, FS = 12, 8000, 1000.0


def noise(s):
    r = np.random.default_rng(s)
    return r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))


def burst(s, m, amp=0.8):
    r = np.random.default_rng(s)
    t = np.arange(T) / FS
    z = r.standard_normal((T, K)) + 1j * r.standard_normal((T, K))
    b = np.exp(1j * 2 * np.pi * 45 * t) * np.exp(-((t - 4.2) / 0.4) ** 2)
    for k in range(m):
        z[:, k] += amp * b * (1 + 0.3 * r.standard_normal(T))
    return z


d = EITDetector()
null = {"mean_pair": [], "eig": []}
for s in range(20):
    o = d._series(noise(s))
    null["mean_pair"].append(o["mean_pair_plv"])
    null["eig"].append(o["eig_sync"])
stats = {k: (np.concatenate(v).mean(), np.concatenate(v).std())
         for k, v in null.items()}
print("Pure-noise null distribution of each statistic (20 trials):")
for k, (mu, sd) in stats.items():
    print(f"  {k:12} mean {mu:.5f}  sd {sd:.5f}")

print("\nSeparation d' = (peak during burst - null mean) / null sd")
print(f"  {'m locked':>9} {'mean_pair d prime':>19} {'eig d prime':>13} {'ratio':>8}")
for m in (12, 8, 6, 4, 3, 2):
    pk = {"mean_pair": [], "eig": []}
    for s in range(20):
        o = d._series(burst(s, m))
        pk["mean_pair"].append(o["mean_pair_plv"].max())
        pk["eig"].append(o["eig_sync"].max())
    dp = {k: (np.mean(pk[k]) - stats[k][0]) / stats[k][1] for k in pk}
    print(f"  {m:>9} {dp['mean_pair']:>19.2f} {dp['eig']:>13.2f} "
          f"{dp['eig']/dp['mean_pair']:>8.2f}x")

print("\nHow each statistic's peak scales with m (raw, minus null mean):")
print(f"  {'m':>4} {'mean_pair':>12} {'eig':>12}")
for m in (12, 8, 6, 4, 3, 2):
    pk = {"mean_pair": [], "eig": []}
    for s in range(10):
        o = d._series(burst(s, m))
        pk["mean_pair"].append(o["mean_pair_plv"].max())
        pk["eig"].append(o["eig_sync"].max())
    print(f"  {m:>4} {np.mean(pk['mean_pair'])-stats['mean_pair'][0]:>12.4f} "
          f"{np.mean(pk['eig'])-stats['eig'][0]:>12.4f}")
