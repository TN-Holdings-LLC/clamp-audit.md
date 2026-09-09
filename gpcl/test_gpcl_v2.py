"""Head-to-head: v1 (as pasted, with its fix) vs v2. Same measurements."""
import time
import torch
import torch.nn as nn

from gpcl_original import GPCLayer as V1, SafeModel as SafeV1
from gpcl_v2 import GPCLayer as V2, SafeModel as SafeV2

torch.manual_seed(0)
OK, BAD = "PASS", "FAIL"


def sec(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ------------------------------------------------------------------ [1]
sec("[1] The vectorised scan must reproduce v1's loop EXACTLY")
for T in (2, 7, 128, 512, 2048):
    z = torch.randn(4, T, 1)
    v1, v2 = V1(lam=0.15), V2(num_features=None, lam=0.15)
    a, b = v1._causal_eit_smoothing(z), v2._causal_eit_smoothing(z)
    dev = (a - b).abs().max().item()
    print(f"  T={T:>5}  max|loop - scan| = {dev:.3e}   "
          f"{OK if dev < 1e-5 else BAD}")

# ------------------------------------------------------------------ [2]
sec("[2] Identity on in-distribution data (v1 crushed clean inputs)")
print(f"  {'dim':>6} {'v1 ||out||/||in||':>19} {'v2 ||out||/||in||':>19}")
for d in (16, 256, 768, 4096):
    x = torch.randn(256, 8, d)
    r1 = (V1(sigma=1.0)(x).norm(dim=-1) / x.norm(dim=-1)).mean().item()
    m2 = V2(num_features=d).calibrate(x).eval()
    r2 = (m2(x).norm(dim=-1) / x.norm(dim=-1)).mean().item()
    print(f"  {d:>6} {r1:>19.5f} {r2:>19.5f}")
print("  (1.0 = untouched. v2 leaves ~half the samples exactly alone and")
print("   only bends the upper tail, so the mean sits just under 1.)")

# ------------------------------------------------------------------ [3]
sec("[3] Spike suppression at d=768 (v1 saturated and could not tell)")
d = 768
clean = torch.randn(512, 1, d)
m2 = V2(num_features=d).calibrate(clean).eval()
m1 = V1(sigma=1.0)
print(f"  {'spike':>9} {'v1 ||out||':>13} {'v1 vs clean':>13} "
      f"{'v2 ||out||':>13} {'v2 vs clean':>13}")
base1 = m1(clean[:1]).norm().item()
base2 = m2(clean[:1]).norm().item()
for mag in (0.0, 1e2, 1e3, 1e4, 1e5):
    xs = clean[:1].clone()
    if mag:
        xs[0, 0, 0] = mag
    n1, n2 = m1(xs).norm().item(), m2(xs).norm().item()
    print(f"  {mag:>9.0f} {n1:>13.4f} {n1/base1:>13.4f} "
          f"{n2:>13.4f} {n2/base2:>13.4f}")
print("  v1's column barely moves -- but it barely moves because it was")
print("  ALREADY saturated on the clean input, not because it filtered.")

# ------------------------------------------------------------------ [4]
sec("[4] Gate: does it actually respond to an anomaly?")
print(f"  {'spike':>9} {'v1 gate':>12} {'v2 gate':>12}")
for mag in (0.0, 1e1, 1e2, 1e3, 1e5):
    xs = clean[:1].clone()
    if mag:
        xs[0, 0, 0] = mag
    xp = m1._projective_clamp(xs)
    g1 = torch.sigmoid(m1._causal_eit_smoothing(m1._hopf_projection(xp)) * 5.0)
    _, diag = m2(xs, return_diagnostics=True)
    print(f"  {mag:>9.0f} {float(g1):>12.6f} {float(diag['gate']):>12.6f}")

# ------------------------------------------------------------------ [5]
sec("[5] Padding bias (identical data, feature count off a multiple of 4)")
for dd in (60, 61, 62, 63, 64):
    x = torch.randn(4096, 1, dd)
    z1 = V1()._hopf_projection(V1()._projective_clamp(x)).mean().item()
    z2 = V2(num_features=dd, gate_mode="hopf")._hopf_projection(x).mean().item()
    print(f"  d={dd:>3} (pad={(4-dd%4)%4})   v1 E[z]={z1:+.5f}   v2 E[z]={z2:+.5f}")

# ------------------------------------------------------------------ [6]
sec("[6] Speed of the temporal stage")
print(f"  {'T':>6} {'v1 loop':>12} {'v2 scan':>12} {'speedup':>10}")
for T in (128, 512, 2048):
    z = torch.randn(32, T, 1)
    v1, v2 = V1(), V2()
    for f in (v1._causal_eit_smoothing, v2._causal_eit_smoothing):
        f(z)
    t0 = time.perf_counter()
    for _ in range(10):
        v1._causal_eit_smoothing(z)
    d1 = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        v2._causal_eit_smoothing(z)
    d2 = (time.perf_counter() - t0) / 10
    print(f"  {T:>6} {d1*1e3:>10.2f}ms {d2*1e3:>10.2f}ms {d1/d2:>9.1f}x")

# ------------------------------------------------------------------ [7]
sec("[7] End-to-end on a frozen classifier (5 seeds, magnitude-encoded task)")
D, C, N = 64, 4, 20000


def make(seed, mag_matters):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (N,), generator=g)
    if mag_matters:
        dirn = nn.functional.normalize(torch.randn(N, D, generator=g), dim=-1)
        rad = (y.float() + 1.0) * 4.0 + torch.randn(N, generator=g) * 0.5
        X = dirn * rad[:, None]
    else:
        mu = torch.randn(C, D, generator=g) * 1.5
        X = mu[y] + torch.randn(N, D, generator=g)
    return X[:16000], y[:16000], X[16000:], y[16000:]


def train(Xtr, ytr, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, C))
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(30):
        for i in range(0, len(Xtr), 256):
            o.zero_grad()
            nn.functional.cross_entropy(m(Xtr[i:i+256]), ytr[i:i+256]).backward()
            o.step()
    return m.eval()


def acc(m, x, y):
    with torch.no_grad():
        return (m(x).argmax(-1) == y).float().mean().item() * 100


def spike(x, seed, mag=1000.0, frac=0.05):
    g = torch.Generator().manual_seed(seed)
    x = x.clone()
    n = int(len(x) * frac)
    i = torch.randperm(len(x), generator=g)[:n]
    x[i, torch.randint(0, D, (n,), generator=g)] = mag
    return x


for mag_matters in (False, True):
    print(("\n  B) magnitude-encoded task" if mag_matters
           else "\n  A) direction-encoded task") + "   (mean of 5 seeds)")
    res = {}
    for seed in range(5):
        Xtr, ytr, Xte, yte = make(seed, mag_matters)
        m = train(Xtr, ytr, seed)
        Xsp = spike(Xte, seed)
        res.setdefault("bare model", []).append((acc(m, Xte, yte), acc(m, Xsp, yte)))
        s1 = SafeV1(m, sigma=1.0).eval()
        res.setdefault("v1 SafeModel", []).append((acc(s1, Xte, yte), acc(s1, Xsp, yte)))
        s2 = SafeV2(m, num_features=D)
        s2.calibrate(Xtr)
        s2.eval()
        res.setdefault("v2 SafeModel", []).append((acc(s2, Xte, yte), acc(s2, Xsp, yte)))
        s3 = SafeV2(m, num_features=D, clamp_norm=False)
        s3.calibrate(Xtr)
        s3.eval()
        res.setdefault("v2 clamp_norm=False", []).append(
            (acc(s3, Xte, yte), acc(s3, Xsp, yte)))
    print(f"    {'':16}{'clean':>9}{'spiked':>9}{'spike damage':>15}")
    for k, v in res.items():
        c = sum(a for a, _ in v) / len(v)
        s = sum(b for _, b in v) / len(v)
        print(f"    {k:16}{c:>9.2f}{s:>9.2f}{c-s:>14.2f} pts")

# ------------------------------------------------------------------ [8]
sec("[8] Failure modes are now loud instead of silent")
def _cnn_case():
    m = V2(gate_mode="off")
    m.calibrate(torch.randn(64, 8))
    return m(torch.randn(2, 3, 32, 32))


for desc, fn in [
    ("4-D CNN tensor (B,C,H,W)", _cnn_case),
    ("integer token ids", lambda: SafeV2(nn.Embedding(100, 16))(
        torch.randint(0, 100, (2, 8)))),
    ("uncalibrated eval()", lambda: V2(num_features=8).eval()(torch.randn(4, 8))),
]:
    try:
        fn()
        print(f"  {desc:28} -> no error raised   {BAD}")
    except Exception as e:
        print(f"  {desc:28} -> {type(e).__name__}   {OK}")
        print(f"  {'':28}    {str(e).splitlines()[0][:100]}")

# ------------------------------------------------------------------ [9]
sec("[9] Gradients still flow (the layer stays trainable-through)")
x = torch.randn(8, 4, 64, requires_grad=True)
m = V2(num_features=64)
m.calibrate(torch.randn(256, 4, 64))
m(x).sum().backward()
g = x.grad
print(f"  grad finite: {bool(torch.isfinite(g).all())}   "
      f"grad nonzero: {bool((g != 0).any())}   "
      f"{OK if torch.isfinite(g).all() and (g != 0).any() else BAD}")
