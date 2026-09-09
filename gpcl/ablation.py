"""Which stage actually earns its place? Isolate each one. 5 seeds."""
import torch
import torch.nn as nn
from gpcl_v2 import GPCLayer, SafeModel

D, C, N = 64, 4, 20000


def make(seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (N,), generator=g)
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


def spike(x, seed, mag, frac=0.05):
    g = torch.Generator().manual_seed(seed)
    x = x.clone()
    n = int(len(x) * frac)
    i = torch.randperm(len(x), generator=g)[:n]
    x[i, torch.randint(0, D, (n,), generator=g)] = mag
    return x


CONFIGS = {
    "bare model (no layer)":        None,
    "norm clamp only":              dict(num_features=None, gate_mode="off", clamp_norm=True),
    "per-feature clamp only":       dict(num_features=D,    gate_mode="off", clamp_norm=False),
    "anomaly gate only":            dict(num_features=D,    gate_mode="anomaly", clamp_norm=False,
                                         feature_clamp_k=1e9),
    "hopf gate only (v1's idea)":   dict(num_features=None, gate_mode="hopf", clamp_norm=False),
    "feature clamp + gate":         dict(num_features=D,    gate_mode="anomaly", clamp_norm=False),
    "everything (v2 default)":      dict(num_features=D,    gate_mode="anomaly", clamp_norm=True),
}

for mag in (100.0, 1000.0):
    print("=" * 78)
    print(f"spike magnitude {mag:.0f} on 5% of test rows   (mean of 5 seeds)")
    print("=" * 78)
    print(f"  {'configuration':30}{'clean':>9}{'spiked':>9}{'damage':>10}"
          f"{'damage removed':>17}")
    base_damage = None
    for name, kw in CONFIGS.items():
        cs, ss = [], []
        for seed in range(5):
            Xtr, ytr, Xte, yte = make(seed)
            m = train(Xtr, ytr, seed)
            Xsp = spike(Xte, seed, mag)
            if kw is None:
                mm = m
            else:
                mm = SafeModel(m, **kw)
                mm.calibrate(Xtr)
                mm.eval()
            cs.append(acc(mm, Xte, yte))
            ss.append(acc(mm, Xsp, yte))
        c, s = sum(cs) / 5, sum(ss) / 5
        dmg = c - s
        if base_damage is None:
            base_damage = dmg
            rem = ""
        else:
            rem = f"{100 * (1 - dmg / base_damage):>15.0f} %"
        print(f"  {name:30}{c:>9.2f}{s:>9.2f}{dmg:>10.2f}{rem:>17}")
    print()
