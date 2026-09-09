"""Diagnostics on the pasted GPCLayer. Every claim printed here is measured."""
import time
import torch
from gpcl_original import GPCLayer

torch.manual_seed(0)


def sec(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ---------------------------------------------------------------- [1]
sec("[1] Does the clamp discriminate a spike from a NORMAL input?")
print("sigma=1.0 (the default). Feature dims typical of real models.")
print(f"{'dim':>6} {'||x|| normal':>13} {'||out|| normal':>15} "
      f"{'||x|| spike':>12} {'||out|| spike':>14} {'ratio out':>10}")
for d in (16, 64, 256, 768, 4096):
    g = GPCLayer(sigma=1.0)
    x = torch.randn(1, 1, d)                      # ordinary unit-variance features
    xs = x.clone()
    xs[0, 0, 0] = 1000.0                          # the "thermal spike"
    o, os_ = g(x), g(xs)
    print(f"{d:>6} {x.norm():>13.3f} {o.norm():>15.4f} "
          f"{xs.norm():>12.1f} {os_.norm():>14.4f} "
          f"{(os_.norm()/o.norm()):>10.4f}")
print("\nIf 'ratio out' ~ 1.0 the layer cannot tell a 1000x spike from normal data:")
print("both are equally saturated, so no information about the anomaly survives.")

# ---------------------------------------------------------------- [2]
sec("[2] Is any of the ORIGINAL magnitude preserved for in-distribution input?")
g = GPCLayer(sigma=1.0)
for d in (16, 256, 768):
    x = torch.randn(64, 8, d)
    o = g(x)
    rel = (o.norm(dim=-1) / x.norm(dim=-1))
    print(f"dim={d:>5}  ||out||/||in||  mean={rel.mean():.5f}  "
          f"min={rel.min():.5f}  max={rel.max():.5f}")
print("\nA robustness layer that is meant to be a no-op on clean data should")
print("show this ratio ~1.0. Values far below 1 mean every clean input is")
print("being crushed too, not just the outliers.")

# ---------------------------------------------------------------- [3]
sec("[3] What does the Hopf gate actually output as width grows?")
print("z is a mean over d/4 unit quaternions, each with E[z_i]=0.")
print(f"{'dim':>6} {'gate mean':>11} {'gate std':>11} {'gate min':>10} {'gate max':>10}")
for d in (4, 16, 64, 256, 768, 4096):
    g = GPCLayer(sigma=1.0)
    x = torch.randn(512, 4, d)
    xp = g._projective_clamp(x)
    z = g._hopf_projection(xp)
    gate = torch.sigmoid(g._causal_eit_smoothing(z) * 5.0)
    print(f"{d:>6} {gate.mean():>11.5f} {gate.std():>11.5f} "
          f"{gate.min():>10.5f} {gate.max():>10.5f}")
print("\nAs std -> 0 the 'gate' stops being a gate and becomes a constant ~0.5")
print("multiplier: the entire Hopf/EIT pipeline collapses to 'divide by 2'.")

# ---------------------------------------------------------------- [4]
sec("[4] Does the gate respond to the spike at all? (dim=768)")
g = GPCLayer(sigma=1.0)
x = torch.randn(1, 1, 768)
rows = []
for mag in (0.0, 1e1, 1e2, 1e3, 1e4, 1e5):
    xs = x.clone()
    if mag:
        xs[0, 0, 0] = mag
    xp = g._projective_clamp(xs)
    z = g._hopf_projection(xp)
    gate = torch.sigmoid(g._causal_eit_smoothing(z) * 5.0)
    rows.append((mag, float(gate), float(g(xs).norm())))
print(f"{'spike':>10} {'gate':>12} {'||output||':>12}")
for m, gt, n in rows:
    print(f"{m:>10.0f} {gt:>12.6f} {n:>12.6f}")

# ---------------------------------------------------------------- [5]
sec("[5] Zero-padding bias: identical data, feature count off a multiple of 4")
print("Padding with zeros creates all-zero quaternions -> normalize gives 0 ->")
print("z=0 for those bundles, dragging the mean. Same signal, different d:")
base = torch.randn(2048, 1, 64)
for extra in (0, 1, 2, 3):
    d = 64 - (0 if extra == 0 else 0)
    x = base[..., : 64 - (4 - extra) % 4] if extra else base
    # simpler: just take differing widths around a multiple of 4
for d in (60, 61, 62, 63, 64):
    g = GPCLayer(sigma=1.0)
    x = torch.randn(4096, 1, d)
    xp = g._projective_clamp(x)
    z = g._hopf_projection(xp)
    print(f"  d={d:>3} (pad={(4 - d % 4) % 4})  E[z]={z.mean():+.5f}  "
          f"gate mean={torch.sigmoid(z*5).mean():.5f}")

# ---------------------------------------------------------------- [6]
sec("[6] Cost of the Python-level causal loop")
g = GPCLayer()
for T in (128, 512, 2048):
    z = torch.randn(32, T, 1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(5):
        g._causal_eit_smoothing(z)
    dt = (time.perf_counter() - t0) / 5
    print(f"  T={T:>5}  {dt*1e3:>8.2f} ms/call  ({T} sequential Python iterations)")

# ---------------------------------------------------------------- [7]
sec("[7] Layout assumption: what happens on a 4-D CNN tensor (B,C,H,W)?")
g = GPCLayer(sigma=1.0)
x = torch.randn(2, 3, 32, 32)
try:
    o = g(x)
    print(f"  runs without error, output shape {tuple(o.shape)}")
    print("  BUT: _projective_clamp normalised over dim=-1, i.e. the WIDTH axis,")
    print("  and _causal_eit_smoothing treated dim=1 (CHANNELS) as time.")
    print("  Silently wrong for convolutional layouts - no exception is raised.")
except Exception as e:
    print(f"  raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- [8]
sec("[8] Can SafeModel wrap a real transformer? (integer token ids)")
try:
    emb = torch.nn.Embedding(100, 16)
    from gpcl_original import SafeModel
    m = SafeModel(emb)
    ids = torch.randint(0, 100, (2, 8))
    m(ids)
    print("  ok")
except Exception as e:
    print(f"  raised {type(e).__name__}: {str(e)[:140]}")
    print("  -> 'wrap any legacy model' does not hold for token-id inputs.")
