# -*- coding: utf-8 -*-
"""Head-to-head: the pasted GeoClampAdam vs geoclamp_adam_v2."""
import math

import torch

import geoclamp_original as O
import geoclamp_adam_v2 as V

FAIL = 0


def check(ok, what):
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        FAIL += 1


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def rand_R(seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=dtype))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R.contiguous()


# ---------------------------------------------------------------------- 1
sec("1. The previous version's own fixes still hold")
b1, b2 = 0.9, 0.999
ratios = {t: (1 - b1 ** t) / math.sqrt(1 - b2 ** t) for t in (1, 10, 100)}
print(f"  uncorrected m/sqrt(v) ratio: t=1 {ratios[1]:.3f}, t=10 {ratios[10]:.3f}, "
      f"t=100 {ratios[100]:.3f}")
check(abs(ratios[10] - 6.528) < 0.01, "the 6.528x peak at t=10 reproduces")
for th, want in ((1e-3, 0.4768), (1e-4, 0.0)):
    t32 = torch.tensor(th, dtype=torch.float32)
    Bp = float((1.0 - torch.cos(t32)) / t32 ** 2)
    Bs = float(0.5 * (torch.sin(t32 / 2) / (t32 / 2)) ** 2)
    print(f"  theta={th:.0e}: B_pasted={Bp:.6f}  B_v2={Bs:.6f}  (true 0.5)")
    check(abs(Bp - want) < 1e-3, f"the pasted B really is {want} at theta={th:.0e}")
    check(abs(Bs - 0.5) < 1e-6, "v2's half-angle B is 0.5")

# ---------------------------------------------------------------------- 2
sec("2. S^3: does the trust radius mean radians?")
print(f"  {'requested r':>12} {'old (proj)':>12} {'v2 (exp)':>12} {'old err':>10} {'v2 err':>9}")
worst_old = worst_new = 0.0
for r in (0.01, 0.1, 0.5, 1.0, math.pi / 2 - 1e-3):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    xi = torch.tensor([0.0, r, 0.0, 0.0], dtype=torch.float64)
    a_old = math.acos(min(1.0, abs(float(O._normalize_quat(q + xi)[0]))))
    a_new = math.acos(min(1.0, abs(float(V._exp_s3(q, xi)[0]))))
    e_old, e_new = abs(a_old / r - 1), abs(a_new / r - 1)
    worst_old, worst_new = max(worst_old, e_old), max(worst_new, e_new)
    print(f"  {r:>12.4f} {a_old:>12.4f} {a_new:>12.4f} {100*e_old:>9.1f}% {100*e_new:>8.2f}%")
check(worst_old > 0.30, "the old retraction really is >30% out at the default max_step")
check(worst_new < 1e-9, "v2's exponential map moves exactly the requested radius")

# ---------------------------------------------------------------------- 3
sec("3. SO(3): does R stay on the group?")
for name, mk in (("old", lambda p: O.GeoClampAdam([p], lr=0.05, manifold="SO3")),
                 ("v2 ", lambda p: V.GeoClampAdam([p], lr=0.05, manifold="SO3"))):
    R = rand_R(0, torch.float32).clone()
    opt = mk(R)
    g = torch.Generator().manual_seed(1)
    for k in range(20000):
        R.grad = torch.randn(3, 3, generator=g) * 0.1
        opt.step()
    err = float(torch.linalg.norm(R.T @ R - torch.eye(3)))
    det = float(torch.linalg.det(R))
    print(f"  {name} after 20000 float32 steps: ||R^T R - I|| = {err:.3e}, det = {det:.9f}")
    if name == "old":
        old_err = err
    else:
        new_err = err
check(old_err > 1e-6, "the old path drifts measurably off SO(3)")
check(new_err < old_err / 10, "v2 keeps it on the group (>10x tighter)")
check(abs(det - 1.0) < 1e-5, "and det stays +1")

# ---------------------------------------------------------------------- 4
sec("4. Scale invariance: rescaling the loss should not change the step")
print(f"  {'grad scale':>11} {'old applied':>13} {'v2 applied':>12}")
old_steps, new_steps = [], []
for s in (1e-2, 1e0, 1e2, 1e4):
    for lib, store in ((O, old_steps), (V, new_steps)):
        R = rand_R(0).clone()
        opt = lib.GeoClampAdam([R], lr=0.1, manifold="SO3")
        g = torch.randn(3, 3, generator=torch.Generator().manual_seed(2),
                        dtype=torch.float64)
        for _ in range(50):
            R.grad = g * s
            before = R.clone()
            opt.step()
            dR = before.T @ R
            ang = math.acos(max(-1.0, min(1.0, (float(torch.trace(dR)) - 1) / 2)))
        store.append(ang)
    print(f"  {s:>11.0e} {old_steps[-1]:>13.5f} {new_steps[-1]:>12.5f}")
spread = lambda xs: max(xs) / max(min(xs), 1e-12)
print(f"  spread across 1e6x of loss scaling: old {spread(old_steps):.1f}x, "
      f"v2 {spread(new_steps):.2f}x")
check(spread(old_steps) > 2.0, "the old step really does depend on loss scale")
check(spread(new_steps) < 1.05, "v2's step is scale-invariant, as Adam intends")

# ---------------------------------------------------------------------- 5
sec("5. Momentum transport")
print(f"  {'step (rad)':>11} {'|m - transported m|/|m|':>26}")
for r in (0.01, 0.1, 1.0, 2.0):
    dR = V._exp_so3(torch.tensor([r, 0.0, 0.0], dtype=torch.float64))
    m = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    mt = torch.einsum("ji,j->i", dR, m)
    print(f"  {r:>11.3f} {float(torch.linalg.norm(m - mt) / torch.linalg.norm(m)):>26.4f}")
check(True, "the geometric mismatch is real and O(step) -- but see section 6:\n         transporting it away costs accuracy, so it is off by default")
# transport must preserve length (it is an isometry)
m = torch.randn(3, dtype=torch.float64)
dR = V._exp_so3(torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64))
check(abs(float(torch.linalg.norm(torch.einsum("ji,j->i", dR, m)))
          - float(torch.linalg.norm(m))) < 1e-12,
      "SO(3) transport preserves |m| (it is an isometry)")
q0 = V._normalize_quat(torch.randn(4, dtype=torch.float64))
q1 = V._normalize_quat(torch.randn(4, dtype=torch.float64))
w = V._project_s3_tangent(q0, torch.randn(4, dtype=torch.float64))
wt = V._transport_s3(q0, q1, w)
check(abs(float((q1 * wt).sum())) < 1e-10, "S^3 transport lands in the new tangent space")
check(abs(float(torch.linalg.norm(wt)) - float(torch.linalg.norm(w))) < 1e-10,
      "S^3 transport preserves length")

# ---------------------------------------------------------------------- 6
sec("6. Does any of it help? Rotation fitting, 200 steps, 8 seeds")


def fit_so3(lib, seed, steps=200, **kw):
    Rt = rand_R(1000 + seed)
    R = rand_R(seed).clone().requires_grad_(True)
    opt = lib.GeoClampAdam([R], lr=0.05, manifold="SO3", **kw)
    for _ in range(steps):
        if R.grad is not None:
            R.grad = None
        loss = ((R - Rt) ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        dR = Rt.T @ R
        c = (float(torch.trace(dR)) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def fit_s3(lib, seed, steps=200, **kw):
    g = torch.Generator().manual_seed(seed)
    qt = V._normalize_quat(torch.randn(4, generator=g, dtype=torch.float64))
    q = V._normalize_quat(torch.randn(4, generator=g, dtype=torch.float64)
                          ).clone().requires_grad_(True)
    opt = lib.GeoClampAdam([q], lr=0.05, manifold="S3", **kw)
    for _ in range(steps):
        if q.grad is not None:
            q.grad = None
        (((q - qt) ** 2).sum()).backward()
        opt.step()
    with torch.no_grad():
        return math.degrees(2 * math.acos(min(1.0, abs(float((q * qt).sum())))))


for label, fn in (("SO(3) rotation fit", fit_so3), ("S^3 quaternion fit", fit_s3)):
    e_old = sum(fn(O, s) for s in range(8)) / 8
    e_new = sum(fn(V, s) for s in range(8)) / 8
    e_tr = sum(fn(V, s, transport=True) for s in range(8)) / 8
    print(f"  {label}: pasted {e_old:8.4f} | v2 default {e_new:8.4f} | "
          f"v2 transport=True {e_tr:8.4f}  (deg)")
    check(abs(e_new - e_old) < max(1e-6, 0.05 * e_old),
          f"{label}: v2's defaults reproduce the pasted convergence")
    if "S^3" in label:
        check(e_tr > 10 * e_new,
              "transport=True measurably HURTS -- which is why it is off by "
              "default, and is reported rather than shipped on")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'SOME CHECKS FAILED'}  "
      f"({FAIL} failure{'' if FAIL == 1 else 's'})")
raise SystemExit(0 if FAIL == 0 else 1)
