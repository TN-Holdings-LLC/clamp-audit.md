# -*- coding: utf-8 -*-
"""
psf_zero_aerospace_core.py — REFERENCE STAND-IN, not your module.

Your real psf_zero_aerospace_core.py was not attached, so the improved test
harness had nothing to import and could not be run — and shipping an
unrun test would contradict the whole point. This file is a correct
reference implementation inferred from how the pasted test calls it:

    allocate_thrusters(tau_des, B, tmax, mib) -> x
    _apply_soft_mib(x, B, tau_des, tmax, mib, W, reg) -> x
    PSFZeroAttitudeController(tau=...).compute_torque(q, q_des, omega, Kq, Kw)
    zero_clamp(x, tau)

Replace this file with yours and re-run the test: the harness is written
against those four signatures only, so it will exercise your code and tell
you which parts disagree with it.
"""
import numpy as np

try:
    import cvxpy as cp
except ImportError:            # the QP path degrades to the fallback
    cp = None

__all__ = ["zero_clamp", "PSFZeroAttitudeController",
           "allocate_thrusters", "_apply_soft_mib"]


def zero_clamp(x, tau=1.0):
    """Smooth saturation, unit slope at 0, asymptote at tau."""
    x = np.asarray(x, dtype=float)
    t = max(float(tau), 1e-12)
    n = np.linalg.norm(x)
    if n <= 1e-14:
        return x
    return x * (t * np.tanh(n / t) / n)


def _q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


class PSFZeroAttitudeController:
    """Quaternion-feedback attitude controller with a saturated P term.

    The sign convention on the error quaternion is the part that decides
    whether a near-antipodal manoeuvre unwinds: flipping qe so qe[0] >= 0
    picks the short rotation.
    """

    def __init__(self, tau=0.8):
        self.tau = float(tau)

    def compute_torque(self, q, q_des, omega, Kq=5.0, Kw=10.0):
        q = np.asarray(q, float)
        q_des = np.asarray(q_des, float)
        omega = np.asarray(omega, float)
        qe = _q_mul(_q_conj(q_des), q)
        if qe[0] < 0.0:                 # shortest-path branch
            qe = -qe
        return -Kq * zero_clamp(qe[1:4], self.tau) - Kw * omega


def _cost(x, B, tau_des, W, reg):
    r = W @ (B @ x - tau_des)
    return 0.5 * float(r @ r) + 0.5 * reg * float(x @ x)


def _apply_soft_mib(x, B, tau_des, tmax, mib, W, reg):
    """Snap any thruster sitting inside its dead zone (0, mib) to whichever
    endpoint costs less, so the command is physically realisable."""
    x = np.clip(np.asarray(x, float).copy(), 0.0, tmax)
    for i in np.flatnonzero((x > 1e-12) & (x < np.asarray(mib) - 1e-12)):
        lo = x.copy(); lo[i] = 0.0
        hi = x.copy(); hi[i] = min(mib[i], tmax[i])
        x = lo if _cost(lo, B, tau_des, W, reg) <= _cost(hi, B, tau_des, W, reg) else hi
    return x


def allocate_thrusters(tau_des, B, tmax, mib, W=None, reg=1e-4):
    """Least-squares thrust allocation with box bounds, then soft-MIB."""
    tau_des = np.asarray(tau_des, float)
    B = np.asarray(B, float)
    tmax = np.asarray(tmax, float)
    mib = np.asarray(mib, float)
    W = np.eye(B.shape[0]) if W is None else np.asarray(W, float)
    n = B.shape[1]

    x_raw = None
    if cp is not None:
        x = cp.Variable(n, nonneg=True)
        obj = 0.5 * cp.sum_squares(W @ (B @ x - tau_des)) + 0.5 * reg * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj), [x <= tmax])
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                x_raw = np.asarray(x.value, float)
        except Exception:
            x_raw = None
    if x_raw is None:                      # fallback: same shape, always (n,)
        x_raw = np.linalg.lstsq(B, tau_des, rcond=None)[0]
    x_raw = np.clip(x_raw, 0.0, tmax)
    return _apply_soft_mib(x_raw, B, tau_des, tmax, mib, W, reg)
