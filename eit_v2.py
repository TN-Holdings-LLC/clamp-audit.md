"""
EIT Phase Synchrony Detector v2
===============================

Multi-channel complex-signal synchrony change-point detector: EIT (EMA)
smoothing -> weighted phase-locking statistic -> CUSUM.

The v1 this replaces had already fixed the important bug (taking the
complex mean BEFORE the modulus, i.e. a real PLV instead of
mean|cos dphi|), and its stated numbers reproduce: pure-noise baseline
measured 0.0691 +/- 0.0005 against its claimed 0.068 +/- 0.0004, burst
found in 10/10 trials at 3.71-3.84 s. What follows are the problems that
remained.

WHAT CHANGED AND WHY -- every item measured, see diag_eit.py / test_eit_v2.py

 1. THE CALIBRATION WAS WELDED TO ONE CONFIGURATION.  This is the big one.
    `kappa=0.07` was chosen to sit just above the pure-noise PLV baseline
    of ~0.069. But that baseline is a function of the window length and the
    smoothing constant, and both are exposed as constructor arguments with
    no warning. Measured pure-noise baseline and false-alarm rate:

        sync_win  128 -> baseline 0.0977 -> 6/6 FALSE ALARMS
        sync_win  192 -> baseline 0.0798 -> 6/6 FALSE ALARMS
        sync_win  256 -> baseline 0.0692 -> 0/6   (the tuned point)
        sync_win 1024 -> baseline 0.0348 -> 0/6   (but now half-deaf)

        eit_alpha 0.2 -> baseline 0.1253 -> 6/6 FALSE ALARMS
        eit_alpha 0.4 -> baseline 0.0893 -> 6/6 FALSE ALARMS
        eit_alpha 0.75-> baseline 0.0692 -> 0/6   (the tuned point)

    Shortening the window or weakening the smoothing reproduces v1's
    original "alarms immediately regardless of input" failure exactly --
    the bug the rewrite was supposed to have removed. The baseline of a PLV
    over N effectively-independent samples is ~0.886/sqrt(N_eff), so any
    fixed threshold is a statement about one window length.

    Fix: CUSUM no longer runs on the raw statistic. It runs on a robustly
    standardised one, u = (rho - median(rho)) / (1.4826 * MAD(rho)), so
    `kappa` and `eta` are in units of the statistic's own noise sigma and
    mean nothing about window length. Median/MAD rather than mean/std
    because a burst must not inflate the scale it is being measured
    against. Measured across win in {128...1024} and alpha in {0.2...3.0}:
    false alarms 0-1 out of 6 at every setting, burst still found 6/6.
    The one blemish is win=1024, where v2 gives 1/6 false alarms against
    v1's 0/6 -- at that window there are only ~110 windows in the record
    and the median/MAD estimate is itself noisy. v2 is not uniformly better
    than v1 at every single setting; it is better at not collapsing.

    Note also that v2's pair statistic is not numerically identical to
    v1's: v1 weighted by zero_clamp(sqrt(a_j a_k)), v2 by
    zero_clamp(a_j)*zero_clamp(a_k), so the matrix form is a single matmul.
    Measured pure-noise means differ accordingly (0.0692 vs 0.0755). This
    is a different weighting, not a correction of one -- and under the
    standardisation in this note the difference is absorbed anyway.

 2. NO WAY TO RUN IT ONLINE HONESTLY.  The median/MAD in (1) is computed
    over the whole record, which is fine offline but is not causal. v2
    makes the choice explicit: `mode="offline"` self-calibrates on the
    record; `mode="online"` requires `calibrate(reference_signal)` first
    and then runs strictly causally. v1 offered no such distinction and
    was implicitly offline anyway (its threshold came from prior tuning
    runs, not from the data in front of it).

 3. THE REPORTED ALARM TIME WAS 128 ms OPTIMISTIC.  `times` holds window
    CENTRES, and `alarm_time` divided a centre by fs. But a PLV over a
    256-sample window cannot be computed until all 256 samples exist, so
    the earliest an alarm is knowable is the window's END. Measured on the
    demo burst: reported 3.840 s, actually knowable at 3.968 s. v2 reports
    `alarm_time` at the window end and additionally returns
    `alarm_window_center` so the old number is still available, clearly
    labelled.

 4. AVERAGING OVER ALL PAIRS DILUTES PARTIAL SYNCHRONY QUADRATICALLY.  The
    demo locks all 12 channels, the easiest possible case. If only m of K
    lock, the mean over all pairs scales as ~(m/K)^2. v2 also computes the
    largest eigenvalue of the Hermitian weighted-PLV matrix and offers
    (lambda_max - 1)/(K - 1) as the default statistic, which decays more
    slowly. Measured separation d' = (burst peak - null mean)/null sd,
    20 trials per point:

        m locked   mean-pair d'   eigenvalue d'   ratio
            12          79.6           80.9       1.02x
             8          34.7           49.1       1.41x
             6          19.3           33.5       1.73x
             4           8.5           18.2       2.14x
             3           5.1           10.8       2.10x
             2           3.3            4.4       1.34x

    So the eigenvalue statistic is 1.3-2.1x better separated whenever
    synchrony is partial, and identical when it is total -- which is the
    expected shape, and the reason the original demo (all 12 locked) could
    not have revealed the difference.

    BE CAREFUL NOT TO OVERSELL THIS. End-to-end detection rates barely
    move, because both statistics are far above threshold at m>=4 and both
    are marginal at m=2: at a matched ~3% false-alarm rate, v1 and v2-eig
    both find 12/12 down to m=3 and both find 4/12 at m=2. The eigenvalue
    statistic buys margin, not new detections, on this test bench. It
    would matter on a harder one -- weaker bursts, more channels, a shorter
    window -- which has not been run here.

 5. REAL-VALUED INPUT PRODUCED SILENT NONSENSE.  np.angle() of a real
    array is only ever 0 or pi, so "phase synchrony" on real sensor data
    was measuring sign agreement. It ran without a warning and returned a
    plausible-looking 0.0877. v2 raises unless you opt in with
    `analytic=True`, which applies the Hilbert transform to build a proper
    analytic signal first.

 6. THE EIT FILTER WAS AN O(T) PYTHON LOOP.  Measured 23.2 ms at T=8000
    and 108.2 ms at T=32000. It is a first-order IIR; scipy.signal.lfilter
    computes the identical recurrence in one call. Verified equal to the
    loop to < 1e-12 max abs deviation. Measured ~30-90x faster.

 7. `mu0` AND `kappa` WERE DOING THE SAME JOB.  The increment was
    `x - mu0 - kappa` with `mu0=0.0`, so the reference level was
    silently `kappa` and the parameter named for the null mean was inert.
    In v2 the null mean is estimated (or calibrated) and `kappa` is only
    the CUSUM slack, in sigma units, which is what that parameter means
    in every textbook treatment.

 8. `fs` WAS STORED ON THE ACCUMULATOR AND NEVER USED.  Removed there;
    it lives on the detector, where it is actually used for time output.

UNCHANGED AND CORRECT, kept as-is: the PLV ordering fix (complex mean then
modulus) that was v1's own contribution; the CUSUM recursion itself; and
`zero_clamp` applied to amplitude as an amplitude-reliability weight,
which is a defensible use of it. Note that with tau=0.8 against noise of
unit scale the weights end up nearly uniform, so the weighting is close to
a no-op in the demo regime -- it earns its place only when channel
amplitudes are genuinely heterogeneous.

SCOPE. This detects a sustained INCREASE in phase locking against a
stationary background. It is not a general anomaly detector: a background
whose own synchrony drifts will defeat the calibration in (1) the same way
a wrong window length defeated v1's fixed threshold. `mode="online"` with
periodic recalibration is the intended answer there, and it is not tested
here.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.signal import hilbert, lfilter, lfilter_zi  # noqa: F401

__all__ = ["zero_clamp", "EITAccumulator", "SynchronyStatistic",
           "CUSUMDetector", "EITDetector"]


# ====================== Core Geometry /0 Projection ======================
def zero_clamp(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Soft-bound an unbounded non-negative quantity. Monotone, saturating
    at tau. Used here as an amplitude-reliability weight, which is a
    quantity that is actually unbounded -- applying it to something already
    in [0, 1] would be a no-op dressed as robustness."""
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(1.0 + (x / tau) ** 2)


# ====================== EIT (Exponential Information Tracking) ==========
@dataclass
class EITAccumulator:
    """First-order exponential smoother. Same recurrence as v1's loop:
        y[0] = z[0];  y[t] = (1-beta) y[t-1] + beta z[t],  beta = 1-exp(-alpha)
    computed with scipy.signal.lfilter instead of T Python iterations."""

    alpha: float = 0.8

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        self.beta = 1.0 - np.exp(-self.alpha)

    def filter(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z))
        if z.shape[0] == 1 and z.ndim == 2 and z.size != 1:
            z = z.reshape(-1, 1) if z.shape[0] == 1 else z
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        b = np.array([self.beta])
        a = np.array([1.0, -(1.0 - self.beta)])
        # zi chosen so that y[0] == z[0] exactly, matching v1's initialisation.
        zi = (1.0 - self.beta) * z[0][None, :]
        y, _ = lfilter(b, a, z, axis=0, zi=zi)
        return y


# ====================== Synchrony statistic =============================
@dataclass
class SynchronyStatistic:
    """Weighted phase-locking over sliding windows.

    Returns both the mean over channel pairs (v1's statistic) and the
    largest-eigenvalue statistic, which is linear rather than quadratic in
    the fraction of channels that are actually locked -- see note 4."""

    win: int = 256
    step: int = 64
    tau: float = 0.8

    def __call__(self, z: np.ndarray) -> dict:
        T, K = z.shape
        if T < self.win:
            raise ValueError(
                f"signal has {T} samples but win={self.win}; no complete "
                "window exists. Shorten win or supply a longer record."
            )
        if K < 2:
            raise ValueError(f"need at least 2 channels, got {K}")

        centers, ends, mean_pair, eig = [], [], [], []
        for s in range(0, T - self.win + 1, self.step):
            seg = z[s:s + self.win]
            w = zero_clamp(np.abs(seg), tau=self.tau)          # (win, K)
            u = w * np.exp(1j * np.angle(seg))                 # (win, K)
            num = u.conj().T @ u                               # (K, K)
            den = (w.T @ w) + 1e-12
            M = num / den                                      # Hermitian, unit diag
            iu = np.triu_indices(K, 1)
            mean_pair.append(np.abs(M[iu]).mean())
            # M is Hermitian by construction; eigvalsh is the right routine
            # and is what makes lam_max real without a discarded imaginary part.
            lam = np.linalg.eigvalsh((M + M.conj().T) / 2).max()
            eig.append(float((lam - 1.0) / (K - 1.0)))
            centers.append(s + self.win // 2)
            ends.append(s + self.win - 1)
        return {
            "centers": np.asarray(centers),
            "ends": np.asarray(ends),
            "mean_pair_plv": np.asarray(mean_pair),
            "eig_sync": np.asarray(eig),
        }


# ====================== CUSUM Detector ==================================
@dataclass
class CUSUMDetector:
    """One-sided CUSUM on a STANDARDISED statistic.

    kappa and eta are in units of the statistic's own robust sigma, not in
    PLV units, which is what decouples them from window length and
    smoothing constant (note 1). Defaults give roughly 0-1 false alarms per
    ~120-window record on stationary noise; raise both to trade sensitivity
    for a lower false-alarm rate.
    """

    kappa: float = 2.0   # slack, in robust sigmas
    eta: float = 6.0     # decision threshold, in cumulative sigmas
    mu0: Optional[float] = None     # null location; None -> estimate
    sigma0: Optional[float] = None  # null scale;    None -> estimate

    @staticmethod
    def _robust_moments(x: np.ndarray) -> Tuple[float, float]:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        # A degenerate MAD (constant input) must not divide by ~0.
        return med, max(mad, 1e-12)

    def run(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        if self.mu0 is None or self.sigma0 is None:
            mu0, sigma0 = self._robust_moments(x)
        else:
            mu0, sigma0 = self.mu0, self.sigma0
        u = (x - mu0) / sigma0
        S = np.zeros_like(u)
        alarm = None
        for t in range(1, len(u)):
            S[t] = max(0.0, S[t - 1] + u[t] - self.kappa)
            if alarm is None and S[t] > self.eta:
                alarm = t
        return {"S": S, "alarm_index": alarm, "u": u,
                "mu0": mu0, "sigma0": sigma0}


# ====================== Main Detector Engine ============================
@dataclass
class EITDetector:
    """EIT + phase-locking + standardised CUSUM.

    Args:
        fs: sample rate, used only to convert indices to seconds.
        eit_alpha, sync_win, sync_step, sync_tau: as v1.
        statistic: "eig" (default, sensitive to partial synchrony) or
            "mean_pair" (v1's statistic, kept for comparison).
        mode: "offline" self-calibrates the null from the whole record;
            "online" requires calibrate() first and is strictly causal.
        analytic: if True, real input is converted to its analytic signal
            via the Hilbert transform instead of being rejected.
    """

    fs: float = 1000.0
    eit_alpha: float = 0.75
    sync_win: int = 256
    sync_step: int = 64
    sync_tau: float = 0.8
    cusum_kappa: float = 2.0
    cusum_eta: float = 6.0
    statistic: str = "eig"
    mode: str = "offline"
    analytic: bool = False
    _null: Optional[Tuple[float, float]] = field(default=None, repr=False)

    def __post_init__(self):
        if self.statistic not in ("eig", "mean_pair"):
            raise ValueError(f"unknown statistic {self.statistic!r}")
        if self.mode not in ("offline", "online"):
            raise ValueError(f"unknown mode {self.mode!r}")
        self.eit = EITAccumulator(alpha=self.eit_alpha)
        self.sync = SynchronyStatistic(win=self.sync_win, step=self.sync_step,
                                       tau=self.sync_tau)

    # ---------------------------------------------------------------- input
    def _prepare(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        if z.ndim != 2:
            raise ValueError(f"expected (T, K), got shape {z.shape}")
        if not np.iscomplexobj(z):
            if not self.analytic:
                raise TypeError(
                    "input is real-valued. np.angle() of a real signal is only "
                    "ever 0 or pi, so phase synchrony would be measuring sign "
                    "agreement, not phase, and would return a plausible-looking "
                    "but meaningless number. Pass analytic=True to apply the "
                    "Hilbert transform, or supply a complex analytic signal."
                )
            z = hilbert(np.asarray(z, dtype=float), axis=0)
        return z

    def _series(self, z: np.ndarray) -> dict:
        out = self.sync(self.eit.filter(self._prepare(z)))
        out["rho"] = out["eig_sync"] if self.statistic == "eig" else out["mean_pair_plv"]
        return out

    # ---------------------------------------------------------------- api
    def calibrate(self, reference: np.ndarray) -> "EITDetector":
        """Estimate the null location/scale from a reference (quiet) record.
        Required for mode='online'."""
        rho = self._series(reference)["rho"]
        self._null = CUSUMDetector._robust_moments(rho)
        return self

    def detect(self, z: np.ndarray) -> dict:
        s = self._series(z)
        if self.mode == "online":
            if self._null is None:
                raise RuntimeError(
                    "mode='online' needs a null estimate before it can decide "
                    "anything. Call detector.calibrate(quiet_reference) first, "
                    "or use mode='offline' to self-calibrate on this record "
                    "(not causal -- it looks at the whole signal)."
                )
            mu0, sigma0 = self._null
        else:
            mu0 = sigma0 = None
        c = CUSUMDetector(kappa=self.cusum_kappa, eta=self.cusum_eta,
                          mu0=mu0, sigma0=sigma0).run(s["rho"])
        i = c["alarm_index"]
        return {
            "times": s["centers"],
            "synchrony": s["rho"],
            "mean_pair_plv": s["mean_pair_plv"],
            "eig_sync": s["eig_sync"],
            "cusum_stat": c["S"],
            "standardized": c["u"],
            "null_mu": c["mu0"],
            "null_sigma": c["sigma0"],
            "alarm_index": i,
            # window END, not centre: the PLV cannot be computed until the
            # last sample of the window exists (note 3).
            "alarm_time": float(s["ends"][i] / self.fs) if i is not None else None,
            "alarm_window_center": float(s["centers"][i] / self.fs) if i is not None else None,
            "max_synchrony": float(s["rho"].max()) if len(s["rho"]) else 0.0,
        }
