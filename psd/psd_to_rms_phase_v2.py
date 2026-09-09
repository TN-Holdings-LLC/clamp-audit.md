# -*- coding: utf-8 -*-
"""
psd_to_rms_phase_v2.py — RMS phase noise from optical telemetry
================================================================

Run against the same stack the previous version names: numpy 2.4.4,
pandas 3.0.2, scipy 1.17.1.

WHAT THE PREVIOUS VERSION GOT RIGHT — all of it reproduces:

  * ``np.trapz`` really is gone on numpy 2.4.4 (``np.trapezoid`` exists);
    the shim is needed and works.
  * The blanket ``zero_clamp`` bias is real: on homogeneous N(0, 2.0) phase
    noise the old path reported 0.7495 rad against a true 2.0 rad, i.e.
    −62.5% (the writeup said −62.7%).
  * The Hampel tuning claims are exact. Window 11 / 5σ false-flags 25.0% of
    a smooth 50 Hz tone; window 41 / 6σ false-flags 0.0%, and recovers the
    homogeneous-noise RMS as 1.9989 rad with 1 sample flagged out of 65536.
  * And the replacement step does NOT reintroduce the distortion it was
    fixing, which the writeup did not check but should have. Third-harmonic
    power relative to the fundamental, on a 50 Hz / 3 rad tone:

        Hampel, no spikes present ....... 1.02e-06
        Hampel, 20 spikes removed ....... 1.13e-06
        old zero_clamp .................. 5.75e-02   (~5e4x worse)

    So switching to Hampel was the right call, and it is verified here
    rather than asserted.

WHAT IS FIXED HERE.

 1. ``fmin`` IS NOT HONOURED, AND THE ANSWER DEPENDS ON A PARAMETER THE
    CALLER NEVER SETS.  This is the serious one. ``nperseg`` defaults to
    ``min(len(x), 2**14)``, so the Welch frequency resolution is
    ``df = fs/nperseg`` and the integration band starts at the first *bin*
    at or above ``fmin`` — which can be far above it, silently.

    Phase noise is dominated by low frequencies, so this is not a rounding
    error. For S(f) = A/f², the fraction of the [1 Hz, fs/2] power that
    falls below the first bin:

        fs = 2 kHz    df = 0.12 Hz  ->   0.00% lost
        fs = 100 kHz  df = 6.10 Hz  ->  83.62% lost
        fs = 1 MHz    df = 61.0 Hz  ->  98.36% lost

    Measured end-to-end on a 1/f² random walk at fs = 1 MHz, the reported
    RMS moved from 7.9380 rad to 11.4854 rad (+45%) purely by changing
    ``nperseg`` from 2^14 to 2^16. The requested ``fmin=1.0`` had no effect
    on either number.

    Fixed: ``nperseg`` is chosen from the requested band
    (``nperseg >= fs/fmin``, capped by the record length), the achieved
    resolution is returned in the result dict as ``f_resolution`` and
    ``band_actual``, and the function raises if the record is too short to
    resolve ``fmin`` at all instead of quietly answering a different
    question. Band edges are interpolated so the integral runs from exactly
    ``fmin`` to exactly ``fmax``.

 2. LINEAR ``trapezoid`` ON A LOG-SPACED PSD OVERSHOOTS, BY A LOT.
    Measured PSDs are almost always supplied on a log frequency grid, and
    ``rms_phase_from_psd`` integrated them with linear-in-f trapezoids.
    Against S(f) = A/f² from 1 Hz to 100 kHz, where the exact integral is
    known:

        3 pts/decade  -> +30.93%
        5 pts/decade  -> +10.79%
       10 pts/decade  ->  +2.66%
       20 pts/decade  ->  +0.66%

    Fixed with piecewise power-law integration: between two grid points a
    phase-noise PSD is a straight line in log-log, so integrate it as one,
    which is exact for that shape instead of biased. Falls back to a
    trapezoid on any interval with a non-positive value.

 3. dBc/Hz INPUT SILENTLY PRODUCED nan.  Phase noise is normally tabulated
    as script-L in dBc/Hz, not as linear rad²/Hz. Feeding a realistic
    −80 dBc/Hz @ 1 Hz, −20 dB/decade table straight in returned ``nan``
    (negative integrand under a square root) with ``status: "OK"``.
    Correctly converted the same table gives 1.415e-04 rad. Fixed with an
    explicit ``units`` argument ("rad2/Hz", "dBc/Hz", "dBrad2/Hz") plus a
    guard that refuses all-negative data labelled as linear rather than
    returning nan.

 4. NON-UNIFORM TIMESTAMPS WERE ACCEPTED SILENTLY.  ``fs`` came from
    ``median(diff(t))`` with no check, and Welch assumes uniform sampling.
    On jittered timestamps the old code reported fs = 2879 Hz for a true
    mean rate of 2000 Hz and an RMS of 1.6608 rad against a true 2.0 rad —
    a 17% error, reported as ``status: "OK"``. Now checked, with a
    tolerance, and raised.

 5. BROKEN TIMESTAMPS FELL BACK TO fs = 1e6.  An all-zero time column gave
    ``dt = 0`` and the code substituted 1 MHz, returning 0.0656 rad and
    ``status: "OK"``. An arbitrary invented sample rate is not a sensible
    default for a physical measurement; it now raises.

 6. NaN/Inf FAILED DEEP INSIDE scipy.  A single NaN produced
    ``ValueError: array must not contain infs or NaNs`` from
    ``scipy.linalg.lstsq`` — loud, but pointing at scipy internals rather
    than at the caller's data. Now checked up front with the index named,
    and optionally interpolated over via ``on_nonfinite="interpolate"``.

 7. ``status`` WAS THE STRING "OK", UNCONDITIONALLY.  It was set to "OK"
    in every return path, including the ones above that returned nonsense,
    so it could never signal anything. It now carries the real warnings, and
    the caller can check ``result["warnings"]``.

KEPT: ``zero_clamp`` (a fine saturating function, still not used in the RMS
path), the Hampel defaults 41 / 6.0 with their tuning rationale, and the
numpy trapezoid shim.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

__all__ = [
    "PhaseNoiseConfig", "zero_clamp", "hampel_despike",
    "integrate_psd", "rms_phase_from_timeseries", "rms_phase_from_psd",
]

# numpy >= 2.0 renamed trapz -> trapezoid and later removed trapz.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclass
class PhaseNoiseConfig:
    fmin: float = 1.0            # [Hz] lower integration limit, honoured exactly
    fmax: float = 50e3           # [Hz] upper integration limit, honoured exactly
    tau: float = 1.0             # zero_clamp() only; unused by the RMS path
    detrend: bool = True
    window: str = "hann"
    nperseg: Optional[int] = None  # None -> derived from fmin (see note 1)
    despike: bool = True
    hampel_window: int = 41
    hampel_n_sigmas: float = 6.0
    # Uniformity tolerance for the time column, as a fraction of median dt.
    uniform_rtol: float = 0.01
    # "raise" | "interpolate" for NaN/Inf samples.
    on_nonfinite: str = "raise"


# ---------------------------------------------------------------- utilities
def zero_clamp(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Smooth odd saturating map, |x| -> tau as |x| grows.

    Perfectly well behaved in itself, and NOT used by the RMS pipeline: as
    a blanket pre-PSD step it deflated genuine 2.0 rad noise to 0.75 rad
    and injected a 3rd harmonic at 5.75e-02 of the fundamental. Kept for
    callers who want a bounded transform for some other purpose.
    """
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(1.0 + (x / tau) ** 2)


def hampel_despike(x: np.ndarray, window: int = 41,
                   n_sigmas: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling-median / MAD outlier replacement. Local, not global.

    Defaults verified: 0.0% false-flag rate on a smooth 50 Hz / 3 rad tone,
    20/20 injected spikes caught, and no measurable spectral distortion
    (3rd-harmonic ratio 1.13e-06 after removing 20 spikes vs 1.02e-06 with
    none present).
    """
    x = np.asarray(x, dtype=float)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * mad
    # A degenerate MAD (locally constant data) would make the threshold 0 and
    # flag every non-identical sample; treat that window as having no outliers.
    outliers = ((abs_dev > threshold) & (threshold > 0)).to_numpy()
    out = x.copy()
    out[outliers] = med.to_numpy()[outliers]
    return out, outliers


def _check_finite(x: np.ndarray, policy: str, warns: List[str]) -> np.ndarray:
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    idx = np.flatnonzero(bad)
    if policy == "raise":
        raise ValueError(
            f"input contains {bad.sum()} non-finite sample(s); first at index "
            f"{idx[0]}. Clean the data, or pass "
            f'on_nonfinite="interpolate" to fill them by linear interpolation.'
        )
    x = x.copy()
    good = ~bad
    if good.sum() < 2:
        raise ValueError("almost every sample is non-finite; nothing to analyse.")
    x[bad] = np.interp(idx, np.flatnonzero(good), x[good])
    warns.append(f"interpolated {int(bad.sum())} non-finite sample(s)")
    return x


def _sample_rate(t: np.ndarray, rtol: float) -> float:
    if t.size < 2:
        raise ValueError("need at least two samples to infer a sample rate.")
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)):
        raise ValueError("time column contains non-finite values.")
    med = float(np.median(dt))
    if med <= 0:
        raise ValueError(
            "time column is not increasing (median dt <= 0), so no sample rate "
            "can be inferred. The previous version silently substituted 1 MHz "
            "here and reported status OK."
        )
    spread = float(np.max(np.abs(dt - med))) / med
    if spread > rtol:
        raise ValueError(
            f"timestamps are not uniform: max deviation {100*spread:.2f}% of the "
            f"median interval, tolerance {100*rtol:.2f}%. Welch's method assumes "
            "uniform sampling; on jittered input the previous version reported a "
            "17% RMS error as status OK. Resample onto a uniform grid, or raise "
            "config.uniform_rtol if this jitter is genuinely negligible."
        )
    return 1.0 / med


# ---------------------------------------------------------------- integration
def integrate_psd(f: np.ndarray, S: np.ndarray, fmin: float, fmax: float,
                  loglog: bool = True) -> float:
    """Integrate S(f) over [fmin, fmax], honouring the endpoints exactly.

    With ``loglog=True`` each interval is treated as a power law, which is
    exact for the piecewise-straight-in-log-log shape real phase-noise PSDs
    have, and removes the +30.9% (3 pts/decade) bias linear trapezoids gave.
    """
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=float)
    order = np.argsort(f)
    f, S = f[order], S[order]

    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must exceed fmin ({fmin}).")
    if fmin < f[0] or fmax > f[-1]:
        raise ValueError(
            f"requested band [{fmin:g}, {fmax:g}] Hz is not covered by the data "
            f"[{f[0]:g}, {f[-1]:g}] Hz."
        )

    # Insert exact band edges so the integral starts and ends where asked,
    # rather than at the nearest available grid point.
    def _interp(x0: float) -> float:
        return float(np.interp(x0, f, S))

    inner = (f > fmin) & (f < fmax)
    fg = np.concatenate(([fmin], f[inner], [fmax]))
    Sg = np.concatenate(([_interp(fmin)], S[inner], [_interp(fmax)]))

    if not loglog:
        return float(_trapezoid(Sg, fg))

    total = 0.0
    for i in range(len(fg) - 1):
        f1, f2, s1, s2 = fg[i], fg[i + 1], Sg[i], Sg[i + 1]
        if f2 <= f1:
            continue
        if s1 <= 0 or s2 <= 0 or f1 <= 0:
            total += 0.5 * (s1 + s2) * (f2 - f1)      # trapezoid fallback
            continue
        p = np.log(s2 / s1) / np.log(f2 / f1)
        if abs(p + 1.0) < 1e-12:
            total += s1 * f1 * np.log(f2 / f1)
        else:
            total += (s2 * f2 - s1 * f1) / (p + 1.0)
    return float(total)


def _choose_nperseg(n: int, fs: float, fmin: float,
                    requested: Optional[int], warns: List[str]) -> int:
    """Pick a segment length that can actually resolve fmin."""
    if requested is not None:
        nps = min(int(requested), n)
        if fs / nps > fmin:
            warns.append(
                f"nperseg={nps} gives df={fs/nps:.4g} Hz > fmin={fmin:g} Hz; "
                "the band below df is not measured"
            )
        return nps
    need = int(np.ceil(fs / fmin))
    if need > n:
        raise ValueError(
            f"record is too short to resolve fmin={fmin:g} Hz: it needs at least "
            f"{need} samples at fs={fs:.6g} Hz but has {n} "
            f"({n/fs:.4g} s of data, {fs/n:.4g} Hz resolution). Record longer, "
            "or raise fmin. The previous version answered anyway, starting the "
            "integral at fs/nperseg instead of at fmin."
        )
    # A power of two >= need, capped by the record, keeps Welch efficient
    # while leaving at least a couple of averaging segments where possible.
    nps = int(2 ** np.ceil(np.log2(need)))
    return min(nps, n)


# ---------------------------------------------------------------- public API
def rms_phase_from_timeseries(csv_path: str,
                              config: Optional[PhaseNoiseConfig] = None,
                              _arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None
                              ) -> Dict:
    """RMS phase noise from a raw (t, phi) time series.

    ``_arrays`` lets callers and tests pass (t, phi) directly instead of a CSV.
    """
    cfg = config or PhaseNoiseConfig()
    warns: List[str] = []

    if _arrays is not None:
        t, phi = (np.asarray(a, dtype=float) for a in _arrays)
    else:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"expected at least 2 columns (t, phi), got {df.shape[1]}")
        t = df.iloc[:, 0].to_numpy(dtype=float)
        phi = df.iloc[:, 1].to_numpy(dtype=float)

    phi = _check_finite(phi, cfg.on_nonfinite, warns)
    fs = _sample_rate(t, cfg.uniform_rtol)

    nyquist = fs / 2.0
    fmax = min(cfg.fmax, nyquist)
    # Only report a reduction that is real, not a float-comparison artifact
    # of fmax already sitting exactly at Nyquist.
    if cfg.fmax - fmax > 1e-9 * max(1.0, cfg.fmax):
        warns.append(f"fmax reduced from {cfg.fmax:g} to Nyquist {fmax:g} Hz")
    else:
        fmax = min(cfg.fmax, nyquist * (1 - 1e-12))
    if fmax <= cfg.fmin:
        raise ValueError(
            f"fmin={cfg.fmin:g} Hz is at or above Nyquist ({fs/2:g} Hz)."
        )

    phi_d = signal.detrend(phi, type="linear") if cfg.detrend else phi

    n_out = 0
    if cfg.despike:
        phi_d, mask = hampel_despike(phi_d, cfg.hampel_window, cfg.hampel_n_sigmas)
        n_out = int(mask.sum())
        if n_out > 0.01 * phi_d.size:
            warns.append(
                f"despiker replaced {100*n_out/phi_d.size:.1f}% of samples; that is "
                "high enough to suspect it is reshaping the signal, not removing "
                "spikes — check hampel_window/hampel_n_sigmas"
            )

    nperseg = _choose_nperseg(phi_d.size, fs, cfg.fmin, cfg.nperseg, warns)
    f, Pxx = signal.welch(phi_d, fs=fs, window=cfg.window, nperseg=nperseg,
                          noverlap=nperseg // 2, scaling="density")

    var = integrate_psd(f, Pxx, cfg.fmin, fmax, loglog=False)
    sigma = float(np.sqrt(max(var, 0.0)))

    return {
        "rms_rad": sigma,
        "rms_deg": float(np.degrees(sigma)),
        "fs": float(fs),
        "fmin": float(cfg.fmin),
        "fmax": float(fmax),
        "nperseg": int(nperseg),
        "f_resolution": float(fs / nperseg),
        "band_actual": (float(cfg.fmin), float(fmax)),
        "n_samples": int(phi.size),
        "n_outliers_removed": n_out,
        "warnings": warns,
        "status": "OK" if not warns else "OK_WITH_WARNINGS",
    }


def rms_phase_from_psd(csv_path: str,
                       config: Optional[PhaseNoiseConfig] = None,
                       units: str = "rad2/Hz",
                       loglog: bool = True,
                       _arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None
                       ) -> Dict:
    """RMS phase noise from a tabulated PSD.

    units:
      "rad2/Hz"    linear one-sided phase PSD S_phi(f)   (default)
      "dBc/Hz"     script-L(f); converted as S_phi = 2 * 10**(L/10)
      "dBrad2/Hz"  S_phi in dB;  converted as S_phi = 10**(value/10)
    """
    cfg = config or PhaseNoiseConfig()
    warns: List[str] = []

    if _arrays is not None:
        f, S = (np.asarray(a, dtype=float) for a in _arrays)
    else:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"expected at least 2 columns (f, S), got {df.shape[1]}")
        f = df.iloc[:, 0].to_numpy(dtype=float)
        S = df.iloc[:, 1].to_numpy(dtype=float)

    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(S)):
        raise ValueError("PSD table contains non-finite values.")

    if units == "dBc/Hz":
        S = 2.0 * 10.0 ** (S / 10.0)
    elif units == "dBrad2/Hz":
        S = 10.0 ** (S / 10.0)
    elif units == "rad2/Hz":
        if np.all(S < 0):
            raise ValueError(
                "every PSD value is negative, which a linear rad^2/Hz spectrum "
                "cannot be — this table is almost certainly in dB. Pass "
                'units="dBc/Hz" (script-L) or units="dBrad2/Hz". The previous '
                "version integrated it as-is and returned nan with status OK."
            )
        if np.any(S < 0):
            raise ValueError("PSD contains negative values; check units.")
    else:
        raise ValueError(f'unknown units {units!r}')

    fmax = min(cfg.fmax, float(f.max()))
    if fmax < cfg.fmax:
        warns.append(f"fmax reduced from {cfg.fmax:g} to {fmax:g} Hz (end of table)")

    spacing = np.diff(np.log10(f[f > 0]))
    if loglog and spacing.size and np.ptp(spacing) < 0.1 * np.median(spacing) + 1e-12:
        ppd = 1.0 / np.median(spacing)
        if ppd < 8:
            warns.append(
                f"log grid is coarse ({ppd:.1f} points/decade); power-law "
                "integration is in use, but consider a finer table"
            )

    var = integrate_psd(f, S, cfg.fmin, fmax, loglog=loglog)
    sigma = float(np.sqrt(max(var, 0.0)))
    return {
        "rms_rad": sigma,
        "rms_deg": float(np.degrees(sigma)),
        "fmin": float(cfg.fmin),
        "fmax": float(fmax),
        "units_in": units,
        "integration": "power-law (log-log)" if loglog else "trapezoid (linear)",
        "warnings": warns,
        "status": "OK" if not warns else "OK_WITH_WARNINGS",
    }


if __name__ == "__main__":
    print("psd_to_rms_phase_v2 — RMS phase noise from optical telemetry")
    print("=" * 62)
    print("Module loaded. See test_psd_to_rms_phase_v2.py for the measurements")
    print("behind every claim in the module docstring.")
