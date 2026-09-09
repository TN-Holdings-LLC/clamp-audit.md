# The pasted "corrected" module, verbatim (import-safe subset).
import numpy as np, pandas as pd
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

@dataclass
class PhaseNoiseConfig:
    fmin: float = 1.0
    fmax: float = 50e3
    tau: float = 1.0
    detrend: bool = True
    window: str = 'hann'
    nperseg: Optional[int] = None
    despike: bool = True
    hampel_window: int = 41
    hampel_n_sigmas: float = 6.0

def zero_clamp(x, tau=1.0):
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(1.0 + (x / tau) ** 2)

def hampel_despike(x, window=11, n_sigmas=5.0):
    x = np.asarray(x, dtype=float)
    if window < 3: window = 3
    if window % 2 == 0: window += 1
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * mad
    outliers = (abs_dev > threshold).to_numpy()
    xd = x.copy(); xd[outliers] = med.to_numpy()[outliers]
    return xd, outliers

def _welch_band_rms(x, fs, config):
    nperseg = config.nperseg or min(len(x), 2 ** 14)
    f, Pxx = signal.welch(x, fs=fs, window=config.window,
                          nperseg=nperseg, noverlap=nperseg // 2, scaling='density')
    band = (f >= config.fmin) & (f <= config.fmax)
    if not np.any(band): raise ValueError("No frequency content in the specified band.")
    return float(np.sqrt(_trapezoid(Pxx[band], f[band]))), f, Pxx

def rms_from_array(t, phi, config=None):
    """Same body as rms_phase_from_timeseries but without the CSV round-trip."""
    if config is None: config = PhaseNoiseConfig()
    dt = np.median(np.diff(t)); fs = 1.0/dt if dt > 0 else 1e6
    phi_d = signal.detrend(phi, type='linear') if config.detrend else phi
    n_out = 0
    if config.despike:
        phi_c, mask = hampel_despike(phi_d, config.hampel_window, config.hampel_n_sigmas)
        n_out = int(mask.sum())
    else:
        phi_c = phi_d
    sig, f, Pxx = _welch_band_rms(phi_c, fs, config)
    return {"rms_rad": sig, "fs": float(fs), "n_outliers_removed": n_out}

def rms_from_psd_arrays(f, Sphi, config=None):
    if config is None: config = PhaseNoiseConfig()
    band = (f >= config.fmin) & (f <= config.fmax)
    if not np.any(band): raise ValueError("No frequency content in the specified band.")
    return float(np.sqrt(_trapezoid(Sphi[band], f[band])))
