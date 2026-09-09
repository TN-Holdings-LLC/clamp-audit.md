"""The pasted 'corrected' detector, verbatim, for head-to-head testing."""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


def zero_clamp(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(1.0 + (x / tau) ** 2)


@dataclass
class EITAccumulator:
    alpha: float = 0.8
    fs: float = 1000.0

    def __post_init__(self):
        self.beta = 1.0 - np.exp(-self.alpha)

    def filter(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        T, K = z.shape
        y = np.zeros((T, K), dtype=np.complex128)
        y[0] = z[0]
        for t in range(1, T):
            y[t] = (1 - self.beta) * y[t - 1] + self.beta * z[t]
        return y


def psf_zero_synchrony(z_smooth, win=256, step=64, tau=1.0):
    T, K = z_smooth.shape
    iu = np.triu_indices(K, 1)
    times, rho = [], []
    for start in range(0, T - win + 1, step):
        segment = z_smooth[start:start + win]
        phi = np.angle(segment)
        amp = np.abs(segment)
        dphi = phi[:, iu[0]] - phi[:, iu[1]]
        pair_amp = np.sqrt(amp[:, iu[0]] * amp[:, iu[1]])
        w = zero_clamp(pair_amp, tau=tau)
        num = np.sum(w * np.exp(1j * dphi), axis=0)
        den = np.sum(w, axis=0) + 1e-12
        plv_per_pair = np.abs(num / den)
        rho.append(plv_per_pair.mean())
        times.append(start + win // 2)
    return np.array(times), np.array(rho)


@dataclass
class CUSUMDetector:
    mu0: float = 0.0
    kappa: float = 0.07
    eta: float = 0.1

    def run(self, x):
        S = np.zeros_like(x, dtype=float)
        alarm_idx = None
        for t in range(1, len(x)):
            S[t] = max(0.0, S[t - 1] + x[t] - self.mu0 - self.kappa)
            if alarm_idx is None and S[t] > self.eta:
                alarm_idx = t
        return S, alarm_idx


@dataclass
class EITDetector:
    fs: float = 1000.0
    eit_alpha: float = 0.75
    sync_win: int = 256
    sync_step: int = 64
    sync_tau: float = 0.8
    cusum_kappa: float = 0.07
    cusum_eta: float = 0.1

    def __post_init__(self):
        self.eit = EITAccumulator(alpha=self.eit_alpha, fs=self.fs)
        self.cusum = CUSUMDetector(kappa=self.cusum_kappa, eta=self.cusum_eta)

    def detect(self, z):
        z_smooth = self.eit.filter(z)
        times, sync = psf_zero_synchrony(z_smooth, win=self.sync_win,
                                         step=self.sync_step, tau=self.sync_tau)
        cusum_stat, alarm_idx = self.cusum.run(sync)
        return {
            "times": times, "synchrony": sync, "cusum_stat": cusum_stat,
            "alarm_index": alarm_idx,
            "alarm_time": times[alarm_idx] / self.fs if alarm_idx is not None else None,
            "max_synchrony": float(sync.max()) if len(sync) > 0 else 0.0,
        }
