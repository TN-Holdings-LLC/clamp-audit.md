# -*- coding: utf-8 -*-
"""
Quantum Kuramoto: Generates 2D phase maps of the order parameter r(K, T_phi).

You can run this code without qiskit / qiskit-aer using `--backend classical` or `--replot`.

Usage:
    python quantum_kuramoto_phase_map.py                            # Exact density matrix simulation
    python quantum_kuramoto_phase_map.py --backend shots        # With shot noise
    python quantum_kuramoto_phase_map.py --backend classical    # Classical Kuramoto (for comparison)
    python quantum_kuramoto_phase_map.py --replot results.npz   # Plot only from saved data
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Definition of the oscillator model."""

    n_qubits: int = 4
    t_steps: int = 15
    dt: float = 0.1
    topology: str = "chain"  # "chain" | "ring"
    omegas: tuple[float, ...] = (0.0, 0.1, -0.05, 0.05)
    phases: tuple[float, ...] = (0.0, 0.6, 1.2, -0.5)

    def resolved(self) -> "ModelConfig":
        """Deterministically pads or adjusts omegas/phases if their lengths mismatch n_qubits."""
        n = self.n_qubits
        omegas = self.omegas
        phases = self.phases
        if len(omegas) != n:
            omegas = tuple(np.round(np.linspace(-0.1, 0.1, n), 4))
        if len(phases) != n:
            phases = tuple(np.round(np.linspace(0.0, 2 * np.pi, n, endpoint=False), 4))
        return ModelConfig(n, self.t_steps, self.dt, self.topology, omegas, phases)

    @property
    def total_time(self) -> float:
        return self.t_steps * self.dt


@dataclass
class NoiseConfig:
    """Hardware noise parameters (in seconds)."""

    T1: float = 80e-6
    T2: float = 60e-6
    tg_1q: float = 35e-9
    tg_2q: float = 300e-9
    excited_state_population: float = 0.0

    def validate(self) -> None:
        if self.T2 > 2 * self.T1:
            raise ValueError(f"T2 <= 2*T1 is required (T1={self.T1}, T2={self.T2})")


def coupling_pairs(n: int, topology: str) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(n - 1)]
    if topology == "ring" and n > 2:
        pairs.append((n - 1, 0))
    elif topology not in ("chain", "ring"):
        raise ValueError(f"Unknown topology: {topology}")
    return pairs


# ---------------------------------------------------------------------------
# Circuit Construction
# ---------------------------------------------------------------------------
def build_kuramoto_circuit(cfg: ModelConfig, K: float):
    """
    Trotterized XY-coupled Kuramoto circuit.

        H = Σ_i (ω_i/2) Z_i + K Σ_<ij> (X_i X_j + Y_i Y_j)

    Initial states are initialized on the equator at phase θ_i.
    RY(π/2) brings the state to the equator, followed by RZ(θ) to assign azimuth angle.
    """
    from qiskit import QuantumCircuit

    cfg = cfg.resolved()
    qc = QuantumCircuit(cfg.n_qubits)

    for q, theta in enumerate(cfg.phases):
        qc.ry(np.pi / 2, q)
        qc.rz(theta, q)

    pairs = coupling_pairs(cfg.n_qubits, cfg.topology)
    for _ in range(cfg.t_steps):
        for q, omega in enumerate(cfg.omegas):
            qc.rz(omega * cfg.dt, q)  # exp(-i ω dt Z/2)
        for i, j in pairs:
            qc.rxx(2 * K * cfg.dt, i, j)  # exp(-i K dt XX)
            qc.ryy(2 * K * cfg.dt, i, j)

    return qc


def make_noise_model(noise: NoiseConfig, tphi: float | None):
    """Noise model combining thermal relaxation and additional phase damping (T_phi)."""
    from qiskit_aer.noise import NoiseModel, phase_damping_error, thermal_relaxation_error

    noise.validate()
    nm = NoiseModel()

    e1 = thermal_relaxation_error(
        noise.T1, noise.T2, noise.tg_1q, noise.excited_state_population
    )
    e2_single = thermal_relaxation_error(
        noise.T1, noise.T2, noise.tg_2q, noise.excited_state_population
    )
    if tphi is not None and np.isfinite(tphi):
        e1 = e1.compose(phase_damping_error(1.0 - np.exp(-noise.tg_1q / tphi)))
        e2_single = e2_single.compose(phase_damping_error(1.0 - np.exp(-noise.tg_2q / tphi)))

    nm.add_all_qubit_quantum_error(e1, ["rx", "ry", "rz", "sx", "x", "h", "s", "sdg", "id"])
    nm.add_all_qubit_quantum_error(e2_single.tensor(e2_single), ["rxx", "ryy", "cx", "ecr"])
    return nm


# ---------------------------------------------------------------------------
# Order Parameter
# ---------------------------------------------------------------------------
def order_parameter(bloch_x: np.ndarray, bloch_y: np.ndarray, normalize: bool = False) -> tuple[float, float]:
    """
    Computes the order parameter from the equatorial components of Bloch vectors.

    Let z_i = <X_i> + i<Y_i>:
        r_raw  = |Σ z_i| / N       Naturally decays to 0 as decoherence shrinks amplitudes
        r_norm = |Σ z_i/|z_i|| / N Focuses purely on phases (closer to classical Kuramoto definition)

    Returns (r, coherence). Coherence = mean(|z_i|) indicates whether phase information is retained,
    allowing isolation of whether r dropped due to phase dispersion or amplitude decay.
    """
    z = np.asarray(bloch_x, dtype=float) + 1j * np.asarray(bloch_y, dtype=float)
    coherence = float(np.mean(np.abs(z)))

    if normalize:
        mag = np.abs(z)
        tol = 1e-9
        if np.any(mag < tol):
            return float("nan"), coherence
        z = z / mag

    return float(np.abs(np.mean(z))), coherence


# ---------------------------------------------------------------------------
# Backend 1: Density Matrix (Exact, no shot noise)
# ---------------------------------------------------------------------------
def bloch_from_density_matrix(qc, simulator, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Extracts <X_i> and <Y_i> for all qubits precisely in a single simulation run."""
    from qiskit.quantum_info import Pauli

    circ = qc.copy()
    circ.save_density_matrix(label="rho")
    rho = simulator.run(circ).result().data(0)["rho"]

    def pauli_label(axis: str, target: int) -> str:
        return "".join(axis if q == target else "I" for q in reversed(range(n_qubits)))

    xs = [float(np.real(rho.expectation_value(Pauli(pauli_label("X", i))))) for i in range(n_qubits)]
    ys = [float(np.real(rho.expectation_value(Pauli(pauli_label("Y", i))))) for i in range(n_qubits)]
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Backend 2: Shot Measurements (Includes realistic statistical errors)
# ---------------------------------------------------------------------------
def bloch_from_shots(qc, simulator, n_qubits: int, shots: int) -> tuple[np.ndarray, np.ndarray]:
    """Measures X and Y bases across two separate circuits to estimate <X_i> and <Y_i>."""
    values = {}
    for basis in ("X", "Y"):
        circ = qc.copy()
        for q in range(n_qubits):
            if basis == "X":
                circ.h(q)
            else:
                circ.sdg(q)
                circ.h(q)
        circ.measure_all()

        counts = simulator.run(circ, shots=shots).result().get_counts()
        exp = np.zeros(n_qubits)
        for bitstr, count in counts.items():
            bits = bitstr.replace(" ", "")[::-1]
            signs = np.array([1.0 if bits[i] == "0" else -1.0 for i in range(n_qubits)])
            exp += signs * count
        values[basis] = exp / shots

    return values["X"], values["Y"]


# ---------------------------------------------------------------------------
# Backend 3: Classical Kuramoto (Sanity check without Qiskit dependency)
# ---------------------------------------------------------------------------
def circuit_duration(cfg: ModelConfig, noise: NoiseConfig) -> float:
    """Estimated circuit execution time [s] based on gate counts and durations."""
    pairs = coupling_pairs(cfg.n_qubits, cfg.topology)
    init = 2 * cfg.n_qubits * noise.tg_1q
    per_step = cfg.n_qubits * noise.tg_1q + 2 * len(pairs) * noise.tg_2q
    return init + cfg.t_steps * per_step


def bloch_classical(cfg: ModelConfig, K: float, tphi_us: float,
                    noise: NoiseConfig | None = None, substeps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates the classical Kuramoto equation dθ_i/dt = ω_i + K Σ_j sin(θ_j - θ_i),
    scaling amplitudes by exp(-t_circuit / T_phi) as a reference trajectory.
    """
    cfg = cfg.resolved()
    noise = noise or NoiseConfig()
    theta = np.array(cfg.phases, dtype=float)
    omega = np.array(cfg.omegas, dtype=float)
    pairs = coupling_pairs(cfg.n_qubits, cfg.topology)
    h = cfg.dt / substeps

    for _ in range(cfg.t_steps * substeps):
        d = omega.copy()
        for i, j in pairs:
            s = np.sin(theta[j] - theta[i])
            d[i] += K * s
            d[j] -= K * s
        theta = theta + h * d

    decay = 1.0 if not np.isfinite(tphi_us) else float(
        np.exp(-circuit_duration(cfg, noise) / (tphi_us * 1e-6))
    )
    return decay * np.cos(theta), decay * np.sin(theta)


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------
def run_sweep(
    cfg: ModelConfig,
    K_list: Sequence[float],
    tphi_list_us: Sequence[float],
    backend: str = "density_matrix",
    noise: NoiseConfig | None = None,
    shots: int = 4000,
    seed: int = 1234,
    normalize: bool = False,
) -> dict:
    """Computes r(K, T_phi) and coherence(K, T_phi), returning them in a dictionary."""
    cfg = cfg.resolved()
    noise = noise or NoiseConfig()
    n_rows, n_cols = len(tphi_list_us), len(K_list)
    R = np.full((n_rows, n_cols), np.nan)
    C = np.full((n_rows, n_cols), np.nan)

    if backend == "classical":
        for i, tphi_us in enumerate(tphi_list_us):
            for j, K in enumerate(K_list):
                x, y = bloch_classical(cfg, float(K), float(tphi_us), noise)
                R[i, j], C[i, j] = order_parameter(x, y, normalize)
        return {"R": R, "C": C}

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    base_sim = AerSimulator(method="density_matrix", seed_simulator=seed)
    circuits = {}
    for K in K_list:
        qc = build_kuramoto_circuit(cfg, float(K))
        circuits[float(K)] = transpile(qc, base_sim, optimization_level=0)

    t0 = time.time()
    for i, tphi_us in enumerate(tphi_list_us):
        tphi = None if not np.isfinite(tphi_us) else tphi_us * 1e-6
        nm = make_noise_model(noise, tphi)
        sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed + i)

        for j, K in enumerate(K_list):
            qc = circuits[float(K)]
            if backend == "shots":
                x, y = bloch_from_shots(qc, sim, cfg.n_qubits, shots)
            else:
                x, y = bloch_from_density_matrix(qc, sim, cfg.n_qubits)
            R[i, j], C[i, j] = order_parameter(x, y, normalize)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_rows - i - 1)
        label = "inf" if not np.isfinite(tphi_us) else f"{tphi_us:g} us"
        print(f"  [{i + 1}/{n_rows}] T_phi = {label:>8}  Elapsed {elapsed:5.1f}s  ETA ~ {eta:5.1f}s")

    return {"R": R, "C": C}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_phase_map(
    R: np.ndarray,
    C: np.ndarray,
    K_list: np.ndarray,
    tphi_list_us: np.ndarray,
    r_contour: float = 0.5,
    normalize: bool = False,
    backend: str = "density_matrix",
    save_path: str = "phase_map_r_vs_K_Tphi.png",
) -> None:
    """Saves a two-tier visualization consisting of a heatmap (top) and K-slices (bottom)."""
    K = np.asarray(K_list, dtype=float)
    rows = np.arange(len(tphi_list_us))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 8.5), dpi=150, gridspec_kw={"height_ratios": [1.25, 1]}
    )

    mesh = ax.pcolormesh(K, rows, R, cmap="magma", shading="nearest", vmin=0.0, vmax=1.0)
    ax.set_yticks(rows)
    ax.set_yticklabels(["inf (no noise)" if not np.isfinite(v) else f"{v:g}" for v in tphi_list_us])
    ax.invert_yaxis()
    ax.set_xlabel("Coupling strength $K$")
    ax.set_ylabel("Phase damping time $T_{\\phi}$ [us]")
    rlabel = "$r_{norm}$ (phase only)" if normalize else "$r$ (amplitude included)"
    ax.set_title(f"Quantum Kuramoto sync map: {rlabel}, backend={backend}")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Order parameter $r$", rotation=270, labelpad=16)

    if np.isfinite(R).any() and np.nanmin(R) < r_contour < np.nanmax(R):
        cs = ax.contour(K, rows, R, levels=[r_contour], colors="white", linewidths=2.0, linestyles="dashed")
        ax.clabel(cs, inline=True, fontsize=10, fmt=f"r = {r_contour}")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(tphi_list_us)))
    for i, v in enumerate(tphi_list_us):
        label = "inf" if not np.isfinite(v) else f"{v:g} us"
        ax2.plot(K, R[i], color=colors[i], lw=1.8, label=label)
    ax2.axhline(r_contour, color="0.4", lw=1.0, ls="dashed")
    ax2.set_xlabel("Coupling strength $K$")
    ax2.set_ylabel("Order parameter $r$")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(title="$T_{\\phi}$", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure to: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generates the r(K, T_phi) phase map for Quantum Kuramoto")
    p.add_argument("--backend", choices=["density_matrix", "shots", "classical"], default="density_matrix")
    p.add_argument("--n", type=int, default=4, help="Number of qubits")
    p.add_argument("--steps", type=int, default=15, help="Number of Trotter steps")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--topology", choices=["chain", "ring"], default="chain")
    p.add_argument("--k-min", type=float, default=0.1)
    p.add_argument("--k-max", type=float, default=1.2)
    p.add_argument("--k-num", type=int, default=23)
    p.add_argument("--tphi", type=float, nargs="*", default=[200, 100, 60, 40, 20, 10],
                    help="List of T_phi [μs] (an 'inf' row is always prepended automatically)")
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--normalize", action="store_true", help="Use phase-only order parameter")
    p.add_argument("--contour", type=float, default=0.5)
    p.add_argument("--out", default="phase_map_r_vs_K_Tphi.png")
    p.add_argument("--npz", default="phase_map_results.npz")
    p.add_argument("--replot", default=None, help="Plot only from a given .npz file without recomputing")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.replot:
        d = np.load(args.replot, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        plot_phase_map(
            d["R"], d["C"], d["K_list"], d["tphi_list_us"],
            r_contour=args.contour, normalize=meta.get("normalize", False),
            backend=meta.get("backend", "?"), save_path=args.out,
        )
        return

    cfg = ModelConfig(n_qubits=args.n, t_steps=args.steps, dt=args.dt, topology=args.topology).resolved()
    K_list = np.round(np.linspace(args.k_min, args.k_max, args.k_num), 4)
    tphi_list_us = np.array([np.inf] + [float(v) for v in args.tphi])

    print(f"Quantum Kuramoto sweep: backend={args.backend}, n={cfg.n_qubits}, "
          f"{len(K_list)} K points × {len(tphi_list_us)} T_phi rows")
    result = run_sweep(
        cfg, K_list, tphi_list_us,
        backend=args.backend, shots=args.shots, seed=args.seed, normalize=args.normalize,
    )

    meta = {"backend": args.backend, "normalize": args.normalize,
            "shots": args.shots, "seed": args.seed, "model": asdict(cfg)}
    np.savez(args.npz, R=result["R"], C=result["C"], K_list=K_list,
             tphi_list_us=tphi_list_us, meta=json.dumps(meta, ensure_ascii=False))
    print(f"Numerical data saved to: {args.npz}")

    plot_phase_map(result["R"], result["C"], K_list, tphi_list_us,
                   r_contour=args.contour, normalize=args.normalize,
                   backend=args.backend, save_path=args.out)


if __name__ == "__main__":
    main()