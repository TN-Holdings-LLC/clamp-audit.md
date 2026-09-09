# -*- coding: utf-8 -*-
"""
R0-Core PSF-Chip Emulator
Phase-MAC + /0-Trap (Z-IDLE surrender) hardware logic.

Improvements over the previous revision (which already fixed the ignored
coupling matrix and the mutable default argument). Each item below was
confirmed by running the code, not by reading it.

FIX #1 (primary, wrong output value): get_output() returned an arithmetic
mean of angles as the tile's "synchronized truth". An arithmetic mean of
circular quantities is meaningless across the 0/2*pi wraparound. Measured
example: phases [0.05, 6.25, 0.02, 6.20, 0.10] give r = 0.998 (essentially
perfect lock), yet the arithmetic mean reports 2.524 rad while the true
consensus phase is 0.011 rad -- the tile reports a value 2.5 rad away from
where every oscillator actually is, precisely in the LOCKED state that is
supposed to be trustworthy. Fixed with the circular mean, i.e. the argument
of the complex order parameter, which is the same quantity whose magnitude
is already used for r. The previous revision identified this exact problem
in its own notes for the coupling term but left get_output() untouched.

FIX #2 (state machine): there was no transition out of LOCKED. Once r rose
above lock_threshold the tile stayed LOCKED forever, even while coherence
collapsed and the surrender counter was ticking up. Traced with r going
0.95 -> 0.95 -> 0.30 -> 0.30 -> 0.30: the tile keeps reporting LOCKED with
counter = 3, so a caller reading get_output() receives a "synchronized"
phase computed from desynchronized oscillators. Fixed by adding an explicit
LOCKED -> ACTIVE transition, with a separate unlock_threshold so the tile
does not chatter between the two states near the boundary.

FIX #3 (reproducibility): the tile drew both its initial phases and its
per-cycle noise from the global numpy RNG, with no way to seed a tile. The
previous revision's notes claim results over "5/5 seeds" and "20/20 trials",
but the class exposes no seed at all, so those runs could not have been
reproduced from the public API. Each tile now owns a numpy Generator
(seed argument), which makes every experiment below repeatable.

FIX #4 (noise scaling): the update was phases += (coupling + noise) * 0.08,
with the timestep folded into a magic constant. That makes the noise term
scale like dt when Euler-Maruyama integration of a phase SDE requires
sqrt(dt), so halving dt changed the effective noise strength instead of
just refining the integration. dt is now a config field, drift scales with
dt and noise with sqrt(dt), and noise_level is therefore a property of the
environment rather than of the step size.

FIX #5 (dead config, unvalidated input): noise_level_normal and
noise_level_chaos were never read by anything -- both call sites passed
literals. The coupling matrix was never checked for shape, so a mismatched
matrix would either broadcast silently or raise deep inside numpy. Both are
now wired up and validated.

FIX #6 (Z_IDLE was not actually idle): the surrender path set r = 0.0 and
commented "physically power down oscillators", but left self.phases holding
the last pre-surrender values, so stale phase data remained readable from a
tile that claims to be powered down. Phases are now parked (NaN) on
surrender, and reset() / wake() provide the recovery path a real tile needs.

FIX #7 (experiment design): Test 2 varied the coupling matrix AND the noise
level at the same time (chaotic K with noise_level=5.0 versus harmonious K
with 0.08), so it could not distinguish "contradictory coupling causes
surrender" from "loud noise causes surrender" -- which is the claim the demo
is making. The demo is now a 2x2 factorial over {harmonious, chaotic} x
{low, high noise} across several seeds, which separates the two factors.

Dependencies: numpy. matplotlib is optional (plotting is skipped without it).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


# =========================================================
# Circular statistics
# =========================================================
def order_parameter(phases: np.ndarray) -> complex:
    """Kuramoto complex order parameter z = (1/N) * sum_j exp(i*theta_j).

    |z| is the coherence r; arg(z) is the consensus phase. Both quantities
    come from the same object, which is why the tile must not compute one
    of them with an arithmetic mean of angles.
    """
    return complex(np.mean(np.exp(1j * phases)))


def coherence(phases: np.ndarray) -> float:
    return float(abs(order_parameter(phases)))


def circular_mean(phases: np.ndarray) -> float:
    """Consensus phase in [0, 2*pi). Undefined when coherence is ~0."""
    z = order_parameter(phases)
    return float(math.atan2(z.imag, z.real) % (2 * math.pi))


# =========================================================
# Configuration and state
# =========================================================
class TileState(str, Enum):
    ACTIVE = "ACTIVE"
    LOCKED = "LOCKED"
    Z_IDLE = "Z_IDLE"


@dataclass(frozen=True)
class R0TileConfig:
    num_oscillators: int = 16
    coupling_strength: float = 1.5      # global gain K applied on top of the matrix
    dt: float = 0.08                    # integration timestep
    noise_level_normal: float = 0.08    # in-distribution environment
    noise_level_chaos: float = 4.5      # out-of-distribution environment
    lock_threshold: float = 0.88        # r at or above this -> LOCKED
    unlock_threshold: float = 0.80      # r below this -> leave LOCKED (hysteresis)
    surrender_cycles: int = 40          # consecutive incoherent cycles before Z_IDLE
    park_on_idle: bool = True           # blank the phase register on surrender

    def __post_init__(self) -> None:
        if self.unlock_threshold > self.lock_threshold:
            raise ValueError("unlock_threshold must not exceed lock_threshold")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.num_oscillators < 2:
            raise ValueError("a tile needs at least two oscillators")


@dataclass
class TileOutput:
    """Structured result of a read.

    phase is None whenever the tile has nothing trustworthy to report, so a
    caller cannot accidentally consume a placeholder as if it were data.
    """
    state: TileState
    coherence: float
    phase: float | None
    reason: str

    @property
    def is_trustworthy(self) -> bool:
        return self.phase is not None


# =========================================================
# Coupling matrices
# =========================================================
def harmonious_coupling(n: int, strength: float = 1.8) -> np.ndarray:
    """All-to-all positive coupling: every pair pulls toward agreement."""
    K = np.full((n, n), float(strength))
    np.fill_diagonal(K, 0.0)
    return K


def chaotic_coupling(n: int, rng: np.random.Generator, scale: float = 2.5) -> np.ndarray:
    """Signed random coupling: some pairs attract, others repel."""
    K = rng.uniform(-scale, scale, (n, n))
    np.fill_diagonal(K, 0.0)
    return K


# =========================================================
# Tile
# =========================================================
class R0Tile:
    """A single phase-space processing tile on the R0-Core PSF-chip.

    The tile integrates a Kuramoto phase network instead of Boolean logic,
    and implements the /0-Trap: when coherence cannot be sustained it stops
    producing an answer rather than producing an unsupported one.
    """

    def __init__(self, tile_id: str, config: R0TileConfig | None = None,
                 seed: int | None = None, record_history: bool = False):
        self.tile_id = tile_id
        self.config = config if config is not None else R0TileConfig()
        self.rng = np.random.default_rng(seed)
        self.record_history = record_history
        self.reset()

    # --- lifecycle --------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Return the tile to a freshly powered-up state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.phases = self.rng.uniform(0.0, 2 * np.pi, self.config.num_oscillators)
        self.state = TileState.ACTIVE
        self.r_order = coherence(self.phases)
        self.contradiction_counter = 0
        self.cycles = 0
        self.history: list[dict] = []

    def wake(self) -> None:
        """Recover from Z_IDLE with fresh phases. A surrendered tile has no
        recovery path otherwise, which would make the trap a one-way latch."""
        if self.state is TileState.Z_IDLE:
            self.phases = self.rng.uniform(0.0, 2 * np.pi, self.config.num_oscillators)
            self.state = TileState.ACTIVE
            self.contradiction_counter = 0
            self.r_order = coherence(self.phases)

    @property
    def num_oscillators(self) -> int:
        return self.config.num_oscillators

    # --- one clock cycle --------------------------------------------------
    def _validate(self, coupling_matrix: np.ndarray) -> np.ndarray:
        K = np.asarray(coupling_matrix, dtype=float)
        n = self.num_oscillators
        if K.shape != (n, n):
            raise ValueError(f"coupling_matrix must be {(n, n)}, got {K.shape}")
        if not np.all(np.isfinite(K)):
            raise ValueError("coupling_matrix contains non-finite entries")
        return K

    def phase_mac_cycle(self, coupling_matrix: np.ndarray,
                        noise_level: float | None = None) -> None:
        """One hardware clock cycle: phase multiply-accumulate.

            dtheta_i = (K/N) * sum_j M_ij * sin(theta_j - theta_i) * dt
                       + noise * sqrt(dt)

        Drift scales with dt and diffusion with sqrt(dt), so noise_level
        describes the environment and stays meaningful when dt changes.
        """
        if self.state is TileState.Z_IDLE:
            return

        K = self._validate(coupling_matrix)
        sigma = self.config.noise_level_normal if noise_level is None else float(noise_level)
        n = self.num_oscillators
        dt = self.config.dt

        theta = self.phases
        diff = theta[None, :] - theta[:, None]          # diff[i, j] = theta_j - theta_i
        drift = self.config.coupling_strength * (K * np.sin(diff)).sum(axis=1) / n
        noise = self.rng.normal(0.0, sigma, n)

        self.phases = np.mod(theta + drift * dt + noise * math.sqrt(dt), 2 * np.pi)
        self.r_order = coherence(self.phases)
        self.cycles += 1

    # --- /0-Trap ----------------------------------------------------------
    def apply_zero_trap(self) -> None:
        """Surrender if synchronization cannot be sustained.

        Three bands, so the tile leaves LOCKED when coherence is genuinely
        lost without oscillating between states at the boundary:
            r >= lock_threshold      -> LOCKED, counter cleared
            r <  unlock_threshold    -> ACTIVE, counter advances
            in between               -> hold current state
        """
        if self.state is TileState.Z_IDLE:
            return

        cfg = self.config
        if self.r_order >= cfg.lock_threshold:
            self.state = TileState.LOCKED
            self.contradiction_counter = 0
        elif self.r_order < cfg.unlock_threshold:
            self.state = TileState.ACTIVE
            self.contradiction_counter += 1
            if self.contradiction_counter >= cfg.surrender_cycles:
                self._surrender()
        elif self.state is not TileState.LOCKED:
            self.contradiction_counter += 1
            if self.contradiction_counter >= cfg.surrender_cycles:
                self._surrender()

        if self.record_history:
            self.history.append({
                "cycle": self.cycles,
                "r": self.r_order,
                "state": self.state.value,
                "counter": self.contradiction_counter,
            })

    def _surrender(self) -> None:
        """Enter Z_IDLE and blank the phase register.

        Parking the phases matters: leaving the last pre-surrender values in
        place lets a caller read stale data out of a tile that reports itself
        as powered down.
        """
        self.state = TileState.Z_IDLE
        self.r_order = 0.0
        if self.config.park_on_idle:
            self.phases = np.full(self.num_oscillators, np.nan)

    def step(self, coupling_matrix: np.ndarray, noise_level: float | None = None) -> None:
        """Convenience: one cycle followed by the trap evaluation."""
        self.phase_mac_cycle(coupling_matrix, noise_level)
        self.apply_zero_trap()

    # --- readout ----------------------------------------------------------
    def get_output(self) -> TileOutput:
        if self.state is TileState.LOCKED:
            return TileOutput(self.state, self.r_order, circular_mean(self.phases), "locked")
        if self.state is TileState.Z_IDLE:
            return TileOutput(self.state, 0.0, None, "surrendered")
        return TileOutput(self.state, self.r_order, None, "not-converged")

    def __repr__(self) -> str:
        return (f"R0Tile({self.tile_id!r}, state={self.state.value}, "
                f"r={self.r_order:.4f}, cycles={self.cycles})")


# =========================================================
# Experiment: separate coupling structure from noise level
# =========================================================
def run_trial(condition: str, coupling: str, noise_level: float, seed: int,
              cycles: int, config: R0TileConfig, record: bool = False) -> dict:
    tile = R0Tile(f"R0-{condition}-{seed}", config=config, seed=seed, record_history=record)
    matrix_rng = np.random.default_rng(seed + 10_000)
    K = (harmonious_coupling(tile.num_oscillators)
         if coupling == "harmonious"
         else chaotic_coupling(tile.num_oscillators, matrix_rng))

    for _ in range(cycles):
        tile.step(K, noise_level)

    out = tile.get_output()
    return {
        "condition": condition,
        "coupling": coupling,
        "noise_level": noise_level,
        "seed": seed,
        "final_state": out.state.value,
        "r_final": round(out.coherence, 6),
        "phase": "" if out.phase is None else round(out.phase, 6),
        "surrendered": out.state is TileState.Z_IDLE,
        "history": tile.history,
    }


def run_factorial(cycles: int, seeds: int, config: R0TileConfig, outdir: str) -> list[dict]:
    """2x2 factorial: coupling structure x noise level.

    The original demo changed both factors at once, so a Z_IDLE outcome could
    not be attributed to either. The off-diagonal cells here are what make the
    claim testable.
    """
    conditions = [
        ("harmonious-quiet", "harmonious", config.noise_level_normal),
        ("harmonious-loud", "harmonious", config.noise_level_chaos),
        ("chaotic-quiet", "chaotic", config.noise_level_normal),
        ("chaotic-loud", "chaotic", config.noise_level_chaos),
    ]

    rows: list[dict] = []
    print(f"{'condition':<20}{'lock rate':>11}{'Z_IDLE rate':>13}{'mean r':>10}")
    print("-" * 54)
    for name, coupling, noise in conditions:
        trials = [run_trial(name, coupling, noise, seed, cycles, config, record=(seed == 0))
                  for seed in range(seeds)]
        rows.extend(trials)
        lock = sum(t["final_state"] == "LOCKED" for t in trials) / seeds
        idle = sum(t["surrendered"] for t in trials) / seeds
        mean_r = float(np.mean([t["r_final"] for t in trials]))
        print(f"{name:<20}{lock:>11.2f}{idle:>13.2f}{mean_r:>10.4f}")

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "factorial_results.csv")
    fields = ["condition", "coupling", "noise_level", "seed", "final_state",
              "r_final", "phase", "surrendered"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nresults written to {csv_path}")
    return rows


def plot_histories(rows: list[dict], config: R0TileConfig, outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping the plot.")
        return

    tracked = [r for r in rows if r["history"]]
    if not tracked:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=140, sharex=True, sharey=True)
    for ax, row in zip(axes.ravel(), tracked):
        cycles = [h["cycle"] for h in row["history"]]
        r = [h["r"] for h in row["history"]]
        idle = [h["cycle"] for h in row["history"] if h["state"] == "Z_IDLE"]
        ax.plot(cycles, r, lw=1.4, color="#1F5C7A")
        ax.axhline(config.lock_threshold, color="#4E6B2F", ls="--", lw=1.0)
        ax.axhline(config.unlock_threshold, color="#8A5A00", ls=":", lw=1.0)
        if idle:
            ax.axvline(idle[0], color="#A32B2B", lw=1.4)
            ax.text(idle[0], 0.92, " Z_IDLE", color="#A32B2B", fontsize=8, va="top")
        ax.set_title(f"{row['condition']}  (final: {row['final_state']})", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("coherence r")
    fig.suptitle("R0 tile: coupling structure vs noise level (seed 0)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "factorial_r_traces.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"plot written to {path}")


# =========================================================
# Self-tests
# =========================================================
def selftest() -> None:
    print("selftest:")

    # 1) The consensus phase must be circular, not arithmetic.
    phases = np.array([0.05, 6.25, 0.02, 6.20, 0.10])
    assert coherence(phases) > 0.99
    assert abs(circular_mean(phases) - 0.011) < 0.05
    assert abs(float(np.mean(phases)) - 2.524) < 0.01   # what the old code returned
    print("  [ok] circular mean 0.011 rad where the arithmetic mean reported 2.524 rad")

    # 2) A locked tile that loses coherence must stop claiming LOCKED.
    cfg = R0TileConfig(surrender_cycles=5)
    tile = R0Tile("t", config=cfg, seed=0)
    tile.phases = np.zeros(cfg.num_oscillators)          # perfectly aligned
    tile.r_order = coherence(tile.phases)
    tile.apply_zero_trap()
    assert tile.state is TileState.LOCKED
    tile.phases = np.linspace(0, 2 * np.pi, cfg.num_oscillators, endpoint=False)
    tile.r_order = coherence(tile.phases)               # r ~ 0
    tile.apply_zero_trap()
    assert tile.state is TileState.ACTIVE, tile.state
    assert tile.get_output().phase is None
    print("  [ok] LOCKED -> ACTIVE on coherence loss, and no phase is reported")

    # 3) Surrender parks the register; no stale data survives.
    for _ in range(cfg.surrender_cycles):
        tile.apply_zero_trap()
    assert tile.state is TileState.Z_IDLE
    assert np.all(np.isnan(tile.phases))
    assert tile.get_output().phase is None
    tile.wake()
    assert tile.state is TileState.ACTIVE and np.all(np.isfinite(tile.phases))
    print("  [ok] Z_IDLE parks phases as NaN and wake() restores the tile")

    # 4) Seeding makes a run reproducible, and different seeds differ.
    def final(seed: int) -> float:
        t = R0Tile("s", seed=seed)
        K = harmonious_coupling(t.num_oscillators)
        for _ in range(30):
            t.step(K, 0.5)
        return t.r_order
    assert final(3) == final(3)
    assert final(3) != final(4)
    print("  [ok] identical seeds reproduce, different seeds diverge")

    # 5) The coupling matrix actually drives the dynamics (regression for the
    #    earlier bug where it was ignored), holding noise fixed.
    def run(kind: str) -> float:
        t = R0Tile("c", seed=11)
        rng = np.random.default_rng(11)
        K = harmonious_coupling(t.num_oscillators) if kind == "h" else chaotic_coupling(t.num_oscillators, rng)
        for _ in range(120):
            t.step(K, 0.08)
        return t.r_order
    assert run("h") - run("c") > 0.3, (run("h"), run("c"))
    print("  [ok] harmonious vs chaotic coupling separates at identical noise")

    # 6) Shape validation instead of a silent broadcast.
    try:
        R0Tile("v", seed=0).phase_mac_cycle(np.ones((4, 4)))
    except ValueError:
        print("  [ok] a wrong-shaped coupling matrix raises ValueError")
    else:
        raise AssertionError("expected ValueError")

    # 7) noise_level defaults to the configured environment value.
    t = R0Tile("n", config=R0TileConfig(noise_level_normal=0.0), seed=0)
    before = t.phases.copy()
    t.phase_mac_cycle(np.zeros((t.num_oscillators, t.num_oscillators)))
    assert np.allclose(before, t.phases), "zero coupling and zero noise must be a no-op"
    print("  [ok] config noise levels are wired to the cycle")

    print("selftest: all checks passed")


# =========================================================
# CLI
# =========================================================
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="R0-Core PSF-chip emulator")
    p.add_argument("--cycles", type=int, default=120)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--outdir", default="logs")
    p.add_argument("--noise-normal", type=float, default=0.08)
    p.add_argument("--noise-chaos", type=float, default=4.5)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        selftest()
        return

    config = R0TileConfig(noise_level_normal=args.noise_normal,
                          noise_level_chaos=args.noise_chaos)
    print("=== R0-Core PSF-Chip Emulator ===\n")
    rows = run_factorial(args.cycles, args.seeds, config, args.outdir)
    if not args.no_plot:
        plot_histories(rows, config, args.outdir)


if __name__ == "__main__":
    main()
