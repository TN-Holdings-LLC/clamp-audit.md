# PSF-Zero: a cross-domain audit of one control idea

**What this repository is:** a single geometric idea — a smooth saturating
"/0 projective clamp" instead of a hard clip — implemented and then
*adversarially measured* across eight unrelated domains: neural network
robustness, signal-processing anomaly detection, optical metrology, closed-loop
control, power-grid swarm control, C++ nonlinear least squares, spacecraft
attitude allocation, and Riemannian optimisation.

**What it is not:** a demonstration that the idea wins. It mostly doesn't. The
measurements below are the point of the repository, and several of them
falsify the hypothesis it started from.

---

## Why this is worth your time

Every subdirectory follows the same discipline:

1. Take a plausible implementation of the idea.
2. Run it. Not read it — **run it**, against the real library versions, and
   record the numbers.
3. Publish what the numbers say, including when they say the idea contributed
   nothing.
4. Keep the falsified version in the repo, dated, next to the correction.

The result is a repository where the headline claims have already been
attacked by the author. If you are evaluating engineering judgement rather
than a product, that is the signal.

Here is what the audit actually found.

| Domain | The claim under test | Measured outcome |
|---|---|---|
| NN robustness (`gpcl/`) | the /0 clamp suppresses OOD spikes | **0% of damage removed.** A per-feature robust clamp — not part of the /0 idea — removed 98–99% |
| Actuator saturation (`saturation/`) | soft beats hard clipping | **7.6% better at matched optima**, and *worse* at tight limits. Earlier "75%" and "6.2%" figures were artifacts |
| Grid swarm control (`swarm/`) | high-gain stability via /0 | simulation was **unconverged** (nadir moved 150% across timesteps); after fixing, a crossover, not a win |
| Riemannian Adam (`geoclamp/`) | parallel transport should help | **made it 2400× worse** on S³ (0.0016° → 5.10°). Shipped off by default |
| Phase synchrony (`eit/`) | PLV + CUSUM detector | fix was correct; but the tuning was welded to one window length — 6/6 false alarms at `win=128` |
| Ceres / GTSAM (`ceres/`) | robust loss + residual clamp | maths verified exact by sympy; found the wrapper's **gradient is identically zero** (2.8e-17) outside the safe zone |
| Optical metrology (`psd/`) | RMS phase noise from PSD | `fmin` was never honoured — **98.4% of 1/f² power dropped** at fs=1 MHz, and the answer moved 45% with an internal parameter |
| Spacecraft allocation (`aerospace/`) | thruster QP + attitude loop | the original test's own fixture was **unreachable**, so it would pass a `return np.zeros(8)` stub |

Four of eight domains produced a **negative result for the core idea**. Two
found the surrounding measurement apparatus was broken before the idea could
even be evaluated. That is a more useful thing to have built than eight
success stories would have been.

---

## The one place the idea does earn its keep

Soft saturation is not useless — it is *narrowly* useful, and the repo pins
down where. In a closed loop with a delayed plant (`saturation/`), matched
gain and matched limit:

```
limit    hard clip   soft sat   winner
  0.4      0.2988     0.3137    hard      <- tight limit: soft wastes authority
  0.6      0.1853     0.2355    hard
  1.0      0.2566     0.1730    soft
  3.0      0.8848     0.4131    soft      <- loose limit: hard clip approaches
                                             the unlimited case, which is unstable
```

Tuned per arm, the gap is **7.6%**. But the robustness to *mis-setting* the
limit is the real result: worst/best error across the sweep is **4.8× for the
hard clip and 2.0× for the soft one**. The value is insensitivity to a
badly-chosen threshold, not raw performance.

That is the honest version of the claim, and it is the only version this
repository supports.

---

## Repository layout

* **[gpcl/](./gpcl/)** — PyTorch pre-activation robustness layer + per-stage ablation
* **[eit/](./eit/)** — Multi-channel phase-synchrony change detection (PLV + CUSUM)
* **[psd/](./psd/)** — RMS phase noise from a time series or a tabulated PSD
* **[saturation/](./saturation/)** — Closed-loop soft-vs-hard actuator saturation benchmark
* **[swarm/](./swarm/)** — Low-inertia grid droop control, timestep-convergence checked
* **[ceres/](./ceres/)** — C++ robust loss + residual wrapper for Ceres 2.2 / GTSAM 4.2
* **[aerospace/](./aerospace/)** — Spacecraft attitude control + thruster allocation (cvxpy)
* **[geoclamp/](./geoclamp/)** — Riemannian Adam on SO(3) and S³
* **[ros2/](./ros2/)** — IMU chaos injector (ROS2) + phase-recovery plotting
* **[quantum/](./quantum/)** — Quantum Kuramoto phase maps (Qiskit)
* **[engine/](./engine/)** — Cross-domain A/B harness driving four domains through one code path
* **[hardware/](./hardware/)** — R0-Core Phase-MAC chip emulator

Every directory that audits a fix contains the same four things:

- the **implementation**,
- the **original version** it replaced — kept, not deleted, so the A/B is
  reproducible rather than asserted,
- a **test that can fail**: assertions and a non-zero exit code, not printed
  booleans,
- a **diagnostic script** (`diag_*.py`) that demonstrates the original defect
  standing alone, before any fix is applied.

`quantum/`, `engine/` and `hardware/` are instruments rather than audits and
carry only their implementation.

## Running the tests

Every test is standalone and prints a pass/fail count.

```bash
python gpcl/test_gpcl_v2.py                          # 21 checks
python eit/test_eit_v2.py                            # 10 checks
python psd/test_psd_to_rms_phase_v2.py               # 14 checks
python swarm/test_swarm_control_v2.py                # 17 checks
python geoclamp/test_geoclamp_adam_v2.py             # 20 checks
python aerospace/test_psf_zero_aerospace_core_v2.py  # 32 checks
python ros2/test_phase_transition_v2.py              # 15 checks

# the benchmarks print their own tables rather than a pass/fail count
python saturation/soft_saturation_bench.py
python gpcl/ablation.py

g++ -std=c++17 -O2 ceres/test_psf_zero_v3.cpp -I/usr/include/eigen3 \
    -lceres -lglog -lgtsam -ltbb -o t && ./t         # 26 checks
g++ -std=c++17 -O2 ros2/test_drain_order.cpp -o t && ./t
```

Each Python test exits non-zero on failure, so `&&`-chaining them works as a
smoke suite. The two C++ tests need their libraries present; everything else
runs on numpy/scipy/torch alone.

Verified against: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, torch 2.13.0,
cvxpy 1.9.2, Ceres 2.2.0, Eigen 3.4.0, GTSAM 4.2.0, g++ 13.3.

## Method notes

Three rules the repository holds itself to, because they are what make the
negative results trustworthy:

**Never test a reimplementation.** Several of the original tests here built
their own copy of the logic inline and tested that instead of the module —
so the module could differ and the test would still pass. Every test now
drives the shipped code path.

**A test that cannot fail is a demo.** One original test file contained zero
assertions: it computed booleans, printed them, and exited 0 regardless. Its
"no unwinding" criterion passed a trajectory that drifted 2,699 degrees.

**Check the fixture, not just the code.** The aerospace test's requested
torque was outside the reachable cone of its own thruster geometry, so the
correct answer was all-zeros — which its assertions accepted. A fixture that
any stub passes is not a test.

## Status and honesty

This is a research audit, not a library. Nothing here has been run on
hardware. The domains are synthetic models chosen to stress one specific
question, and a different plant, network, or dataset can move any of these
crossovers. Where a claim is theoretical rather than measured, the docstring
says so; where a measurement contradicted the author's own expectation, the
expectation was retracted in place rather than quietly deleted.

If you find a number here that does not reproduce, that is a bug worth an
issue.
