#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_transition_proof.py

Plots the phase error injected by the chaos node against the EIT tracker's
recovery, and quantifies how long recovery actually takes.

Improvements over the previous revision, which fixed position-based
interpolation by switching to method='index':

FIX #1 (the plot asserted data that does not exist): time-based
interpolation is correct across small gaps, but it still draws a straight
line across arbitrarily long ones. The chaos injector's whole job is to
create those gaps, so a dropout of several seconds was rendered as a smooth
green line at a plausible-looking value -- measured on a synthetic log with
an 8 s dropout, the interpolated value at the midpoint was 0.219 ms, drawn
as if it had been sampled. A recovery-time figure whose most dramatic
region is invented is worse than one with a hole in it. Gaps longer than
--max-gap are now left as NaN, so matplotlib breaks the line there, and the
gaps are shaded and counted.

FIX #2 (no numbers): the script produced a picture and nothing else, while
the claim being made is quantitative -- that the tracker returns inside the
epsilon bound quickly after each phase step. Recovery time is now measured
per event (time from the rising edge of TNow_event until |phi_eit| stays
within epsilon for --settle-hold seconds), printed as a table, annotated on
the plot, and written to a CSV next to the figure.

FIX #3 (schema assumptions): the required columns were read directly, so a
log with a different schema failed with a bare KeyError from deep inside
pandas. Missing columns are now reported by name up front. Rows with a NaN
timestamp are dropped, and duplicate timestamps -- which the injector's own
duplicate feature produces -- are averaged instead of silently breaking the
time index.

FIX #4 (epsilon was hard-coded twice): the 5 ms bound was written into the
axhline calls only, so the drawn bound could disagree with whatever the
node used to compute `settled`. It is a parameter now and the same value
drives the recovery metric.

FIX #5 (headless use): plt.show() at the end blocks or warns under a
headless CI runner, which is where this script usually runs. The backend is
now chosen automatically and the window only opens with --show.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["t", "phi_inj_ms", "phi_eit_ms"]
OPTIONAL_COLUMNS = ["TNow_event", "settled"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot and score the phase recovery log")
    p.add_argument("--csv", default="/tmp/phase_tester.csv")
    p.add_argument("--out", default="/tmp/phase_transition_proof.png")
    p.add_argument("--metrics", default=None,
                   help="path for the per-event metrics CSV (defaults to --out with a .csv suffix)")
    p.add_argument("--epsilon", type=float, default=5.0, help="epsilon bound in ms")
    p.add_argument("--max-gap", type=float, default=0.5,
                   help="sample gaps longer than this (seconds) are left unfilled")
    p.add_argument("--settle-hold", type=float, default=0.2,
                   help="seconds the signal must stay inside epsilon to count as settled")
    p.add_argument("--show", action="store_true", help="open a window as well as writing the file")
    return p.parse_args(argv)


def load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found. Run the phase tester node first.")

    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"Error: {path} is empty (no rows logged).")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"Error: {path} is missing required column(s): {', '.join(missing)}. "
                 f"Found: {', '.join(df.columns)}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=["t"]).sort_values("t")
    # The injector duplicates messages, so identical timestamps do occur.
    # Averaging keeps the time index usable instead of leaving ambiguity.
    if df["t"].duplicated().any():
        df = df.groupby("t", as_index=False).mean(numeric_only=True)

    df = df.reset_index(drop=True)
    df["t_rel"] = df["t"] - df["t"].iloc[0]
    return df


def interpolate_with_gaps(df: pd.DataFrame, column: str, max_gap: float) -> pd.Series:
    """Interpolate against real elapsed time, but refuse to bridge long gaps.

    Row position is meaningless here because the injector produces irregular
    sampling, so interpolation uses the timestamp as the x-axis. Anything
    wider than max_gap stays NaN so the line breaks instead of inventing a
    trajectory through the dropout.
    """
    s = df.set_index("t_rel")[column].interpolate(method="index", limit_area="inside")

    t = df["t_rel"].to_numpy()
    known = df[column].notna().to_numpy()
    if known.any():
        known_t = t[known]
        # Distance from each sample to the nearest sample that was actually
        # logged, on either side.
        left = np.searchsorted(known_t, t, side="right") - 1
        right = np.clip(left + 1, 0, len(known_t) - 1)
        left = np.clip(left, 0, len(known_t) - 1)
        span = known_t[right] - known_t[left]
        too_wide = (span > max_gap) & ~known
        s = s.reset_index(drop=True)
        s[too_wide] = np.nan
    else:
        s = s.reset_index(drop=True)
    return s


def find_gaps(df: pd.DataFrame, max_gap: float) -> list[tuple[float, float]]:
    t = df["t_rel"].to_numpy()
    d = np.diff(t)
    return [(float(t[i]), float(t[i + 1])) for i in np.where(d > max_gap)[0]]


def recovery_events(df: pd.DataFrame, phi: pd.Series, epsilon: float,
                    settle_hold: float) -> pd.DataFrame:
    """Time from each event's rising edge until the tracker settles.

    Settled means |phi| stays within epsilon for settle_hold seconds, which
    is stricter than a single sample dipping under the bound during a
    swing through zero.
    """
    ev = pd.to_numeric(df["TNow_event"], errors="coerce").fillna(0.0).to_numpy() > 0.5
    t = df["t_rel"].to_numpy()
    inside = (phi.abs() <= epsilon).to_numpy() & phi.notna().to_numpy()

    rising = np.where(ev & ~np.r_[False, ev[:-1]])[0]
    rows = []
    for start in rising:
        settle_t = np.nan
        for k in range(start, len(t)):
            if not inside[k]:
                continue
            end = t[k] + settle_hold
            window = (t >= t[k]) & (t <= end)
            if window.any() and inside[window].all() and t[window][-1] >= end - 1e-9:
                settle_t = t[k]
                break
        peak = np.nanmax(np.abs(phi.to_numpy()[start:])) if start < len(phi) else np.nan
        rows.append({
            "event_start_s": float(t[start]),
            "settled_s": float(settle_t) if np.isfinite(settle_t) else np.nan,
            "recovery_s": float(settle_t - t[start]) if np.isfinite(settle_t) else np.nan,
            "peak_abs_phi_ms": float(peak),
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, phi: pd.Series, events: pd.DataFrame,
         gaps: list[tuple[float, float]], args: argparse.Namespace) -> None:
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.scatter(df["t_rel"], df["phi_inj_ms"], color="red", s=10, alpha=0.5,
                label="Injected chaos (raw phase error)")
    ax1.plot(df["t_rel"], phi, color="lime", linewidth=2.5,
             label="EIT now-phase (recovery)")
    ax1.axhline(args.epsilon, color="gray", linestyle="--", alpha=0.5,
                label=f"Epsilon bound ({args.epsilon:g} ms)")
    ax1.axhline(-args.epsilon, color="gray", linestyle="--", alpha=0.5)

    for i, (g0, g1) in enumerate(gaps):
        ax1.axvspan(g0, g1, color="#555555", alpha=0.45,
                    label="Data gap (not interpolated)" if i == 0 else None)

    for _, e in events.iterrows():
        if np.isfinite(e["recovery_s"]):
            ax1.annotate(f"{e['recovery_s']:.2f}s",
                         xy=(e["settled_s"], 0), xytext=(e["settled_s"], args.epsilon * 2.0),
                         color="white", fontsize=8, ha="center",
                         arrowprops=dict(color="white", arrowstyle="->", alpha=0.6))

    ax1.set_title("Phase chaos recovery", fontsize=16, fontweight="bold", color="white")
    ax1.set_ylabel("Phase discrepancy (ms)", fontsize=12)
    ax1.grid(True, color="#333333")
    ax1.legend(loc="upper right")

    ax2.fill_between(df["t_rel"], 0, pd.to_numeric(df["TNow_event"], errors="coerce").fillna(0),
                     color="orange", alpha=0.3, step="post", label="Phase step detected")
    ax2.plot(df["t_rel"], pd.to_numeric(df["settled"], errors="coerce").fillna(0),
             color="cyan", drawstyle="steps-post", label="Settled")
    ax2.set_xlabel("Elapsed time (s)", fontsize=12)
    ax2.set_ylabel("Boolean state", fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.legend(loc="center right")
    ax2.grid(True, color="#333333", axis="x")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"plot saved to {args.out}")
    if args.show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_log(args.csv)

    phi = interpolate_with_gaps(df, "phi_eit_ms", args.max_gap)
    gaps = find_gaps(df, args.max_gap)
    events = recovery_events(df, phi, args.epsilon, args.settle_hold)

    span = float(df["t_rel"].iloc[-1])
    within = float((phi.abs() <= args.epsilon).mean() * 100.0)
    print(f"samples={len(df)}  span={span:.2f}s  "
          f"gaps>{args.max_gap:g}s={len(gaps)}  inside epsilon={within:.1f}% of samples")

    if events.empty:
        print("no TNow_event rising edges found; recovery time not measured")
    else:
        print("\nevent_start_s  recovery_s  peak_abs_phi_ms")
        for _, e in events.iterrows():
            rec = "not settled" if not np.isfinite(e["recovery_s"]) else f"{e['recovery_s']:.3f}"
            print(f"{e['event_start_s']:>13.2f}  {rec:>10}  {e['peak_abs_phi_ms']:>15.3f}")
        settled = events["recovery_s"].dropna()
        if not settled.empty:
            print(f"\nrecovery: median {settled.median():.3f}s, "
                  f"worst {settled.max():.3f}s, {len(settled)}/{len(events)} events settled")

    metrics_path = args.metrics or os.path.splitext(args.out)[0] + "_metrics.csv"
    events.to_csv(metrics_path, index=False)
    print(f"metrics saved to {metrics_path}")

    plot(df, phi, events, gaps, args)


if __name__ == "__main__":
    main()
