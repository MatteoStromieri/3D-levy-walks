#!/usr/bin/env python3
"""Minimal NumPy reproduction of Fig. 3b (Ball) on a smaller 3D torus."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path

import numpy as np


DEFAULT_MU = np.round(np.arange(1.0, 3.01, 0.2), 1)
DEFAULT_DELTA = np.array([30.0, 60.0, 120.0, 240.0, 480.0, 960.0])


def get_normalization_constant(mu: float, lmax: float) -> float:
    if mu == 1.0:
        return 1.0 / (1.0 + math.log(lmax))
    return 1.0 / (1.0 - (lmax ** (1.0 - mu) - 1.0) / (mu - 1.0))


def sample_levy_lengths(mu: float, lmax: float, a: float, size: int, rng: np.random.Generator) -> np.ndarray:
    toss = rng.random(size)
    unif = rng.random(size)
    out = np.empty(size, dtype=float)

    short = toss <= a
    out[short] = unif[short]

    if np.any(~short):
        if mu == 1.0:
            out[~short] = np.exp(((1.0 - a) / a) * unif[~short])
        else:
            base = ((1.0 - a) / a) * (1.0 - mu) * unif[~short] + 1.0
            out[~short] = np.power(base, 1.0 / (1.0 - mu))

    return np.minimum(out, lmax)


def ball_effective_radius_from_delta(delta: float) -> float:
    # Same mapping used in C for TargetShape="Ball":
    # D = sqrt(delta / pi) - 2, then detect if dist <= D/2 + 1.
    d = math.sqrt(delta / math.pi) - 2.0
    return 0.5 * d + 1.0


def simulate_mean_detection_time_ball(
    mu: float,
    delta: float,
    side: int,
    n_runs: int,
    rng: np.random.Generator
) -> tuple[float, float]:
    lmax = side / 2.0
    a = get_normalization_constant(mu, lmax)
    detect_radius = ball_effective_radius_from_delta(delta)
    detect_radius_sq = detect_radius * detect_radius

    # Match fig3b.py behavior: random target placement per independent trial.
    positions = rng.random((n_runs, 3)) * side
    targets = rng.random((n_runs, 3)) * side
    times = np.zeros(n_runs, dtype=float)
    alive = np.ones(n_runs, dtype=bool)

    steps = 0
    while np.any(alive):
        steps += 1

        idx = np.flatnonzero(alive)
        n_alive = idx.size

        lengths = sample_levy_lengths(mu, lmax, a, n_alive, rng)

        # Keep the same angular sampling used in C (phi uniform in [0, pi]).
        theta = rng.random(n_alive) * (2.0 * math.pi)
        phi = rng.random(n_alive) * math.pi
        directions = np.column_stack(
            (
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            )
        )

        positions[idx] = (positions[idx] + directions * lengths[:, None]) % side
        times[idx] += lengths

        delta_pos = np.abs(positions[idx] - targets[idx])
        delta_pos = np.minimum(delta_pos, side - delta_pos)
        found = np.einsum("ij,ij->i", delta_pos, delta_pos) <= detect_radius_sq
        alive[idx[found]] = False

    mean_time = float(times.mean())
    std_time = float(times.std(ddof=1)) if n_runs > 1 else 0.0
    return mean_time, std_time


def run_single_condition(
    i_delta: int,
    i_mu: int,
    mu: float,
    delta: float,
    side: int,
    n_runs: int,
    seed: int,
) -> tuple[int, int, float, float]:
    rng = np.random.default_rng(seed)
    mean_time, std_time = simulate_mean_detection_time_ball(
        mu=mu,
        delta=delta,
        side=side,
        n_runs=n_runs,
        rng=rng
    )
    return i_delta, i_mu, mean_time, std_time


def save_ratios_csv(mu_values: np.ndarray, deltas: np.ndarray, ratios: np.ndarray, csv_path: Path) -> None:
    header = "delta," + ",".join(f"mu_{mu:.1f}" for mu in mu_values)
    table = np.column_stack((deltas, ratios))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="", fmt="%.8f")


def plot_ratios(
    mu_values: np.ndarray,
    deltas: np.ndarray,
    ratios: np.ndarray,
    ratio_err: np.ndarray,
    save_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, delta in enumerate(deltas):
        ax.errorbar(
            mu_values,
            ratios[i],
            yerr=ratio_err[i],
            marker="o",
            linewidth=1.2,
            markersize=3.5,
            capsize=2.0,
            elinewidth=0.8,
            label=f"{delta:.1f}",
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$t_{\mathrm{detect}}(X^\mu)/t_{\mathrm{detect}}(X^2)$")
    ax.grid(alpha=0.35)
    ax.legend(title=r"$\Delta$", ncol=3, fontsize=8, title_fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=128, help="Torus side length (paper used 512).")
    parser.add_argument("--n-runs", type=int, default=200, help="Independent runs per (mu, delta).")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker processes for parallel simulation (default: number of CPUs).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("validation/fig3b_codex.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("validation/fig3b_codex_ratios.csv"),
        help="CSV dump of the simulated normalized curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mu_values = DEFAULT_MU
    deltas = DEFAULT_DELTA
    ratios = np.zeros((len(deltas), len(mu_values)), dtype=float)
    ratio_err = np.zeros((len(deltas), len(mu_values)), dtype=float)
    mean_times = np.zeros((len(deltas), len(mu_values)), dtype=float)
    std_times = np.zeros((len(deltas), len(mu_values)), dtype=float)

    print(
        f"Simulating Fig. 3b-style curves (side={args.side}, n_runs={args.n_runs}, threads={args.workers}, "
        f"mu={len(mu_values)}, delta={len(deltas)})",
        flush=True,
    )

    if args.workers <= 1:
        for i, delta in enumerate(deltas):
            for j, mu in enumerate(mu_values):
                cond_seed = args.seed + 1000 * i + j
                mean_time, std_time = run_single_condition(
                    i_delta=i,
                    i_mu=j,
                    mu=float(mu),
                    delta=float(delta),
                    side=args.side,
                    n_runs=args.n_runs,
                    seed=cond_seed
                )[2:]
                mean_times[i, j] = mean_time
                std_times[i, j] = std_time
            print(f"  delta={delta:7.1f} done", flush=True)
    else:
        n_mu = len(mu_values)
        delta_completed = np.zeros(len(deltas), dtype=int)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i, delta in enumerate(deltas):
                for j, mu in enumerate(mu_values):
                    cond_seed = args.seed + 1000 * i + j
                    futures.append(
                        executor.submit(
                            run_single_condition,
                            i,
                            j,
                            float(mu),
                            float(delta),
                            args.side,
                            args.n_runs,
                            cond_seed
                        )
                    )

            for future in as_completed(futures):
                i, j, mean_time, std_time = future.result()
                mean_times[i, j] = mean_time
                std_times[i, j] = std_time
                delta_completed[i] += 1
                if delta_completed[i] == n_mu:
                    print(f"  delta={deltas[i]:7.1f} done", flush=True)

    mu2_idx = int(np.argmin(np.abs(mu_values - 2.0)))
    ratios = mean_times / mean_times[:, [mu2_idx]]
    sem_times = std_times / math.sqrt(args.n_runs)
    sem_ref = sem_times[:, [mu2_idx]]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_sem = np.sqrt((sem_times / mean_times) ** 2 + (sem_ref / mean_times[:, [mu2_idx]]) ** 2)
    ratio_err = ratios * rel_sem
    ratio_err = np.nan_to_num(ratio_err, nan=0.0, posinf=0.0, neginf=0.0)

    save_ratios_csv(mu_values, deltas, ratios, args.csv)
    print(f"Saved ratios: {args.csv}", flush=True)

    wrote_plot = plot_ratios(mu_values, deltas, ratios, ratio_err, args.save)
    if wrote_plot:
        print(f"Saved figure: {args.save}", flush=True)
    else:
        print("matplotlib not available; skipped PNG generation.", flush=True)


if __name__ == "__main__":
    main()
