"""
Minimal numpy simulation reproducing Figure 3(b) from the paper:

    t_detect(μ) / t_detect(μ=2)   vs   μ

for Ball (solid-sphere) targets on a 3-D torus.
Uses a smaller torus (side=128, lmax=64) for faster execution.

The Lévy-step distribution and detection logic are consistent with the
C codebase (func.c / simulation.c).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Simulation parameters (smaller torus) ────────────────────────────
SIDE = 128                                        # torus edge length (paper: 512)
LMAX = SIDE // 2                                  # max Lévy step
MU_VALUES = np.round(np.arange(1.0, 3.2, 0.2), 1)
SURFACE_VALUES = [30, 60, 120, 240, 480, 960]               # Δ = π D²
N_RUNS = 200                                     # Monte-Carlo trials per (μ, Δ)
SEED = 42


# ── Lévy machinery (matches func.c) ─────────────────────────────────

def normalization_constant(mu: float, lmax: int) -> float:
    """Probability of drawing a short (detection-phase) step — see
    ``get_normalization_constant`` in func.c."""
    if mu == 1.0:
        return 1.0 / (1.0 + np.log(lmax))
    return 1.0 / (1.0 - (lmax ** (1.0 - mu) - 1.0) / (mu - 1.0))


def levy_steps(mu: float, lmax: int, a: float, n: int, rng) -> np.ndarray:
    """Draw *n* step lengths from the intermittent Lévy distribution
    (``Levy()`` in func.c)."""
    toss = rng.random(n)
    u = rng.random(n)
    short = u                           # Uniform[0, 1]  — detection phase
    if mu == 1.0:
        long = np.exp((1.0 - a) / a * u)
    else:
        long = ((1.0 - a) / a * (1.0 - mu) * u + 1.0) ** (1.0 / (1.0 - mu))
    return np.where(toss <= a, short, long)


def random_directions(n: int, rng) -> np.ndarray:
    """Random 3-D direction vectors (same convention as the C code:
    θ ~ U[0, 2π], φ ~ U[0, π])."""
    theta = rng.random(n) * (2.0 * np.pi)
    phi = rng.random(n) * np.pi
    sp = np.sin(phi)
    return np.column_stack([sp * np.cos(theta),
                            sp * np.sin(theta),
                            np.cos(phi)])


def toroidal_distance(a: np.ndarray, b: np.ndarray, side: float) -> np.ndarray:
    """Toroidal Euclidean distance (vectorised over rows)."""
    d = np.abs(a - b)
    d = np.minimum(d, side - d)
    return np.sqrt((d ** 2).sum(axis=-1))


# ── Batched simulation ───────────────────────────────────────────────

def simulate_batch(mu, lmax, D, side, n_runs, rng):
    """
    Run *n_runs* independent Lévy-search trials in parallel.

    Target shape : Ball_no_boundary (solid sphere of diameter *D*).
    Detection    : toroidal_distance(walker, target) ≤ D / 2.
    Detection time = total path length (sum of step lengths).
    """
    a = normalization_constant(mu, lmax)
    half_D = 0.5 * D

    walkers = rng.random((n_runs, 3)) * side
    targets = rng.random((n_runs, 3)) * side
    times = np.zeros(n_runs)
    active = np.ones(n_runs, dtype=bool)

    while True:
        n_act = int(active.sum())
        if n_act == 0:
            break
        idx = np.where(active)[0]

        l = levy_steps(mu, lmax, a, n_act, rng)
        dirs = random_directions(n_act, rng)

        walkers[idx] = (walkers[idx] + dirs * l[:, None]) % side
        times[idx] += l

        dist = toroidal_distance(walkers[idx], targets[idx], side)
        found = dist <= half_D
        active[idx[found]] = False

    return times


# ── Worker (top-level for pickling) ──────────────────────────────────

def _run_one(args):
    """Worker executed in a child process."""
    mu, surface, seed = args
    D = np.sqrt(surface / np.pi)
    if D < 1 or D > SIDE:
        return mu, surface, None, None
    rng = np.random.default_rng(seed)
    times = simulate_batch(mu, LMAX, D, SIDE, N_RUNS, rng)
    return mu, surface, float(np.mean(times)), float(np.std(times, ddof=1) / np.sqrt(N_RUNS))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    n_workers = os.cpu_count() or 4
    print(f"Using {n_workers} worker processes")

    # Give each (mu, surface) combo its own reproducible seed
    base_rng = np.random.default_rng(SEED)
    combos = [(mu, s) for mu in MU_VALUES for s in SURFACE_VALUES]
    seeds = base_rng.integers(0, 2**63, size=len(combos))
    tasks = [(mu, s, int(seed)) for (mu, s), seed in zip(combos, seeds)]
    total = len(tasks)

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            mu, surface, mean_time, sem = fut.result()
            done += 1
            if mean_time is not None:
                results[(mu, surface)] = (mean_time, sem)
            print(f"\r  [{done}/{total}]  μ = {mu:.1f}  Δ = {surface:<6}", end="", flush=True)

    print()

    # ── Plot: ratio vs μ ─────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    for surface in SURFACE_VALUES:
        ref = results.get((2.0, surface))
        if ref is None or ref[0] == 0:
            continue
        t_ref, sem_ref = ref
        mus, ratios, errs = [], [], []
        for mu in MU_VALUES:
            key = (mu, surface)
            if key in results:
                t_mu, sem_mu = results[key]
                ratio = t_mu / t_ref
                # Error propagation for ratio: δ(a/b) = (a/b)*sqrt((δa/a)² + (δb/b)²)
                err = ratio * np.sqrt((sem_mu / t_mu) ** 2 + (sem_ref / t_ref) ** 2)
                mus.append(mu)
                ratios.append(ratio)
                errs.append(err)
        ax.errorbar(mus, ratios, yerr=errs, fmt="o-", capsize=3, label=rf"$\Delta = {surface}$")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$t_{\mathrm{detect}}(X^\mu)\;/\;t_{\mathrm{detect}}(X^2)$")
    ax.set_xticks(MU_VALUES)
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Fig 3(b) — Ball target, 3-D torus (side = {SIDE})")

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "fig3b_claude.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
