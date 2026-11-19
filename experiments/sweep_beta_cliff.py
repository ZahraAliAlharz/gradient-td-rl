"""Tests whether beta* = 2 / (rho_max * max_s ||phi(s)||^2) predicts where
GTD2 and TDC's beta-divergence cliff actually sits, across setups that move
the prediction: feature scale, the behavior policy's P(solid), and a
full-column-rank feature variant (see predicted_beta_cliff() in
envs/bairds_star.py for the derivation).

All (algorithm, alpha, beta) combinations for a given setup replay the same
simulated behavior trajectory. That's valid, not a shortcut: the behavior
policy's transitions never depend on the learner's weights, so the same
trajectory can legitimately be fed to many learners at once. It also means
the Markov chain only has to be simulated once per setup instead of once per
(algorithm, alpha, beta, seed) -- the thing that makes the full sweep run in
well under the ~20 minute budget.

Generates results/beta_cliff.csv and results/beta_cliff_predicted_vs_observed.png.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_td_rl.envs.bairds_star import (
    CLASSIC_W_INIT, DASHED, GAMMA, N_STATES, SOLID, TARGET_POLICY, TOP_STATE,
    build_feature_matrix, predicted_beta_cliff,
)
from gradient_td_rl.vectorized import run_batch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_SEEDS = 10
BASE_ALPHAS = [0.001, 0.005, 0.01]
N_BETAS = 15
BETA_RANGE = (0.6, 1.6)  # multiples of the predicted cliff
DIVERGENCE_NORM = 1e8
ALGORITHMS = ["GTD2", "TDC"]


def setups():
    default_behavior = {SOLID: 1.0 / 7.0, DASHED: 6.0 / 7.0}
    return [
        dict(name="default", behavior=default_behavior, c=1.0, full_rank=False),
        dict(name="c=0.5", behavior=default_behavior, c=0.5, full_rank=False),
        dict(name="c=2", behavior=default_behavior, c=2.0, full_rank=False),
        dict(name="p=1/4", behavior={SOLID: 0.25, DASHED: 0.75}, c=1.0, full_rank=False),
        dict(name="p=1/10", behavior={SOLID: 0.1, DASHED: 0.9}, c=1.0, full_rank=False),
        dict(name="full_rank", behavior=default_behavior, c=1.0, full_rank=True),
    ]


def simulate_trajectories(behavior: dict, n_steps: int, n_seeds: int):
    """Vectorized across seeds. Returns states, next_states (int, (n_steps,
    n_seeds)) and rho (float, (n_steps, n_seeds): the importance ratio for
    whichever action each seed actually sampled at that step).
    """
    rho_solid = TARGET_POLICY[SOLID] / behavior[SOLID]
    rho_dashed = TARGET_POLICY[DASHED] / behavior[DASHED]  # always 0.0

    rngs = [np.random.default_rng(seed) for seed in range(n_seeds)]
    cur = np.array([rng.integers(N_STATES) for rng in rngs])

    states = np.empty((n_steps, n_seeds), dtype=int)
    next_states = np.empty((n_steps, n_seeds), dtype=int)
    rho = np.empty((n_steps, n_seeds))

    for t in range(n_steps):
        states[t] = cur
        u = np.array([rng.random() for rng in rngs])
        is_solid = u < behavior[SOLID]
        rho[t] = np.where(is_solid, rho_solid, rho_dashed)
        dashed_targets = np.array([rng.integers(6) for rng in rngs])
        cur = np.where(is_solid, TOP_STATE, dashed_targets)
        next_states[t] = cur

    return states, next_states, rho


def observed_cliff(betas: np.ndarray, frac_diverged: np.ndarray) -> float:
    """Smallest beta where at least half the seeds diverged."""
    order = np.argsort(betas)
    for beta, frac in zip(betas[order], frac_diverged[order]):
        if frac >= 0.5:
            return float(beta)
    return float("nan")


def run_setup(setup: dict, n_steps: int) -> tuple[list[dict], float]:
    phi = build_feature_matrix(c=setup["c"], full_rank=setup["full_rank"])
    n_features = phi.shape[1]
    behavior = setup["behavior"]
    predicted = predicted_beta_cliff(behavior, phi)

    betas = np.linspace(BETA_RANGE[0], BETA_RANGE[1], N_BETAS) * predicted
    alpha_scale = 1.0 / (setup["c"] ** 2)
    alphas = [a * alpha_scale for a in BASE_ALPHAS]

    w_init = CLASSIC_W_INIT[:n_features] if n_features <= len(CLASSIC_W_INIT) else np.ones(n_features)

    states, next_states, rho = simulate_trajectories(behavior, n_steps, N_SEEDS)

    rows = []
    for algorithm in ALGORITHMS:
        for alpha in alphas:
            frac_diverged = np.empty(N_BETAS)
            for i, beta in enumerate(betas):
                diverged = run_batch(phi, states, next_states, rho, GAMMA, alpha, beta,
                                      algorithm, w_init, DIVERGENCE_NORM)
                frac_diverged[i] = diverged.mean()
            observed = observed_cliff(betas, frac_diverged)
            for beta, frac in zip(betas, frac_diverged):
                rows.append(dict(
                    setup=setup["name"], n_steps=n_steps, algorithm=algorithm,
                    alpha=alpha, beta=beta, predicted_cliff=predicted,
                    frac_diverged=frac, observed_cliff=observed,
                ))
    return rows, predicted


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    all_rows = []

    for setup in setups():
        rows, predicted = run_setup(setup, n_steps=20000)
        all_rows.extend(rows)
        print(f"{setup['name']}: predicted beta* = {predicted:.4g}")
        if setup["name"] == "default":
            rows_100k, _ = run_setup(setup, n_steps=100000)
            for r in rows_100k:
                r["setup"] = "default_100k"
            all_rows.extend(rows_100k)

    with open(RESULTS_DIR / "beta_cliff.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    groups = {}
    for r in all_rows:
        groups[(r["setup"], r["algorithm"], r["alpha"])] = r  # observed_cliff is already per-group

    print(f"\n{'setup':<14}{'algorithm':<10}{'alpha':<10}{'predicted':<12}{'observed':<12}")
    for (setup_name, algorithm, alpha), r in sorted(groups.items()):
        print(f"{setup_name:<14}{algorithm:<10}{alpha:<10.4g}{r['predicted_cliff']:<12.4g}{r['observed_cliff']:<12.4g}")

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"GTD2": "o", "TDC": "^"}
    setup_names = sorted(set(k[0] for k in groups))
    colors = dict(zip(setup_names, plt.cm.tab10(np.linspace(0, 1, len(setup_names)))))
    for (setup_name, algorithm, alpha), r in groups.items():
        if not np.isfinite(r["observed_cliff"]):
            continue
        ax.scatter(r["predicted_cliff"], r["observed_cliff"], marker=markers[algorithm],
                   color=colors[setup_name], label=f"{setup_name}/{algorithm}")

    predicted_vals = [r["predicted_cliff"] for r in groups.values()]
    lims = [min(predicted_vals) * 0.7, max(predicted_vals) * 1.4]
    ax.plot(lims, lims, "k--", linewidth=1, label="predicted = observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted beta* = 2 / (rho_max * max||phi||^2)")
    ax.set_ylabel("Observed cliff (smallest beta where >=50% of seeds diverge)")
    ax.set_title("Predicted vs. observed beta divergence cliff")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "beta_cliff_predicted_vs_observed.png", dpi=150)

    print(f"\nDone in {time.time() - t0:.1f}s. Wrote results/beta_cliff.csv and "
          f"results/beta_cliff_predicted_vs_observed.png")


if __name__ == "__main__":
    main()
