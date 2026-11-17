"""Reproduces the central comparison of Sutton, Maei & Precup (2009) on
Baird's star: naive off-policy semi-gradient TD(0) diverges, GTD2 and TDC
converge to the (closed-form, exactly computed) MSPBE minimizer. Also
sweeps beta to show the two-timescale step-size sensitivity found while
tuning these algorithms.

Generates results/bairds_divergence_vs_convergence.png and
results/bairds_beta_sensitivity.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_td_rl.agents.gtd2 import GTD2
from gradient_td_rl.agents.semi_gradient_td import SemiGradientTD
from gradient_td_rl.agents.tdc import TDC
from gradient_td_rl.envs.bairds_star import BairdsStar, CLASSIC_W_INIT
from gradient_td_rl.metrics import run_training

N_STEPS = 20000
TRACK_EVERY = 200
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    agents = {
        "Semi-gradient TD(0) (alpha=0.005)": SemiGradientTD(8, alpha=0.005, w_init=CLASSIC_W_INIT),
        "GTD2 (alpha=0.005, beta=0.05)": GTD2(8, alpha=0.005, beta=0.05, w_init=CLASSIC_W_INIT),
        "TDC (alpha=0.005, beta=0.05)": TDC(8, alpha=0.005, beta=0.05, w_init=CLASSIC_W_INIT),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, agent in agents.items():
        env = BairdsStar(seed=0)
        weight_norms, mspbe_values = run_training(agent, env, n_steps=N_STEPS, track_every=TRACK_EVERY)
        steps = np.arange(len(weight_norms)) * TRACK_EVERY
        ax1.plot(steps, weight_norms, label=name)
        ax2.plot(steps, np.clip(mspbe_values, 1e-8, None), label=name)
        print(f"{name}: start ||w||={weight_norms[0]:.2f}, end ||w||={weight_norms[-1]:.3g}, "
              f"start MSPBE={mspbe_values[0]:.3g}, end MSPBE={mspbe_values[-1]:.3g}")

    ax1.set_yscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("||w||")
    ax1.set_title("Weight-vector norm")
    ax1.legend(fontsize=8)

    ax2.set_yscale("log")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("MSPBE (exact, closed-form)")
    ax2.set_title("Mean-Squared Projected Bellman Error")
    ax2.legend(fontsize=8)

    fig.suptitle("Baird's counterexample: divergence vs. provably convergent gradient-TD methods")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bairds_divergence_vs_convergence.png", dpi=150)

    # Beta sensitivity: holding alpha fixed, sweep beta for both GTD2 and
    # TDC to show the two-timescale requirement is a real, visible
    # boundary here, not a theoretical nicety.
    betas = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for Agent, name in [(GTD2, "GTD2"), (TDC, "TDC")]:
        final_mspbe = []
        for beta in betas:
            env = BairdsStar(seed=0)
            agent = Agent(8, alpha=0.005, beta=beta, w_init=CLASSIC_W_INIT)
            _, mspbe_values = run_training(agent, env, n_steps=N_STEPS, track_every=N_STEPS - 1)
            final_mspbe.append(np.clip(mspbe_values[-1], 1e-8, 1e30))
        ax.plot(betas, final_mspbe, marker="o", label=name)
        print(f"{name} beta sweep, final MSPBE: " +
              ", ".join(f"beta={b}->{v:.3g}" for b, v in zip(betas, final_mspbe)))

    ax.axvline(0.005, color="gray", linestyle=":", linewidth=1, label="alpha (fixed)")
    ax.set_yscale("log")
    ax.set_xlabel("beta (fast-timescale step size, alpha=0.005 fixed)")
    ax.set_ylabel("Final MSPBE after 20000 steps")
    ax.set_title("Two-timescale sensitivity: beta too large destabilizes both methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bairds_beta_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()
