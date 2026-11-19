"""A seed-vectorized GTD2/TDC update, mathematically identical to
agents/gtd2.py and agents/tdc.py (cross-checked against them directly in
tests/test_vectorized.py) but applied to many independent seeds' worth of
weights at once via numpy array ops. Used by experiments/sweep_beta_cliff.py,
where the sweep's (setup x algorithm x alpha x beta) grid is too large to
run through the scalar per-step agent classes one seed at a time within a
reasonable wall-clock budget.
"""
from __future__ import annotations

import numpy as np


def update_batch(w: np.ndarray, v: np.ndarray, phi_s: np.ndarray, phi_s_next: np.ndarray,
                  rho: np.ndarray, gamma: float, alpha: float, beta: float, algorithm: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One step for a batch of independent seeds.

    w, v: (n_seeds, n_features). phi_s, phi_s_next: (n_seeds, n_features).
    rho: (n_seeds,). Returns the updated (w, v), same shapes as the input.
    """
    v_dot_phi = np.einsum("sf,sf->s", phi_s, v)
    delta = gamma * np.einsum("sf,sf->s", phi_s_next, w) - np.einsum("sf,sf->s", phi_s, w)

    v_new = v + (beta * rho * (delta - v_dot_phi))[:, None] * phi_s

    if algorithm == "GTD2":
        w_new = w + (alpha * rho * v_dot_phi)[:, None] * (phi_s - gamma * phi_s_next)
    elif algorithm == "TDC":
        w_new = w + alpha * rho[:, None] * (delta[:, None] * phi_s - gamma * v_dot_phi[:, None] * phi_s_next)
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    return w_new, v_new


def run_batch(phi: np.ndarray, states: np.ndarray, next_states: np.ndarray, actions_rho: np.ndarray,
              gamma: float, alpha: float, beta: float, algorithm: str, w_init: np.ndarray,
              divergence_norm: float) -> np.ndarray:
    """Runs n_seeds independent chains (one per column of states/next_states/
    actions_rho) through update_batch for the whole trajectory, returning a
    bool array of shape (n_seeds,): did that seed ever diverge?

    states, next_states: (n_steps, n_seeds) state indices.
    actions_rho: (n_steps, n_seeds) the importance ratio for the action
    actually taken at that step (0.0 on "dashed" steps, rho_max on "solid").
    w_init: (n_features,), broadcast to all seeds at the start.
    """
    n_steps, n_seeds = states.shape
    n_features = phi.shape[1]
    w = np.tile(w_init, (n_seeds, 1)).astype(float)
    v = np.zeros((n_seeds, n_features))
    diverged = np.zeros(n_seeds, dtype=bool)

    for t in range(n_steps):
        phi_s = phi[states[t]]
        phi_sn = phi[next_states[t]]
        w, v = update_batch(w, v, phi_s, phi_sn, actions_rho[t], gamma, alpha, beta, algorithm)

        norms = np.linalg.norm(w, axis=1)
        diverged |= ~np.isfinite(norms) | (norms > divergence_norm)
        if diverged.all():
            break

    return diverged
