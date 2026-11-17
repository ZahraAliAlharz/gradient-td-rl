from __future__ import annotations

import numpy as np

from gradient_td_rl.envs.bairds_star import (
    GAMMA, PHI, BairdsStar, importance_ratio, stationary_distribution,
    transition_matrix, BEHAVIOR_POLICY, TARGET_POLICY, N_STATES,
)
from gradient_td_rl.mspbe import mspbe


def run_training(agent, env: BairdsStar, n_steps: int, track_every: int = 1):
    """Runs the agent for n_steps behavior-policy transitions, returning
    the weight-norm trajectory and the exact MSPBE trajectory (tracked
    via closed-form linear algebra, not estimated from the run itself).
    """
    p_behavior = transition_matrix(BEHAVIOR_POLICY)
    p_target = transition_matrix(TARGET_POLICY)
    d = stationary_distribution(p_behavior)
    r_pi = np.zeros(N_STATES)  # reward is 0 everywhere in Baird's star

    weight_norms = []
    mspbe_values = []

    state = env.reset()
    for step in range(n_steps):
        action = env.sample_behavior_action()
        next_state, reward = env.step(action)
        rho = importance_ratio(action)

        agent.update(PHI[state], PHI[next_state], reward, rho, GAMMA)
        state = next_state

        if step % track_every == 0:
            weight_norms.append(float(np.linalg.norm(agent.w)))
            mspbe_values.append(mspbe(agent.w, PHI, d, p_target, r_pi, GAMMA))

    return np.array(weight_norms), np.array(mspbe_values)
