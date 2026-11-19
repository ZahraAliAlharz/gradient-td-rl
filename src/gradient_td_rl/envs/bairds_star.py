"""Baird's counterexample (Baird, 1995; Sutton & Barto, 2018, Example 11.4):
a 7-state Markov chain built specifically to show that off-policy
semi-gradient TD(0) with linear function approximation can diverge, even
though the true value function (all zeros here) is exactly representable
by the function class.

Seven states. Two actions: "solid" always transitions to state 6 (the
last state, 0-indexed); "dashed" transitions uniformly to one of states
0-5. Reward is 0 everywhere, so the true value function under any policy
is V = 0.

- Target policy pi: always "solid" (deterministically ends up at state 6).
- Behavior policy b: "solid" with probability 1/7, "dashed" with
  probability 6/7 -- i.e. we're learning about pi while only acting
  according to b, which is what makes this off-policy.

Each state s has an 8-dimensional feature vector phi(s), deliberately
chosen (Baird, 1995) so that w=0 is a solution the TD fixed point can
represent, yet the off-policy update direction is not a contraction --
this is exactly the gap gradient-TD methods were built to close.
"""
from __future__ import annotations

import numpy as np

N_STATES = 7
N_FEATURES = 8
TOP_STATE = 6  # 0-indexed: state 7 in Baird's original 1-indexed figure

SOLID, DASHED = 0, 1
TARGET_POLICY = {SOLID: 1.0, DASHED: 0.0}
BEHAVIOR_POLICY = {SOLID: 1.0 / 7.0, DASHED: 6.0 / 7.0}

GAMMA = 0.99

# Classic initialization from Sutton & Barto (2018), Figure 11.2 -- chosen
# so that the fixed point w=0 is visibly far from the starting point.
CLASSIC_W_INIT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])


def build_feature_matrix(c: float = 1.0, full_rank: bool = False) -> np.ndarray:
    """phi(s) for s = 0..5: 2 at position s, 1 at the shared last feature.
    phi(6) (the top state): 1 at position 6, 2 at the shared last feature.

    full_rank=True drops Baird's deliberate over-parameterization and uses
    one-hot features instead (phi = c * I, 7 features for 7 states) --
    trivially full column rank, for testing whether the beta-cliff
    predicted below is specific to Baird's rank-deficient construction.
    c scales every feature, for testing whether the predicted cliff moves
    the way the formula says it should as ||phi|| changes.
    """
    if full_rank:
        return c * np.eye(N_STATES)
    phi = np.zeros((N_STATES, N_FEATURES))
    for s in range(6):
        phi[s, s] = 2.0
        phi[s, 7] = 1.0
    phi[TOP_STATE, TOP_STATE] = 1.0
    phi[TOP_STATE, 7] = 2.0
    return c * phi


PHI = build_feature_matrix()


def importance_ratio(action: int, behavior_policy: dict[int, float] = BEHAVIOR_POLICY) -> float:
    return TARGET_POLICY[action] / behavior_policy[action]


def predicted_beta_cliff(behavior_policy: dict[int, float], phi: np.ndarray) -> float:
    """beta* = 2 / (rho_max * max_s ||phi(s)||^2): the step size at which
    GTD2/TDC's v-update, v += beta*rho*(delta - phi@v)*phi, stops being a
    contraction along the active feature direction (the homogeneous gain
    there is 1 - beta*rho*||phi||^2). This is a necessary-condition bound
    for v's own recursion treating w as fixed (the two-timescale
    assumption) -- not a full joint-stability proof -- and it is the
    structural match for the classic LMS/stochastic-approximation
    step-size bound (mu*||phi||^2 < 2 for x += mu*e*phi).
    """
    rho_max = max(importance_ratio(a, behavior_policy) for a in (SOLID, DASHED))
    max_sq_norm = float(np.max(np.sum(phi ** 2, axis=1)))
    return 2.0 / (rho_max * max_sq_norm)


def transition_matrix(policy: dict[int, float]) -> np.ndarray:
    """P[i, j] = probability of moving from state i to state j under the
    given policy's action probabilities. Built generically from the
    policy dict and the environment's actual transition rule, not
    hand-asserted -- so that, e.g., discovering P_behavior is uniform
    is a real computation, not a shortcut.
    """
    p = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        p[i, TOP_STATE] += policy[SOLID]
        for j in range(6):
            p[i, j] += policy[DASHED] * (1.0 / 6.0)
    return p


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    """Solve d^T P = d^T, sum(d) = 1 via the left eigenvector of P for
    eigenvalue 1 (equivalently, the right eigenvector of P^T).
    """
    eigvals, eigvecs = np.linalg.eig(p.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    d = np.real(eigvecs[:, idx])
    d = d / d.sum()
    return d


class BairdsStar:
    def __init__(self, seed: int | None = None, feature_scale: float = 1.0,
                 p_solid: float = BEHAVIOR_POLICY[SOLID], full_rank: bool = False):
        self.rng = np.random.default_rng(seed)
        self.state = 0
        self.behavior_policy = {SOLID: p_solid, DASHED: 1.0 - p_solid}
        self.phi = build_feature_matrix(c=feature_scale, full_rank=full_rank)

    def reset(self) -> int:
        self.state = int(self.rng.integers(N_STATES))
        return self.state

    def sample_behavior_action(self) -> int:
        return SOLID if self.rng.random() < self.behavior_policy[SOLID] else DASHED

    def step(self, action: int) -> tuple[int, float]:
        if action == SOLID:
            next_state = TOP_STATE
        else:
            next_state = int(self.rng.integers(6))
        self.state = next_state
        return next_state, 0.0

    def feature(self, state: int) -> np.ndarray:
        return self.phi[state]
