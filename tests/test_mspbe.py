import numpy as np

from gradient_td_rl.envs.bairds_star import (
    GAMMA, PHI, N_STATES, BEHAVIOR_POLICY, TARGET_POLICY,
    transition_matrix, stationary_distribution,
)
from gradient_td_rl.mspbe import mspbe, mspbe_via_explicit_projection


def _setup():
    p_behavior = transition_matrix(BEHAVIOR_POLICY)
    p_target = transition_matrix(TARGET_POLICY)
    d = stationary_distribution(p_behavior)
    r_pi = np.zeros(N_STATES)
    return p_target, d, r_pi


def test_feature_matrix_is_rank_deficient():
    # Baird's star has 8 features but only 7 states, so Phi (7x8) cannot
    # have 8 linearly independent columns -- there's a genuine nonzero
    # w with Phi @ w = 0. This is why mspbe() must use a pseudoinverse
    # rather than a plain solve: Phi^T D Phi is singular, not just
    # ill-conditioned.
    assert np.linalg.matrix_rank(PHI) == 7
    assert PHI.shape[1] == 8


def test_mspbe_is_zero_at_the_true_fixed_point_w_equals_zero():
    # Reward is 0 everywhere, so V=0 is the true value function under
    # pi, and it is exactly representable (w=0 gives Phi @ 0 = 0) -- so
    # the projected Bellman error at w=0 should be exactly zero.
    p_target, d, r_pi = _setup()
    w = np.zeros(8)
    assert np.isclose(mspbe(w, PHI, d, p_target, r_pi, GAMMA), 0.0, atol=1e-10)


def test_mspbe_is_nonnegative_for_random_weights():
    p_target, d, r_pi = _setup()
    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.normal(size=8)
        assert mspbe(w, PHI, d, p_target, r_pi, GAMMA) >= -1e-10


def test_closed_form_mspbe_matches_independent_explicit_projection():
    # Two independently coded computations of the same mathematical
    # object should agree: the closed-form matrix identity (used during
    # training, for speed) against an explicit projection-then-measure
    # computation (used only here, as a correctness cross-check).
    p_target, d, r_pi = _setup()
    rng = np.random.default_rng(1)
    for _ in range(10):
        w = rng.normal(size=8)
        closed_form = mspbe(w, PHI, d, p_target, r_pi, GAMMA)
        explicit = mspbe_via_explicit_projection(w, PHI, d, p_target, r_pi, GAMMA)
        assert np.isclose(closed_form, explicit, atol=1e-8)
