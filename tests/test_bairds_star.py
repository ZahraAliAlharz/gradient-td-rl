import numpy as np

from gradient_td_rl.envs.bairds_star import (
    BairdsStar, BEHAVIOR_POLICY, TARGET_POLICY, N_STATES, PHI, TOP_STATE,
    transition_matrix, stationary_distribution, importance_ratio, SOLID, DASHED,
)


def test_feature_matrix_matches_bairds_construction():
    # States 0-5: 2 at own index, 1 at the shared last feature.
    for s in range(6):
        expected = np.zeros(8)
        expected[s] = 2.0
        expected[7] = 1.0
        assert np.allclose(PHI[s], expected)
    # State 6 (the "top"/hub state): 1 at index 6, 2 at the shared feature.
    expected_top = np.zeros(8)
    expected_top[6] = 1.0
    expected_top[7] = 2.0
    assert np.allclose(PHI[TOP_STATE], expected_top)


def test_target_policy_transition_matrix_always_goes_to_top_state():
    p_pi = transition_matrix(TARGET_POLICY)
    assert p_pi.shape == (N_STATES, N_STATES)
    for i in range(N_STATES):
        assert np.allclose(p_pi[i], np.eye(N_STATES)[TOP_STATE])


def test_behavior_policy_transition_matrix_is_exactly_uniform():
    # Derived, not assumed: 1/7 chance of "solid" (always -> state 6) plus
    # 6/7 chance of "dashed" (uniform over states 0-5, so 6/7 * 1/6 = 1/7
    # to each) adds up to a uniform 1/7 probability to every state from
    # every state -- a nontrivial consequence of the specific 1/7 vs 6/7
    # policy split, not something assumed up front.
    p_behavior = transition_matrix(BEHAVIOR_POLICY)
    assert np.allclose(p_behavior, np.full((N_STATES, N_STATES), 1.0 / N_STATES))


def test_stationary_distribution_of_behavior_policy_is_uniform():
    p_behavior = transition_matrix(BEHAVIOR_POLICY)
    d = stationary_distribution(p_behavior)
    assert np.allclose(d, np.full(N_STATES, 1.0 / N_STATES), atol=1e-8)
    assert np.isclose(d.sum(), 1.0)


def test_empirical_state_visitation_matches_uniform_stationary_distribution():
    # Independent check of the claim above: simulate the actual Markov
    # chain under the behavior policy and compare empirical visitation
    # frequency to the closed-form 1/7 prediction.
    env = BairdsStar(seed=0)
    state = env.reset()
    counts = np.zeros(N_STATES)
    n_steps = 200_000
    for _ in range(n_steps):
        action = env.sample_behavior_action()
        state, _ = env.step(action)
        counts[state] += 1
    empirical = counts / n_steps
    assert np.allclose(empirical, np.full(N_STATES, 1.0 / N_STATES), atol=0.01)


def test_importance_ratio():
    assert np.isclose(importance_ratio(SOLID), 7.0)
    assert np.isclose(importance_ratio(DASHED), 0.0)
