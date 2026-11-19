"""The seed-vectorized update in vectorized.py is a separate implementation
of the same GTD2/TDC math in agents/gtd2.py and agents/tdc.py, written for
speed (experiments/sweep_beta_cliff.py needs to run many seeds at once).
A reimplementation that quietly drifted from the real agents would make the
whole sweep meaningless, so this checks step-by-step agreement against the
actual scalar classes on a real trajectory -- the same "two independent code
paths must agree" standard test_mspbe.py already holds the MSPBE computation
to.
"""
import numpy as np

from gradient_td_rl.agents.gtd2 import GTD2
from gradient_td_rl.agents.tdc import TDC
from gradient_td_rl.envs.bairds_star import BairdsStar, CLASSIC_W_INIT, PHI
from gradient_td_rl.vectorized import update_batch


def _check_matches_scalar_agent(agent_cls, algorithm_name):
    env = BairdsStar(seed=0)
    agent = agent_cls(n_features=8, alpha=0.005, beta=0.05, w_init=CLASSIC_W_INIT)

    w_batch = CLASSIC_W_INIT.reshape(1, -1).astype(float)
    v_batch = np.zeros((1, 8))

    state = env.reset()
    for _ in range(500):
        action = env.sample_behavior_action()
        next_state, reward = env.step(action)
        rho = 7.0 if action == 0 else 0.0  # SOLID=0, DASHED=1 in this env

        agent.update(PHI[state], PHI[next_state], reward, rho, gamma=0.99)

        w_batch, v_batch = update_batch(
            w_batch, v_batch,
            PHI[state].reshape(1, -1), PHI[next_state].reshape(1, -1),
            np.array([rho]), gamma=0.99, alpha=0.005, beta=0.05, algorithm=algorithm_name,
        )

        assert np.allclose(agent.w, w_batch[0], atol=1e-10)
        assert np.allclose(agent.v, v_batch[0], atol=1e-10)

        state = next_state


def test_vectorized_gtd2_matches_scalar_gtd2_class():
    _check_matches_scalar_agent(GTD2, "GTD2")


def test_vectorized_tdc_matches_scalar_tdc_class():
    _check_matches_scalar_agent(TDC, "TDC")


def test_update_batch_runs_several_seeds_independently():
    # Two seeds with different phi should produce different, independent
    # updates -- i.e. no accidental mixing across the batch dimension.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(2, 8))
    v = rng.normal(size=(2, 8))
    phi_s = np.stack([PHI[0], PHI[1]])
    phi_sn = np.stack([PHI[6], PHI[6]])
    rho = np.array([7.0, 7.0])

    w_batch, v_batch = update_batch(w, v, phi_s, phi_sn, rho, gamma=0.99, alpha=0.005,
                                     beta=0.05, algorithm="TDC")

    w0, v0 = update_batch(w[0:1], v[0:1], phi_s[0:1], phi_sn[0:1], rho[0:1],
                           gamma=0.99, alpha=0.005, beta=0.05, algorithm="TDC")
    w1, v1 = update_batch(w[1:2], v[1:2], phi_s[1:2], phi_sn[1:2], rho[1:2],
                           gamma=0.99, alpha=0.005, beta=0.05, algorithm="TDC")

    assert np.allclose(w_batch[0], w0[0])
    assert np.allclose(w_batch[1], w1[0])
    assert np.allclose(v_batch[0], v0[0])
    assert np.allclose(v_batch[1], v1[0])
