"""These tests reproduce, empirically, the qualitative result each of
these algorithms is proven (in Sutton, Maei & Precup, 2009) to have:
semi-gradient off-policy TD(0) diverges on Baird's star regardless of
step size; GTD2 and TDC converge to the MSPBE minimizer. This is the
same "verify the known theoretical behavior actually happens" spirit as
the BFS ground-truth check in tabular-control-rl and the finite-difference
gradient check in deep-q-rl -- not just "the numbers moved."
"""
import numpy as np

from gradient_td_rl.envs.bairds_star import BairdsStar, CLASSIC_W_INIT
from gradient_td_rl.agents.semi_gradient_td import SemiGradientTD
from gradient_td_rl.agents.gtd2 import GTD2
from gradient_td_rl.agents.tdc import TDC
from gradient_td_rl.metrics import run_training


def test_naive_semi_gradient_td_diverges_on_bairds_star():
    env = BairdsStar(seed=0)
    agent = SemiGradientTD(n_features=8, alpha=0.005, w_init=CLASSIC_W_INIT)
    weight_norms, mspbe_values = run_training(agent, env, n_steps=20000, track_every=1000)
    # Exponential blow-up: the final norm should be orders of magnitude
    # above where it started, and still climbing at the end (not just
    # transiently large then settling).
    assert weight_norms[-1] > 1000 * weight_norms[0]
    assert weight_norms[-1] > weight_norms[-5]


def test_gtd2_converges_to_the_mspbe_minimizer():
    env = BairdsStar(seed=0)
    agent = GTD2(n_features=8, alpha=0.005, beta=0.05, w_init=CLASSIC_W_INIT)
    weight_norms, mspbe_values = run_training(agent, env, n_steps=20000, track_every=1000)
    assert mspbe_values[-1] < 1e-2  # started at ~68
    assert weight_norms[-1] < 2 * weight_norms[0]  # bounded, not exploding


def test_tdc_converges_to_the_mspbe_minimizer():
    env = BairdsStar(seed=0)
    agent = TDC(n_features=8, alpha=0.005, beta=0.05, w_init=CLASSIC_W_INIT)
    weight_norms, mspbe_values = run_training(agent, env, n_steps=20000, track_every=1000)
    assert mspbe_values[-1] < 1e-2
    assert weight_norms[-1] < 2 * weight_norms[0]


def test_gtd2_and_tdc_both_become_unstable_at_too_large_a_beta():
    # A genuine, independently discovered sensitivity: beta=0.1 destabilizes
    # both GTD2 and TDC regardless of alpha, which points at v's own update
    # (controlled only by beta) rather than anything about the w update.
    for Agent in (GTD2, TDC):
        env = BairdsStar(seed=0)
        agent = Agent(n_features=8, alpha=0.005, beta=0.1, w_init=CLASSIC_W_INIT)
        weight_norms, _ = run_training(agent, env, n_steps=20000, track_every=1000)
        assert not np.isfinite(weight_norms[-1]) or weight_norms[-1] > 1000 * weight_norms[0]
