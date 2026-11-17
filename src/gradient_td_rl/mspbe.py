"""The Mean-Squared Projected Bellman Error (MSPBE), computed exactly by
linear algebra rather than estimated from samples -- possible here only
because Baird's star is small enough (7 states) to write out the full
feature matrix, transition matrices, and stationary distribution in
closed form.

MSPBE(w) = (Phi^T D delta_bar(w))^T (Phi^T D Phi)^-1 (Phi^T D delta_bar(w))

where delta_bar(w) = R_pi + gamma * P_pi @ (Phi w) - Phi w is the vector,
over all states, of expected one-step TD errors under the target policy
pi, and D is diagonal with the behavior policy's stationary distribution
on the diagonal (Sutton, Maei & Precup, 2009, Eq. 7). This is exactly the
objective GTD2 and TDC are proven to minimize via stochastic gradient
descent, so tracking it during training is a direct check against theory,
not a proxy for it.
"""
from __future__ import annotations

import numpy as np


def expected_td_error_vector(w: np.ndarray, phi: np.ndarray, p_pi: np.ndarray,
                              r_pi: np.ndarray, gamma: float) -> np.ndarray:
    v = phi @ w
    return r_pi + gamma * (p_pi @ v) - v


def mspbe(w: np.ndarray, phi: np.ndarray, d: np.ndarray, p_pi: np.ndarray,
          r_pi: np.ndarray, gamma: float) -> float:
    """Note: Phi^T D Phi is singular here -- Baird's star has 8 features
    but only 7 states, so rank(Phi) <= 7 < 8 (confirmed directly via
    np.linalg.matrix_rank(PHI) == 7 in tests/test_mspbe.py). There really
    is a nonzero w with Phi @ w = 0: the feature construction is
    deliberately over-parameterized, which is part of what makes the
    example interesting (semi-gradient TD can diverge partly along a
    direction that doesn't even change the predicted values). We use the
    Moore-Penrose pseudoinverse, which gives the minimum-norm solution
    and is the standard fix when a feature matrix isn't full column rank.
    """
    delta_bar = expected_td_error_vector(w, phi, p_pi, r_pi, gamma)
    d_diag = np.diag(d)
    a = phi.T @ d_diag @ phi
    b = phi.T @ d_diag @ delta_bar
    return float(b @ np.linalg.pinv(a) @ b)


def mspbe_via_explicit_projection(w: np.ndarray, phi: np.ndarray, d: np.ndarray,
                                   p_pi: np.ndarray, r_pi: np.ndarray, gamma: float) -> float:
    """Independent code path for the same quantity: build the projection
    operator onto span(Phi) in the D-weighted inner product explicitly,
    project the Bellman target, and measure the squared D-weighted
    distance directly -- rather than using the closed-form matrix
    identity above. Used only as a cross-check in tests.
    """
    d_diag = np.diag(d)
    projection = phi @ np.linalg.pinv(phi.T @ d_diag @ phi) @ phi.T @ d_diag
    v = phi @ w
    bellman_target = r_pi + gamma * (p_pi @ v)
    projected_target = projection @ bellman_target
    diff = v - projected_target
    return float(diff @ d_diag @ diff)
