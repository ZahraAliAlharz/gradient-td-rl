"""GTD2 (Sutton, Maei & Precup, 2009): true stochastic gradient descent on
the MSPBE. A second weight vector v is used as a fast-timescale estimate
of E[phi phi^T]^-1 E[delta phi] -- the thing needed to turn the MSPBE's
gradient, which involves a product of two expectations, into something
that can be estimated from a single sample without that product's bias
(the "double sampling problem" gradient-TD methods are built to avoid).

Two-timescale stochastic approximation requires v to adapt faster than w
(beta should be large relative to alpha) so that, from w's perspective, v
always looks like it has already converged to its current target.
"""
from __future__ import annotations

import numpy as np


class GTD2:
    def __init__(self, n_features: int, alpha: float, beta: float, w_init: np.ndarray | None = None):
        self.alpha = alpha
        self.beta = beta
        self.w = np.zeros(n_features) if w_init is None else w_init.copy()
        self.v = np.zeros(n_features)

    def update(self, phi_s: np.ndarray, phi_s_next: np.ndarray, reward: float,
               rho: float, gamma: float) -> float:
        delta = reward + gamma * (phi_s_next @ self.w) - (phi_s @ self.w)
        v_dot_phi = phi_s @ self.v
        self.v = self.v + self.beta * rho * (delta - v_dot_phi) * phi_s
        self.w = self.w + self.alpha * rho * (phi_s - gamma * phi_s_next) * v_dot_phi
        return float(delta)
