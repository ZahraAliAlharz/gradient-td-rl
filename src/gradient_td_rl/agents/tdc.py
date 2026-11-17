"""TDC / GTD(0) with gradient correction (Sutton, Maei & Precup, 2009):
like GTD2, but the w-update keeps the ordinary TD(0) update term
delta * phi_s and adds a correction term, rather than replacing it
entirely the way GTD2 does. In the paper's experiments this version
usually converges faster in practice than GTD2 while sharing the same
convergence guarantee.
"""
from __future__ import annotations

import numpy as np


class TDC:
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
        self.w = self.w + self.alpha * rho * (delta * phi_s - gamma * phi_s_next * v_dot_phi)
        return float(delta)
