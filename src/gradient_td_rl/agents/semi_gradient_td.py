"""Naive off-policy semi-gradient TD(0): the ordinary on-policy TD(0)
update, just reweighted by the importance-sampling ratio rho = pi/b. This
is "semi-gradient" in the usual sense (it ignores the dependence of the
bootstrap target phi(s')^T w on w when differentiating), which is exactly
what makes it not a true gradient method and lets it diverge under
off-policy sampling -- the thing this whole repo is about.
"""
from __future__ import annotations

import numpy as np


class SemiGradientTD:
    def __init__(self, n_features: int, alpha: float, w_init: np.ndarray | None = None):
        self.alpha = alpha
        self.w = np.zeros(n_features) if w_init is None else w_init.copy()

    def update(self, phi_s: np.ndarray, phi_s_next: np.ndarray, reward: float,
               rho: float, gamma: float) -> float:
        delta = reward + gamma * (phi_s_next @ self.w) - (phi_s @ self.w)
        self.w = self.w + self.alpha * rho * delta * phi_s
        return float(delta)
