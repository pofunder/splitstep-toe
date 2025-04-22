import numpy as np
from .laplacian import laplacian_2d

__all__ = ["step_2d"]

def step_2d(R_prev: np.ndarray,
            R_curr: np.ndarray,
            kappa: float,
            lam: float,
            gamma: float,
            h: float) -> np.ndarray:
    """
    One explicit split‑step of the 2‑D diffusion‑wave equation used
    in the wave‑speed benchmark.

    R_prev, R_curr : 2‑D arrays (previous two time‑levels)
    Returns the next field array (same shape).
    """
    lap = laplacian_2d(R_curr, h)
    R_next = (2 + lam - gamma) * R_curr \
             - (1 - lam) * R_prev \
             + kappa * lap
    return R_next
