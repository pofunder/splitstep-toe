"""Finite‑difference 2‑D Laplacian (5‑point stencil, Dirichlet BC)."""

import numpy as np

__all__ = ["laplacian_2d"]


def laplacian_2d(u: np.ndarray, h: float = 1.0) -> np.ndarray:
    """Return ∇²u for a 2‑D array *u* using a 5‑point stencil.

    The boundary is forced to zero (Dirichlet).  Grid spacing *h*.
    """
    lap = (
        -4.0 * u
        + np.roll(u,  1, 0) + np.roll(u, -1, 0)
        + np.roll(u,  1, 1) + np.roll(u, -1, 1)
    ) / (h * h)

    # keep the boundaries zero so the wrap‑around rolls don’t leak in
    lap[0, :] = lap[-1, :] = lap[:, 0] = lap[:, -1] = 0.0
    return lap

