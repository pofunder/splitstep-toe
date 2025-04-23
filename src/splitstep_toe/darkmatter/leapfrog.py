"""
Ultra-simplified two-body Leapfrog integrator
(units G = 1, masses m1 = m2 = 1)
"""
from __future__ import annotations
import numpy as np

def leapfrog_step(x: np.ndarray,
                  v: np.ndarray,
                  dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One velocity-Verlet step for two bodies in 3-D.

    Parameters
    ----------
    x  (2,3) array – positions
    v  (2,3) array – velocities
    dt float        – timestep

    Returns
    -------
    x_new, v_new : updated arrays
    """
    r_vec = x[1] - x[0]
    r     = np.linalg.norm(r_vec)
    a     = r_vec / r**3                          # accel on m1 (m2 opposite)

    v_half     = v.copy()
    v_half[0] -= 0.5 * dt * a
    v_half[1] += 0.5 * dt * a

    x_new = x + dt * v_half

    r_vec_new = x_new[1] - x_new[0]
    r_new     = np.linalg.norm(r_vec_new)
    a_new     = r_vec_new / r_new**3

    v_new     = v_half.copy()
    v_new[0] -= 0.5 * dt * a_new
    v_new[1] += 0.5 * dt * a_new

    return x_new, v_new

# ---------------------------------------------------------------------- #
def demo_two_body(n_steps: int = 1_000,
                  dt: float = 1e-3) -> np.ndarray:
    """
    Evolve two equal-mass halos; return their separation vs time.

    Starts with R = 1, v_tangential = 0.5 circularish.
    """
    x = np.array([[-0.5, 0, 0],
                  [ 0.5, 0, 0]], dtype=float)
    v = np.array([[ 0,  0.5, 0],
                  [ 0, -0.5, 0]], dtype=float)

    sep = np.empty(n_steps)
    for i in range(n_steps):
        x, v = leapfrog_step(x, v, dt)
        sep[i] = np.linalg.norm(x[1] - x[0])
    return sep
