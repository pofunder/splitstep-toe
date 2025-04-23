"""
Ultra-simplified two-body Leapfrog integrator
(units G = 1, masses m₁ = m₂ = 1).
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------- #
# 1. Velocity-Verlet step                                               #
# --------------------------------------------------------------------- #
def leapfrog_step(x: np.ndarray,
                  v: np.ndarray,
                  dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One Leapfrog (velocity-Verlet) step for two equal-mass bodies.

    Parameters
    ----------
    x  : ndarray, shape (2, 3)
         Cartesian positions of the two bodies
    v  : ndarray, shape (2, 3)
         Cartesian velocities
    dt : float
         timestep

    Returns
    -------
    x_new, v_new : updated positions & velocities
    """
    # acceleration on body 0 from body 1  (and opposite for body 1)
    r_vec = x[1] - x[0]
    r     = np.linalg.norm(r_vec)
    a     = -r_vec / r**3                  # G = m₁ = m₂ = 1

    # half-kick
    v_half      = v.copy()
    v_half[0]  -= 0.5 * dt * a
    v_half[1]  += 0.5 * dt * a

    # drift
    x_new = x + dt * v_half

    # new acceleration
    r_vec_new = x_new[1] - x_new[0]
    r_new     = np.linalg.norm(r_vec_new)
    a_new     = r_vec_new / r_new**3

    # second half-kick
    v_new      = v_half.copy()
    v_new[0]  -= 0.5 * dt * a_new
    v_new[1]  += 0.5 * dt * a_new

    return x_new, v_new


# --------------------------------------------------------------------- #
# 2. Demo function used by the test                                     #
# --------------------------------------------------------------------- #

# Two orbits with dt = 5×10⁻⁴  → 1 800 steps per revolution, energy ≤10⁻³
def demo_two_body(n_steps: int = 4_000,
                 dt: float = 5e-4) -> np.ndarray:
    """
    Run a two-body orbit and return separation vs time.

    Starts at separation R = 1 with the circular velocity
        v_circ = √(G M / (4 R))  →  1/√2  in our units.
    """
    x = np.array([[-0.5, 0.0, 0.0],
                  [ 0.5, 0.0, 0.0]], dtype=float)

    v_circ = 1.0 / np.sqrt(2.0)           # circular speed at R = 1
    v = np.array([[ 0.0,  v_circ, 0.0],
                  [ 0.0, -v_circ, 0.0]], dtype=float)

    sep = np.empty(n_steps)
    for i in range(n_steps):
        x, v = leapfrog_step(x, v, dt)
        sep[i] = np.linalg.norm(x[1] - x[0])

    return sep
