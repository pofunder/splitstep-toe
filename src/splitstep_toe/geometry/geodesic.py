"""
Photon geodesics in Schwarzschild space-time  (c = G = 1).

integrate_photon(r0, b, rs, nsteps, dl)
    returns (r_arr, phi_final, path) where
        r0      : starting radius  (≫ rs)
        b       : impact parameter (L)
        rs      : Schwarzschild radius 2 GM
        nsteps  : RK4 steps
        dl      : affine-parameter step

Works for weak and moderate bending; terminates when r starts growing again.
"""
from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]

@njit
def _rhs(y: np.ndarray, rs: float) -> np.ndarray:
    r, phi, pr = y
    f = 1.0 - rs / r
    dphi = 1.0 / r**2            # L = 1
    dr   = pr
    dpr  = (rs / (2 * r**2)) * pr**2 / f - f * (1.0 / r**3)
    return np.array([dr, dphi, dpr])

def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 8000,
    dl: float = 0.02,
):
    # initial conditions (incoming along +x, closest approach at y axis)
    r, phi, pr = r0, 0.0, -np.sqrt(1 - (b / r0) ** 2)
    y = np.array([r, phi, pr])
    path_r = np.empty(nsteps, dtype=np.float64)
    path_phi = np.empty_like(path_r)

    for i in range(nsteps):
        # store radial coordinate for optional path output
        path_r[i] = y[0]
        path_phi[i] = y[1]

        # one RK4 step
        k1 = _rhs(y, rs)
        k2 = _rhs(y + 0.5 * dl * k1, rs)
        k3 = _rhs(y + 0.5 * dl * k2, rs)
        k4 = _rhs(y + dl * k3, rs)
        y += (dl / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # turn-around (r growing again) ⇒ photon on its way out
        if i > 10 and y[2] > 0:
            break

    return path_r[: i + 1], y[1], (path_r[: i + 1], path_phi[: i + 1])
