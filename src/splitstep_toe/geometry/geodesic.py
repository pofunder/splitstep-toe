"""
Photon geodesics in Schwarzschild space-time  (c = G = 1).

integrate_photon(r0, b, rs, nsteps, dl) → r_hist, phi_final, (r_hist, φ_hist)
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


@njit
def _rhs(y: np.ndarray, rs: float, L: float) -> np.ndarray:
    """
    First-order system for a null geodesic:
        y = (r, φ, p_r)  with  p_r = dr/dλ   (affine parameter λ)

    Equations (see MTW §25.2 or Wald §6.1):
        dr/dλ     = p_r
        dφ/dλ     = L / r²
        dp_r/dλ   =  L² (1/r³ − r_s / r⁴)  −  r_s p_r² / [2 r (1−r_s/r)]
    """
    r, phi, pr = y
    f = 1.0 - rs / r

    dr   = pr
    dphi = L / r**2
    dpr  = L**2 * (1.0 / r**3 - rs / r**4) - 0.5 * rs * pr**2 / (r * f)
    return np.array([dr, dphi, dpr])


def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 6000,
    dl: float = 0.02,
):
    """
    4-th order Runge–Kutta integration until the photon turns around
    (p_r changes sign from − to +).

    Parameters
    ----------
    r0   : start radius (≫ r_s)
    b    : impact parameter  (L = b, energy E = 1)
    rs   : Schwarzschild radius 2GM
    nsteps, dl : RK-4 control
    """
    L = b
    f0 = 1.0 - rs / r0
    pr0 = -np.sqrt(1.0 - f0 * L**2 / r0**2)     # inward root

    y = np.array([r0, 0.0, pr0])
    r_hist   = np.empty(nsteps)
    phi_hist = np.empty_like(r_hist)

    for i in range(nsteps):
        r_hist[i]   = y[0]
        phi_hist[i] = y[1]

        k1 = _rhs(y, rs, L)
        k2 = _rhs(y + 0.5 * dl * k1, rs, L)
        k3 = _rhs(y + 0.5 * dl * k2, rs, L)
        k4 = _rhs(y + dl  * k3, rs, L)
        y += (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if i > 10 and y[2] > 0.0:          # photon now outgoing
            break

    return r_hist[: i + 1], y[1], (r_hist[: i + 1], phi_hist[: i + 1])
