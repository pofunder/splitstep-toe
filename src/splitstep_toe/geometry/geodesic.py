"""
Photon geodesics in Schwarzschild space-time  (c = G = 1).

integrate_photon(r0, b, rs, nsteps, dl)
    → r_trace, phi_final, (r_trace, φ_trace)
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


@njit
def _rhs(y: np.ndarray, rs: float, L: float) -> np.ndarray:
    """
    First-order system y = (r, φ, p_r) for null geodesic.

        dr/dλ     = p_r
        dφ/dλ     = L / r²
        dp_r/dλ   = −rs/(2r²f)  +  rs p_r²/(2r² f)  +  f L² / r³
                     where f = 1 − rs / r
    """
    r, phi, pr = y
    f = 1.0 - rs / r

    dr   = pr
    dphi = L / r**2
    dpr  = -0.5 * rs / (r**2 * f) + 0.5 * rs * pr**2 / (r**2 * f) + f * L**2 / r**3
    return np.array([dr, dphi, dpr])


def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 8000,
    dl: float = 0.02,
):
    """
    4-th order RK integration until the photon has returned to r ≥ r0 on
    its outgoing leg (so bending angle is measured at infinity).
    """
    L = b
    f0 = 1.0 - rs / r0
    pr0 = -np.sqrt(1.0 - f0 * L**2 / r0**2)     # inward root

    y = np.array([r0, 0.0, pr0], dtype=np.float64)
    r_hist   = np.empty(nsteps)
    phi_hist = np.empty_like(r_hist)

    turned = False
    for i in range(nsteps):
        r_hist[i]   = y[0]
        phi_hist[i] = y[1]

        # RK-4 step
        k1 = _rhs(y, rs, L)
        k2 = _rhs(y + 0.5 * dl * k1, rs, L)
        k3 = _rhs(y + 0.5 * dl * k2, rs, L)
        k4 = _rhs(y + dl  * k3, rs, L)
        y += (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # periapsis reached → p_r changes sign to positive
        if not turned and y[2] > 0.0:
            turned = True
        # once outgoing, stop when r ≥ start radius
        elif turned and y[0] >= r0:
            break

    return r_hist[: i + 1], y[1], (r_hist[: i + 1], phi_hist[: i + 1])
