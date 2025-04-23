"""
Photon deflection solved in u(φ)=1/r.

integrate_photon(r0, b, rs, nsteps, dphi)
    returns r_hist, phi_final, (r_hist, φ_hist)
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


@njit
def _rhs(y: np.ndarray, rs: float) -> np.ndarray:
    # y = (u, w)  with  u = 1/r ,  w = du/dφ
    u, w = y
    du_dphi = w
    dw_dphi = -u + 1.5 * rs * u * u
    return np.array([du_dphi, dw_dphi])


def integrate_photon(
    r0: float,
    b: float,          # kept for signature compat – not needed here
    rs: float,
    nsteps: int = 50_000,
    dphi: float = 2e-4,
):
    """
    Integrate Eq. (★) until the photon has returned to r ≥ r0 on the
    outgoing leg.  This yields an accurate weak-field bending angle.
    """
    u0 = 1.0 / r0
    w0 = -u0                 # straight-line incoming slope

    y = np.array([u0, w0], np.float64)
    u_hist = np.empty(nsteps)
    phi_hist = np.empty_like(u_hist)

    turned = False
    phi = 0.0
    for i in range(nsteps):
        u_hist[i] = 1.0 / y[0]          # store r = 1/u
        phi_hist[i] = phi

        # RK-4 in φ
        k1 = _rhs(y, rs)
        k2 = _rhs(y + 0.5 * dphi * k1, rs)
        k3 = _rhs(y + 0.5 * dphi * k2, rs)
        k4 = _rhs(y + dphi * k3, rs)
        y += (dphi / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        phi += dphi

        if not turned and y[1] > 0.0:        # w = du/dφ changes sign at periapsis
            turned = True
        elif turned and y[0] <= u0:          # back to r ≥ r0
            break

    r_hist = u_hist[: i + 1]
    phi_final = phi_hist[i]
    return r_hist, phi_final, (r_hist, phi_hist[: i + 1])
