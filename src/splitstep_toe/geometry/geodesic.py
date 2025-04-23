"""
Photon geodesics in Schwarzschild space-time (geometric units c = G = 1).

integrate_photon(r0, b, rs, nsteps, dl)
    * r0      : start radius  (≫ rs), equatorial plane, φ = 0
    * b       : impact parameter  (conserved angular-momentum L = b E, E = 1)
    * rs      : Schwarzschild radius 2 GM
    * nsteps  : RK-4 integration steps
    * dl      : affine-parameter step size

Returns
-------
r_hist  : ndarray, radial coordinate along path  (length ≤ nsteps)
phi_fin : final azimuthal angle when photon has turned and escaped
path    : (r_hist, φ_hist)   # for plotting or GIFs
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


# ──────────────────────────────────────────────────────────────────────────────
#   d/dλ (r, φ, p_r)  for null geodesic in Schwarzschild
#
#   (see e.g. Misner–Thorne–Wheeler §25.2)
#
#     r″ = L² ( 1/r³  –  3 rs / (2 r⁴) )
#     φ′ = L / r²
#     r′ = p_r
# ──────────────────────────────────────────────────────────────────────────────
@njit
def _rhs(y: np.ndarray, rs: float, L: float) -> np.ndarray:
    r, phi, pr = y                    # y = (r, φ, p_r)
    dr   = pr
    dphi = L / r**2
    dpr  = L**2 * (1.0 / r**3 - 0.5 * rs / r**4)
    return np.array([dr, dphi, dpr])


# ──────────────────────────────────────────────────────────────────────────────
def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 6000,
    dl: float = 0.02,
):
    """
    Fourth-order Runge–Kutta integration of a photon trajectory.
    Stops once r has passed periapsis and is increasing again.
    """
    L = b

    # inward-moving initial radial momentum (choose the negative root)
    f0 = 1.0 - rs / r0
    pr0 = -np.sqrt(1.0 - f0 * L**2 / r0**2)

    y = np.array([r0, 0.0, pr0])          # (r, φ, p_r)
    r_hist  = np.empty(nsteps, dtype=np.float64)
    phi_hist = np.empty_like(r_hist)

    for i in range(nsteps):
        r_hist[i]   = y[0]
        phi_hist[i] = y[1]

        # one RK-4 step
        k1 = _rhs(y, rs, L)
        k2 = _rhs(y + 0.5 * dl * k1, rs, L)
        k3 = _rhs(y + 0.5 * dl * k2, rs, L)
        k4 = _rhs(y + dl  * k3, rs, L)
        y += (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # turn-around: p_r switches from negative to positive
        if i > 10 and y[2] > 0:
            break

    return r_hist[: i + 1], y[1], (r_hist[: i + 1], phi_hist[: i + 1])
