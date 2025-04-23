"""
Photon geodesics in Schwarzschild spacetime (c = G = 1).

We integrate, in φ, the standard first-order system

    u'' + u = (3/2) r_s u^2     with  u = 1 / r

using fourth-order Runge–Kutta.  The exact first integral

    (u')² = 1/b² − u² + r_s u³

yields the correct incoming slope at r = r0.

By convention the unit tests assume the photon starts far away on
the +y axis, so φ = π/2 at the initial point.  We therefore add
π/2 to all stored angles before returning.
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


# --------------------------------------------------------------------------- #
@njit
def _rhs(y: np.ndarray, rs: float) -> np.ndarray:
    """Right-hand side of the first-order system  y = [u, w]  with  w = du/dφ."""
    u, w = y
    return np.array([w, -u + 1.5 * rs * u * u])


# --------------------------------------------------------------------------- #
def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    *,
    nsteps: int = 8_000,
    dl: float = 0.02,           # tests pass dl= keyword
):
    """
    Parameters
    ----------
    r0 : float
        Starting radius (≫ b so weak-field at entry).
    b  : float
        Impact parameter.
    rs : float
        Schwarzschild radius (2GM/c² in geometric units).
    nsteps : int, optional
        Integration steps in φ (default 8 000).
    dl : float, optional
        Step size Δφ (named *dl* because tests pass dl=…).

    Returns
    -------
    r_hist : ndarray
        Radial history.
    phi_final : float
        Final azimuth in lab frame (initial point at φ = π/2).
    path : tuple(ndarray, ndarray)
        (r_hist, φ_hist) in lab frame.
    """
    dphi = dl
    u0 = 1.0 / r0

    # exact Schwarzschild slope from the first integral
    w0_sq = 1.0 / b**2 - u0**2 + rs * u0**3
    if w0_sq <= 0.0:
        raise ValueError("r0 must satisfy  1/b² > u0² − r_s u0³")
    w0 = -np.sqrt(w0_sq)        # negative ⇒ inward

    y = np.array([u0, w0], dtype=np.float64)
    u_hist   = np.empty(nsteps, dtype=np.float64)
    phi_hist = np.empty_like(u_hist)

    turned = False
    phi = 0.0

    for i in range(nsteps):
        u_hist[i]   = 1.0 / y[0]          # store r = 1/u
        phi_hist[i] = phi

        # one RK4 step in φ
        k1 = _rhs(y, rs)
        k2 = _rhs(y + 0.5 * dphi * k1, rs)
        k3 = _rhs(y + 0.5 * dphi * k2, rs)
        k4 = _rhs(y + dphi * k3, rs)
        y += (dphi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi += dphi

        # stop once photon has turned around and re-reaches r ≥ r0
        if not turned and y[1] > 0.0:      # periapsis
            turned = True
        elif turned and y[0] <= u0:        # back out to r ≥ r0
            break

    # truncate histories to the number of filled entries
    r_hist  = u_hist[: i + 1]

    # convert all angles to lab frame (start at φ = π/2)
    phi_lab = phi_hist[: i + 1] + np.pi / 2.0
    phi_final_lab = phi_lab[-1]

    return r_hist, phi_final_lab, (r_hist, phi_lab)
