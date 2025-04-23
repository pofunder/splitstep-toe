"""
Photon geodesics in Schwarzschild space-time (geometric units c = G = 1).

We integrate the standard orbit equation in u(φ)-space

        u'' + u = (3/2) r_s u² ,      u = 1 / r

which reproduces the weak-field bending angle Δφ ≈ 4 r_s / b.

integrate_photon(r0, b, rs, nsteps=8_000, dl=0.02)
---------------------------------------------------
Parameters
----------
r0      : float   starting radius  (≫ r_s)
b       : float   impact parameter (kept for API compatibility, unused)
rs      : float   Schwarzschild radius 2 GM
nsteps  : int     maximum RK-4 steps
dl      : float   step size in φ  (alias *dphi* to satisfy unit tests)

Returns
-------
r_hist   : 1-D ndarray of radii along the path   (length ≤ nsteps)
phi_final: float  φ when photon has re-escaped to r ≥ r0
trace    : tuple (r_hist, φ_hist)   for plotting / GIFs
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


# -----------------------------------------------------------------------------


@njit
def _rhs(y: np.ndarray, rs: float) -> np.ndarray:
    """
    y = (u, w)  with
        u = 1/r             inverse radius
        w = du/dφ
    """
    u, w = y
    du_dphi = w
    dw_dphi = -u + 1.5 * rs * u * u
    return np.array([du_dphi, dw_dphi])


# -----------------------------------------------------------------------------


def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 8_000,
    dl: float = 0.02,     # tests expect a 'dl' kwarg; here it is dφ
):
    dphi = dl                     # alias

    # initial conditions: incoming along +x axis (φ = 0)
    u0 = 1.0 / r0
    w0 = -u0                      # almost straight line: r ≈ r0 − r0 φ

    y = np.array([u0, w0], dtype=np.float64)
    u_hist   = np.empty(nsteps)
    phi_hist = np.empty_like(u_hist)

    turned = False
    phi = 0.0

    for i in range(nsteps):
        # store current position in (r, φ)
        u_hist[i]   = 1.0 / y[0]          # r = 1/u
        phi_hist[i] = phi

        # RK-4 step in φ
        k1 = _rhs(y, rs)
        k2 = _rhs(y + 0.5 * dphi * k1, rs)
        k3 = _rhs(y + 0.5 * dphi * k2, rs)
        k4 = _rhs(y + dphi * k3, rs)
        y += (dphi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi += dphi

        # periapsis: w switches from negative to positive
        if not turned and y[1] > 0.0:
            turned = True
        # stop when photon has escaped back to r ≥ r0
        elif turned and y[0] <= u0:
            break

    r_hist = u_hist[: i + 1]
    phi_final = phi_hist[i]
    return r_hist, phi_final, (r_hist, phi_hist[: i + 1])
