"""
Photon geodesics in Schwarzschild space-time (c = G = 1).

integrate_photon(r0, b, rs, nsteps=8_000, dl=0.02)
    → r_hist, phi_final, (r_hist, φ_hist)
"""

from __future__ import annotations
import numpy as np
from numba import njit

__all__ = ["integrate_photon"]


@njit
def _rhs(y: np.ndarray, rs: float) -> np.ndarray:
    u, w = y                       # u = 1/r ,  w = du/dφ
    du_dphi = w
    dw_dphi = -u + 1.5 * rs * u * u
    return np.array([du_dphi, dw_dphi])


# ──────────────────────────────────────────────────────────────────────────────
def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    nsteps: int = 8_000,
    dl: float = 0.02,        # tests expect keyword ‘dl’
):
    dphi = dl                # alias

    # initial u and slope from the invariant  (u')² + u² = 1/b²
    u0 = 1.0 / r0
    w0_sq = 1.0 / b**2 - u0**2
    if w0_sq <= 0.0:
        raise ValueError("start radius must satisfy r0 > b")
    w0 = -np.sqrt(w0_sq)     # negative ⇒ moving inward

    y = np.array([u0, w0], dtype=np.float64)
    u_hist   = np.empty(nsteps)
    phi_hist = np.empty_like(u_hist)

    turned = False
    phi = 0.0

    for i in range(nsteps):
        u_hist[i]   = 1.0 / y[0]          # store r = 1/u
        phi_hist[i] = phi

        # RK-4 step
        k1 = _rhs(y, rs)
        k2 = _rhs(y + 0.5 * dphi * k1, rs)
        k3 = _rhs(y + 0.5 * dphi * k2, rs)
        k4 = _rhs(y + dphi * k3, rs)
        y += (dphi / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        phi += dphi

        if not turned and y[1] > 0.0:          # periapsis reached
            turned = True
        elif turned and y[0] <= u0:            # back to r ≥ r0
            break

    r_hist = u_hist[: i + 1]
    phi_final = phi_hist[i]
    return r_hist, phi_final, (r_hist, phi_hist[: i + 1])
