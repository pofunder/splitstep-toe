from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

# ---------------------------------------------------------------------
# constants (Mpc, km/s, Gyr units not needed for dimensionless a & H/H0)
# ---------------------------------------------------------------------
H0 = 1.0            # work in units where H0 = 1
T_H = 1.0 / H0      # Hubble time unit

# ---------------------------------------------------------------------
def Hub(a: float, Ω_m: float, Ω_Λ: float) -> float:
    """Dimensionless H(a)/H0 for flat FLRW (radiation neglected)."""
    return np.sqrt(Ω_m / a**3 + Ω_Λ)


def _rhs(t: float, y: float, Ω_m: float, Ω_Λ: float) -> float:
    """ da/dt  (dimensionless; t in Hubble-time units). """
    return Hub(y, Ω_m, Ω_Λ) * y


def scale_factor(
    t: np.ndarray,
    Ω_m: float = 1.0,
    Ω_Λ: float = 0.0,
    a0: float = 1.0,
) -> Tuple[np.ndarray, Callable[[float], float]]:
    """
    Integrate the Friedmann equation for a(t) on a given time grid.

    Parameters
    ----------
    t   : array of times (Hubble-units, can be negative for look-back)
    Ω_m : matter density parameter today
    Ω_Λ : dark-energy density parameter today
    a0  : scale factor at t = 0  (default 1)

    Returns
    -------
    a   : array, same shape as t
    f   : callable a(t) interpolator
    """
    dt = np.diff(t)
    a  = np.empty_like(t)
    a[0] = a0

    for i, h in enumerate(dt, 0):
        # 4th-order RK
        k1 = _rhs(t[i],     a[i],             Ω_m, Ω_Λ)
        k2 = _rhs(t[i]+h/2, a[i]+h*k1/2,      Ω_m, Ω_Λ)
        k3 = _rhs(t[i]+h/2, a[i]+h*k2/2,      Ω_m, Ω_Λ)
        k4 = _rhs(t[i]+h,   a[i]+h*k3,        Ω_m, Ω_Λ)
        a[i+1] = a[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return a, lambda τ: np.interp(τ, t, a)
