"""
Very light-weight Friedmann-equation integrator.

The only external dependency is SciPy (solve_ivp).  In CI we rely on the
`scipy` wheel already present in the test image; locally you can

    pip install scipy
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
#  Hubble parameter H(a)  (H0 has been scaled out of the unit tests)          #
# --------------------------------------------------------------------------- #
def Hub(a: float | NDArray[np.floating],
        Ω_m: float,
        Ω_Λ: float,
        H0: float = 1.0) -> float | NDArray[np.floating]:
    r"""Hubble parameter

        H(a) = H₀ √(Ω_m / a³ + Ω_Λ)

    Parameters
    ----------
    a   : scale factor(s)
    Ω_m : matter density parameter
    Ω_Λ : cosmological-constant density parameter
    H0  : present-day Hubble constant (set to 1 in unit tests)
    """
    return H0 * np.sqrt(Ω_m / a**3 + Ω_Λ)


# --------------------------------------------------------------------------- #
#  Internal: protect against divide-by-zero when a₀ = 0                       #
# --------------------------------------------------------------------------- #
def _a_initial(a0: float) -> float:
    """Return a strictly positive start value for ODE integration."""
    return a0 if a0 > 0.0 else 1.0e-6   # 10⁻⁶ easily avoids 1/a³ blow-up


# --------------------------------------------------------------------------- #
#  Public: scale_factor(t, Ω_m, Ω_Λ, a0, H0)                                  #
# --------------------------------------------------------------------------- #
def scale_factor(t: NDArray[np.floating],
                 Ω_m: float,
                 Ω_Λ: float,
                 a0: float = 0.0,
                 H0: float = 1.0
                 ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Integrate  ȧ = H(a) a  forward in cosmic time.

    Parameters
    ----------
    t   : 1-D array of cosmic times (same units as 1/H0)
    Ω_m : matter density parameter
    Ω_Λ : cosmological-constant density parameter
    a0  : initial scale factor (default 0 ⇒ start at 10⁻⁶)
    H0  : present-day Hubble constant (set to 1 in the unit tests)

    Returns
    -------
    a(t), ȧ(t) – two NumPy arrays of the same length as *t*.
    """
    # --- safe starting value to avoid divide-by-zero ------------------------
    y0 = _a_initial(a0)

    # ODE:  da/dτ = H(a) a   with   τ = t·H0  (so H0 ≡ 1 in code)
    rhs = lambda τ, a: Hub(a, Ω_m, Ω_Λ, H0) * a

    sol = solve_ivp(rhs,
                    (t[0], t[-1]),
                    (y0,),
                    t_eval=t,
                    rtol=1.0e-8,
                    atol=1.0e-10,
                    dense_output=False)

    a = sol.y[0]
    adot = Hub(a, Ω_m, Ω_Λ, H0) * a
    return a, adot
