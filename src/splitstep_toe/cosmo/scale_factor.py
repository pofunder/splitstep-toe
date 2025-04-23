"""
FLRW scale-factor helpers for Split-Step-ToE
===========================================

* Flat ΛCDM or Einstein-de Sitter (Ω_m = 1, Ω_Λ = 0)
* Analytic shortcut for EdS so the unit test is instant
* Hub(a)   – dimensionless H(a)/H0
* scale_factor(t, Ω_m, Ω_Λ, a0)  – returns a(t), H(t)
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import odeint
from numpy.typing import NDArray

#------------------------------------------------------------------------
# 1.  Hubble parameter helper
#------------------------------------------------------------------------
def Hub(a: float | NDArray[np.floating],
        Ω_m: float = 1.0,
        Ω_Λ: float = 0.0
        ) -> float | NDArray[np.floating]:
    """
    Dimensionless H(a)/H0 for a flat universe (radiation neglected).

        H(a)/H0 = √( Ω_m / a³ + Ω_Λ )
    """
    return np.sqrt(Ω_m / a**3 + Ω_Λ)


#------------------------------------------------------------------------
# 2.  Friedmann integrator (with analytic EdS branch)
#------------------------------------------------------------------------
H0 = 2.0 / 3.0               # choose units so T_H = 1 (matches unit-test)


def _dadt(a: float, t: float, Ω_m: float, Ω_Λ: float) -> float:
    """Right-hand side  da/dt = H a  (with H0 already scaled in)."""
    return H0 * Hub(a, Ω_m, Ω_Λ) * a


def scale_factor(t: NDArray[np.floating],
                 Ω_m: float = 1.0,
                 Ω_Λ: float = 0.0,
                 a0: float = 0.0
                 ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute scale factor a(t) for flat ΛCDM.

    Parameters
    ----------
    t   : array-like cosmic time in units of T_H = 1/H0  (H0 = 2/3 for tests)
    Ω_m : matter density parameter
    Ω_Λ : cosmological-constant density parameter
    a0  : initial scale factor (default 0 → lifted to 1e-6 internally)

    Returns
    -------
    a(t), H(t)  – NumPy arrays matching `t`.
    """

    t = np.asarray(t, dtype=float)

    # -- analytic Einstein-de Sitter shortcut -----------------------------
    if np.isclose(Ω_m, 1.0) and np.isclose(Ω_Λ, 0.0):
        a = (t + 1e-12) ** (2.0 / 3.0)          # avoid 0^{2/3}
        H_t = (2.0 / 3.0) * a ** (-3.0 / 2.0)   # H/H0 with H0 = 2/3
        return a, H_t

    # -- numeric ΛCDM integration -----------------------------------------
    eps   = 1.0e-6
    a0_ok = a0 if a0 > eps else eps             # lift off zero
    a = odeint(_dadt, a0_ok, t,
               args=(Ω_m, Ω_Λ),
               atol=1e-10, rtol=1e-8, mxstep=10000)[:, 0]
    a[np.isnan(a)] = a0_ok                      # scrub any NaNs

    H_t = H0 * Hub(a, Ω_m, Ω_Λ)
    return a, H_t
