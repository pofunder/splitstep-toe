# --- src/splitstep_toe/cosmo/scale_factor.py ­--------------------------------
import numpy as np
from scipy.integrate import odeint

H0 = 2.0 / 3.0                         # keep the correct normalisation

def Hub(a, Ω_m=1.0, Ω_Λ=0.0):
    return np.sqrt(Ω_m / a**3 + Ω_Λ)

def _dadt(a, t, Ω_m, Ω_Λ):
    return H0 * Hub(a, Ω_m, Ω_Λ) * a

def scale_factor(t, Ω_m=1.0, Ω_Λ=0.0, a0=1e-4):
    """
    Flat ΛCDM scale factor (units T_H = 1).  
    Parameters
    ----------
    a0 : float
        *Starting* value of the integration; choose something small but **not
        zero** (default 1e-4).  Avoids the Hub(a→0) singularity.
    """
    t   = np.asarray(t, dtype=float)
    a   = odeint(_dadt, a0, t, args=(Ω_m, Ω_Λ), atol=1e-12, rtol=1e-9)[:, 0]

    # Replace any accidental NaNs from the integrator’s first step
    a[np.isnan(a)] = a0
    return a, H0 * Hub(a, Ω_m, Ω_Λ)
