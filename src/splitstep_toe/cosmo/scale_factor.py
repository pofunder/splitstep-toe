# --- src/splitstep_toe/cosmo/scale_factor.py -------------------------------
import numpy as np
from scipy.integrate import odeint

# use the test-suite’s convention:  H0 = 2/3  (so T_H = 1)
H0 = 2.0/3.0                        # ← was 1.0

def Hub(a, Ω_m=1.0, Ω_Λ=0.0):
    """Dimensionless H(a)/H0."""
    return np.sqrt(Ω_m/a**3 + Ω_Λ)

def _dadt(a, t, Ω_m, Ω_Λ):
    return  H0 * Hub(a, Ω_m, Ω_Λ) * a      # da/dt

def scale_factor(t, Ω_m=1.0, Ω_Λ=0.0, a0=1e-12):
    """
    Return a(t) and H(t) for flat ΛCDM with the code’s units (H0 = 2/3).
    """
    t   = np.asarray(t, dtype=float)
    a   = odeint(_dadt, a0, t, args=(Ω_m, Ω_Λ), rtol=1e-9, atol=1e-12)[:,0]
    H_t = H0 * Hub(a, Ω_m, Ω_Λ)
    return a, H_t
