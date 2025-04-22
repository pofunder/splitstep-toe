
import numba as nb
import numpy as np

@nb.njit
def rhs(y, rs):
    # y = [r, phi, pr]
    r, phi, pr = y
    f = 1.0 - rs / r
    dphi = 1.0 / r**2          # L=1
    dr   = pr
    dpr  = -0.5*rs/(r*r) * (1.0/r**2) + (rs)/(2*r**2) + (1.0/r**3)
    return np.array([dr, dphi, dpr])

@nb.njit
def integrate_photon(r0, b, rs, nsteps, dl):
    # initial conditions far from mass at y axis
    r = r0
    phi = 0.0
    pr = -np.sqrt(1.0 - (b/r)**2)
    y = np.array([r, phi, pr])
    for _ in range(nsteps):
        k1 = rhs(y, rs)
        k2 = rhs(y + 0.5*dl*k1, rs)
        k3 = rhs(y + 0.5*dl*k2, rs)
        k4 = rhs(y + dl*k3, rs)
        y += (dl/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y
