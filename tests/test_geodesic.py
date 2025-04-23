import math, numpy as np
from splitstep_toe.geometry.geodesic import integrate_photon

def test_photon_deflection():
    """
    For b ≫ r_s the bending angle should approach Δφ = 4 rs / b.
    Accept ±2 %.
    """
    rs, b, r0 = 2.0, 40.0, 200.0
    _, phi_fin, _ = integrate_photon(r0, b, rs, nsteps=6000, dl=0.02)

    delta_phi_num = math.pi - 2 * phi_fin
    delta_phi_th  = 4 * rs / b
    rel_err = abs(delta_phi_num / delta_phi_th - 1.0)
    assert rel_err < 0.02
