import math
from splitstep_toe.geometry.geodesic import integrate_photon

def test_light_bending():
    rs, b, r0 = 2.0, 50.0, 100.0
    dl, nsteps = 0.05, 8000

    _, phi_f, _ = integrate_photon(r0, b, rs, nsteps, dl)
    delta_phi_num = math.pi - 2 * phi_f
    delta_phi_analytic = 4 * rs / b  # weak‑field deflection

    # Accept within 20 %
    assert abs(delta_phi_num / delta_phi_analytic - 1) < 0.2
