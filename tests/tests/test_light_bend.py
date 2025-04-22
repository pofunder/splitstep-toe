from splitstep_toe.geometry.geodesic import integrate_photon
import math

def test_light_bending():
    rs = 2.0          # Schwarzschild radius (lattice units)
    b  = 50.0         # impact parameter
    r0 = 100.0        # start far away
    dl = 0.05
    nsteps = 8000

    _, phi_f, _ = integrate_photon(r0, b, rs, nsteps, dl)
    delta_phi_num = math.pi - 2*phi_f           # numeric
    delta_phi_analytic = 4 * rs / b             # weak‑field formula

    assert abs(delta_phi_num / delta_phi_analytic - 1) < 0.2  # within 20 %
