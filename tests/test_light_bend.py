import math
from splitstep_toe.geometry import integrate_photon


def test_light_bending():
    """
    Photon deflection angle should match weak‑field analytic
    Δφ ≈ 4 rs / b  (accept ±20 %)
    """
    rs, b, r0 = 2.0, 50.0, 100.0
    _, phi_final, _ = integrate_photon(r0, b, rs, nsteps=2_000, dl=0.1)

    delta_phi_num = math.pi - 2.0 * phi_final
    delta_phi_anal = 4.0 * rs / b
    assert abs(delta_phi_num / delta_phi_anal - 1.0) < 0.20
