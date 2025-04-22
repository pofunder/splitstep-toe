
from splitstep_toe.geometry.geodesic import integrate_photon
def test_geodesic_runs():
    r0=100.0; b=50.0; rs=2.0
    y = integrate_photon(r0,b,rs,10,0.1)
    assert y[0] < r0
