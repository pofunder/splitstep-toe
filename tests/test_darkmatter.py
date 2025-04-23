import numpy as np
from splitstep_toe.darkmatter import demo_two_body

def test_two_body_energy_stable():
    sep = demo_two_body(n_steps=1_000, dt=2e-3)
    # In a stable near-circular orbit separation should stay ~1 (±15 %)
    assert np.allclose(sep.mean(), 1.0, rtol=0.15)
