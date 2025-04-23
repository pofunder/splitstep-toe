import numpy as np
from splitstep_toe.darkmatter import demo_two_body

def test_two_body_energy_stable():
    sep = demo_two_body(n_steps=400, dt=2e-3)
    # In a stable near-circular orbit separation should stay ~1 (±15 %)
    assert sep.std() / sep.mean() < 0.15
