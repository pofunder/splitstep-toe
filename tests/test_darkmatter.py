import numpy as np
from splitstep_toe.darkmatter import demo_two_body

def test_two_body_energy_stable():
    sep = demo_two_body()                    # use the function defaults
    # In a stable near-circular orbit separation should stay ~1 (±15 %)
    # near-circular orbit → <5 % spread
    assert sep.std() / sep.mean() < 0.05
