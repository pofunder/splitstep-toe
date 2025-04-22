from splitstep_toe.geometry import schwarzschild_f

def test_metric_lapse():
    """sanity check f(r) = 1 − rs/r"""
    assert abs(schwarzschild_f(10.0, 2.0) - 0.8) < 1e-12
