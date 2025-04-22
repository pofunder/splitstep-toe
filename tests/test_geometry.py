from splitstep_toe.geometry.metric import schwarzschild_metric
def test_metric():
    assert schwarzschild_metric(10,2)[0][0] < 0
