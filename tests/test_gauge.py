from splitstep_toe.gauge import anomaly_sums

def test_anomaly_cancellation():
    sum_Q, sum_Q3 = anomaly_sums()
    assert abs(sum_Q)  < 1e-12
    assert abs(sum_Q3) < 1e-12
