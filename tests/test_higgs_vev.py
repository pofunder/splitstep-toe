from splitstep_toe.ew_breaking import higgs_vev

def test_higgs_value():
    v = higgs_vev()
    assert 245.0 < v < 247.0        # ±1 GeV tolerance
