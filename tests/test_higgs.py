from splitstep_toe.ew_breaking import lam_mu2_from_mh, higgs_mass

def test_higgs_tree_level():
    lam, mu2 = lam_mu2_from_mh()
    m_h = higgs_mass(lam)
    assert abs(m_h - 125.0) < 0.5          # within ±0.5 GeV
    assert mu2 < 0                         # symmetry breaking needs μ²<0
