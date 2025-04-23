import numpy as np
from splitstep_toe.cosmo import scale_factor, Hub

def test_EdS_scale():
    # Einstein-de Sitter => analytic a(t) = (t / T_H)^{2/3}
    t = np.linspace(0.0, 1.0, 50)
    a_num, _ = scale_factor(t, Ω_m=1.0, Ω_Λ=0.0, a0=0.0)
    a_ana = (t + 1e-12) ** (2 / 3)           # avoid 0^{2/3}
    assert np.allclose(a_num[5:], a_ana[5:], rtol=5e-2)

def test_LCDM_H():
    # Flat LCDM Ω_m=0.3, Ω_Λ=0.7  => H(a=1)/H0 = 1
    assert abs(Hub(1.0, 0.3, 0.7) - 1.0) < 1e-12
