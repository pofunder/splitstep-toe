def scale_factor(t, Ω_m=1.0, Ω_Λ=0.0, a0: float = 0.0):
    """
    Flat ΛCDM scale factor solver (units T_H = 1).

    • If Ω_m == 1 and Ω_Λ == 0  → return analytic EdS  a(t) = (t)^{2/3}.
    • Otherwise integrate da/dt = H a  with a safe, positive a0≥ε.
    """
    t = np.asarray(t, dtype=float)

    # ---- analytic Einstein–de Sitter branch ------------------------------
    if np.isclose(Ω_m, 1.0) and np.isclose(Ω_Λ, 0.0):
        a = (t + 1e-12) ** (2.0 / 3.0)
        H_t = (2.0 / 3.0) * a ** (-3.0 / 2.0)        # H/H0
        return a, H_t

    # ---- numeric ΛCDM branch ---------------------------------------------
    eps = 1e-6
    a0_safe = a0 if a0 > eps else eps               # lift off zero
    a = odeint(_dadt, a0_safe, t, args=(Ω_m, Ω_Λ),
               atol=1e-10, rtol=1e-8)[:, 0]
    a[np.isnan(a)] = a0_safe                        # scrub any NaN
    H_t = Hub(a, Ω_m, Ω_Λ) * (2.0/3.0)              # restore H0 factor
    return a, H_t
