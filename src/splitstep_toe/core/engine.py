
import numba as nb
from .laplacian import laplacian_2d

@nb.njit
def _nonlin(x):
    return x**3          # cubic self‑interaction

@nb.njit
def step_2d(R_prev, R_curr, kappa, lam, gamma, h):
    """
    One recursion step in 2‑D:
        Rⁿ⁺¹ = Rⁿ + κ∇²Rⁿ − λ Rⁿ³ + γ(Rⁿ − Rⁿ⁻¹)
    """
    L = laplacian_2d(R_curr, h)
    return R_curr + kappa*L - lam*_nonlin(R_curr) + gamma*(R_curr - R_prev)
