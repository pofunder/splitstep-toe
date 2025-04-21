
import numba as nb
import numpy as np

@nb.njit
def step(R_prev, R_curr, kappa, lam, gamma, laplacian, G):
    """One recursion step given prev & curr arrays (in-place)."""
    # R_next = R_curr + kappa*laplacian(R_curr)+lam*G(R_curr)+gamma*(R_curr-R_prev)
    R_next = np.empty_like(R_curr)
    for i in range(R_curr.size):
        R_next[i] = (R_curr[i] + kappa*laplacian(R_curr, i)
                     + lam*G(R_curr[i]) + gamma*(R_curr[i]-R_prev[i]))
    return R_next
