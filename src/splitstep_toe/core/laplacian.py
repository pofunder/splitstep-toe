"""
Fast 5‑point Laplacian on a 2‑D periodic grid.
"""
import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def laplacian_2d(a: np.ndarray, h: float = 1.0) -> np.ndarray:
    ny, nx = a.shape
    out = np.empty_like(a)
    for j in nb.prange(ny):
        jm, jp = (j - 1) % ny, (j + 1) % ny
        for i in range(nx):
            im, ip = (i - 1) % nx, (i + 1) % nx
            out[j, i] = (a[jm, i] + a[jp, i] + a[j, im] + a[j, ip] - 4 * a[j, i]) / h**2
    return out
