import numba as nb
import numpy as np

@nb.njit
def laplacian_2d(field, h):
    """
    Five‑point stencil Laplacian on a 2‑D array.
    Dirichlet boundary (edges set to zero).
    """
    ny, nx = field.shape
    out = np.empty_like(field)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            out[j, i] = (
                field[j+1, i] + field[j-1, i] +
                field[j, i+1] + field[j, i-1] -
                4.0 * field[j, i]
            ) / (h * h)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0.0
    return out
