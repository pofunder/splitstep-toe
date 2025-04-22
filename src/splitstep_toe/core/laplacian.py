import numba as nb
import numpy as np

@nb.njit
def laplacian_2d(field, h):
    """
 