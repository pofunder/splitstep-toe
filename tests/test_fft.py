import numpy as np
from splitstep_toe.core.fft import laplacian_fft
from splitstep_toe.core.laplacian import laplacian_2d  # existing FD stencil

def finite_difference_3d(f, h=1.0):
    """Second-order 3-D finite-difference Laplacian, periodic BC."""
    lap = np.zeros_like(f)
    for axis in range(3):
        f_forward = np.roll(f, -1, axis)
        f_back    = np.roll(f,  1, axis)
        lap += (f_forward - 2 * f + f_back) / h**2
    return lap

def test_fft_laplacian_accuracy():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((32, 32, 32))
    lap_fd  = finite_difference_3d(f)
    lap_fft = laplacian_fft(f)
    rel_err = np.linalg.norm(lap_fft - lap_fd) / np.linalg.norm(lap_fd)
    assert rel_err < 1e-6
