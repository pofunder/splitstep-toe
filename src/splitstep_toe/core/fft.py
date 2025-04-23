"""
Fast-Fourier-Transform utilities.

laplacian_fft(f, h)
    Return ∇²f for a 3-D scalar field *f* sampled on a cubic grid with spacing *h*.

The FFT Laplacian is exact (up to FP error) and runs in O(N log N),
while the finite-difference stencil is only second-order and O(N).
"""
from __future__ import annotations

import numpy as np
from numpy.fft import rfftn, irfftn

__all__ = ["laplacian_fft"]

def laplacian_fft(f: np.ndarray, h: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    f : ndarray (nx, ny, nz)
        Real-valued scalar field.
    h : float, optional
        Grid spacing (assumed equal in all three directions).

    Returns
    -------
    lap : ndarray, same shape as *f*
        Laplacian ∇²f computed via spectral differentiation.
    """
    if f.ndim != 3:
        raise ValueError("laplacian_fft expects a 3-D array")

    nx, ny, nz = f.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=h)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=h)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=h)     # nz → positive freq half

    # Broadcast wave-vector squares on r2c grid
    k2 = (
        kx[:, None, None] ** 2
        + ky[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )

    f_hat = rfftn(f, workers=-1)
    lap_hat = -k2 * f_hat
    return irfftn(lap_hat, s=f.shape, workers=-1).real
