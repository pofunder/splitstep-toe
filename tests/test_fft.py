# tests/test_fft.py
"""
Accuracy test for the spectral 3-D Laplacian.

We compare the FFT-based result to the *analytic* Laplacian of a smooth,
periodic trigonometric field.  Because the field is infinitely
differentiable and exactly band-limited on the grid, the FFT Laplacian
should match to machine precision (‖error‖ / ‖true‖ < 1e-12).
"""
from __future__ import annotations
import numpy as np
from splitstep_toe.core.fft import laplacian_fft


def test_fft_laplacian_against_analytic() -> None:
    # --- build a smooth 3-D function with known Laplacian ------------------
    nx = ny = nz = 32
    L = 2 * np.pi               # domain length so that k is integer
    h = L / nx                  # grid spacing (same in all directions)

    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)
    z = np.linspace(0, L, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    k1, k2, k3 = 2, 3, 5
    f = np.sin(k1 * X) * np.sin(k2 * Y) * np.sin(k3 * Z)

    lap_true = -(k1**2 + k2**2 + k3**2) * f      # analytic ∇²f
    lap_fft  = laplacian_fft(f, h=h)

    rel_err = np.linalg.norm(lap_fft - lap_true) / np.linalg.norm(lap_true)
    assert rel_err < 1e-12
