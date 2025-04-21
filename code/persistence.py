"""
Module 0 & 1: Persistence field update (gravity & dark matter)
"""

import numpy as np
from numpy.fft import fftn, ifftn

def split_step_persistence(A, A_prev, kappa, gamma, eta, beta_nl, R0):
    # 1. Memory
    M = A + gamma * (A - A_prev)

    # 2. Local diffusion (6 nearest neighbors)
    M_pad = (
        np.roll(M,  1, axis=0) + np.roll(M, -1, axis=0) +
        np.roll(M,  1, axis=1) + np.roll(M, -1, axis=1) +
        np.roll(M,  1, axis=2) + np.roll(M, -1, axis=2)
    )
    L = (M + kappa * M_pad) / (1 + 6 * kappa)

    # 3. Non‑local echo via thin shell convolution
    N = A.shape[0]
    coords = np.arange(N) - N//2
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    shell = (np.abs(r - R0) < 0.5).astype(np.float32)

    shell_k = fftn(shell,    s=A.shape)
    L_k     = fftn(L,        s=A.shape)
    E       = eta * np.real(ifftn(shell_k * L_k))

    # 4. Self‑damping & combine
    A_next = L + E - beta_nl * (L**3)
    return A_next
