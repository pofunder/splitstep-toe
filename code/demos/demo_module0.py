"""
Demo Module 0: Persistence evolution and gravity seed.
"""

import numpy as np
import matplotlib.pyplot as plt
from code.persistence import split_step_persistence

def main():
    N, R0 = 64, 16
    kappa, gamma = 0.1, 0.2
    eta, beta_nl = 1e-5, 1e-3

    A      = np.zeros((N,N,N), dtype=np.float32)
    A_prev = np.zeros_like(A)
    # seed shell
    coords = np.arange(N) - N//2
    X,Y,Z  = np.meshgrid(coords,coords,coords, indexing='ij')
    r      = np.sqrt(X**2+Y**2+Z**2)
    A[np.abs(r-R0)<0.5] = 1.0

    # collect center‐slice at steps 0 and 10
    slices = [A[:,:,N//2].copy()]
    for i in range(10):
        A_next = split_step_persistence(A, A_prev, kappa, gamma, eta, beta_nl, R0)
        A_prev, A = A, A_next
    slices.append(A[:,:,N//2].copy())

    # plot
    fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8,4))
    ax0.imshow(slices[0], cmap='inferno'); ax0.set_title('Step 0')
    ax1.imshow(slices[1], cmap='inferno'); ax1.set_title('Step 10')
    plt.tight_layout()
    plt.savefig('figures/module0_steps_0_10.png', dpi=150)
    plt.show()

if __name__=='__main__':
    main()

