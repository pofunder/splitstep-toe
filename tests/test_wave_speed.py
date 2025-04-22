import numpy as np
from splitstep_toe.core.engine import step_2d

def test_pulse_speed():
    ny = nx = 61
    kappa = 0.25
    lam = 1e-4
    gamma = 1.0
    h = 1.0

    R_prev = np.zeros((ny, nx))
    R_curr = np.zeros_like(R_prev)
    R_curr[ny//2, nx//2] = 1.0

    for _ in range(120):
        R_prev, R_curr = R_curr, step_2d(R_prev, R_curr, kappa, lam, gamma, h)

    y, x = np.unravel_index(np.argmax(np.abs(R_curr)), R_curr.shape)
    r = np.hypot(y - ny//2, x - nx//2)
    c_measured = r / 120
    assert abs(c_measured - kappa**0.5) < 0.05
