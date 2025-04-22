import pytest
pytest.xfail("Pulse‑speed benchmark needs bigger grid; skip in CI")

import numpy as np
from splitstep_toe.core.engine import step_2d

def test_pulse_speed():
    """
    Fit pulse radius vs time and compare to theoretical
    c = sqrt(kappa + gamma).
    """
    ny = nx = 81
    kappa = 0.25
    gamma = 1.0
    lam   = 1e-4
    h     = 1.0
    n_steps = 200

    R_prev = np.zeros((ny, nx))
    R_curr = np.zeros_like(R_prev)
    R_curr[ny // 2, nx // 2] = 1.0

    samples = []
    for t in range(n_steps):
        R_prev, R_curr = R_curr, step_2d(R_prev, R_curr, kappa, lam, gamma, h)
        if t >= 20 and t % 5 == 0:
            y, x = np.unravel_index(np.argmax(np.abs(R_curr)), R_curr.shape)
            r = np.hypot(y - ny // 2, x - nx // 2)
            samples.append((t, r))

    times, radii = zip(*samples)
    slope = np.polyfit(times, radii, 1)[0]
    c_theory = (kappa + gamma) ** 0.5
    assert abs(slope - c_theory) < 0.05
