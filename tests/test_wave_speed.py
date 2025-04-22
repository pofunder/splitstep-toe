import numpy as np
from splitstep_toe.core.engine import step_2d

def test_pulse_speed():
    """
    Measure pulse speed by linear‑fitting radius vs time.
    Should match sqrt(kappa)=0.5 within 0.05.
    """
    ny = nx = 81          # a bit wider grid
    kappa = 0.25
    lam = 1e-4
    gamma = 0.0
    h = 1.0
    n_steps = 200

    R_prev = np.zeros((ny, nx))
    R_curr = np.zeros_like(R_prev)
    R_curr[ny // 2, nx // 2] = 1.0

    samples = []
    for t in range(n_steps):
        R_prev, R_curr = R_curr, step_2d(R_prev, R_curr, kappa, lam, gamma, h)
        # sample every 5 steps after the first 20 to avoid start‑up transients
        if t >= 20 and t % 5 == 0:
            y, x = np.unravel_index(np.argmax(np.abs(R_curr)), R_curr.shape)
            r = np.hypot(y - ny // 2, x - nx // 2)
            samples.append((t, r))

    times, radii = zip(*samples)
    slope = np.polyfit(times, radii, 1)[0]   # linear fit => speed
    assert abs(slope - kappa**0.5) < 0.05
