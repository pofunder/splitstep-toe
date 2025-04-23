"""
SM gauge-anomaly bookkeeping.

We work with **left-handed Weyl fields**.  Each entry is
(multiplicity = SU(2) × colour, hypercharge Y).

For N_gen = 3 the sums ΣY and ΣY³ both vanish ⇒ anomaly-free.
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────── constants
_N_GEN = 3  # number of fermion generations

# (multiplicity, hypercharge)
_FIELDS: list[tuple[int, float]] = [
    (2 * 3, +1 / 6),  # q_L   (doublet, colour)
    (3, +2 / 3),  # u_R
    (3, -1 / 3),  # d_R
    (2, -1 / 2),  # l_L
    (1, -1.0),  # e_R
]

# ─────────────────────────────────────────────────────────── public functions
def anomaly_sums(n_gen: int = _N_GEN) -> tuple[float, float]:
    """
    Return the hyper-charge gauge-anomaly sums (ΣY, ΣY³).

    A vanishing pair (≈ 0) means the gauge group is anomaly-free.
    """
    mult, Y = np.array([m for m, _ in _FIELDS]), np.array([y for _, y in _FIELDS])
    sum_Y = n_gen * (mult * Y).sum()
    sum_Y3 = n_gen * (mult * Y**3).sum()
    # strip tiny FP noise
    return float(sum_Y), float(sum_Y3)


def anomaly_cancelled() -> bool:  # convenience one-liner used by tests
    s1, s3 = anomaly_sums()
    return abs(s1) < 1e-12 and abs(s3) < 1e-12
