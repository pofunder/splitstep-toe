"""
Gauge-anomaly bookkeeping for a single Standard-Model generation.

The SM is free of all chiral gauge anomalies provided that both
  Σ Y   = 0   and   Σ Y³ = 0
when every left-handed Weyl fermion state is counted with its full
multiplicity:  colour factor × SU(2) dimension.

Notation
--------
multiplicity =  (# colours) · (SU(2) dimension)
Y  = U(1)_Y hyper-charge  (Q = T³ + Y)

References
----------
* Peskin & Schroeder §20
* arXiv:hep-ph/9507286
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-charge table  (one generation, left-handed chiral fields only)

_SM_FIELD_LIST: list[tuple[int, float]] = [
    (3 * 2, +1 / 6),   # q_L  : (u_L , d_L)   multiplicity 3 colours × 2 iso-comp.
    (3,     +2 / 3),   # u_R  : up-type singlet
    (3,     -1 / 3),   # d_R  : down-type singlet
    (1 * 2, -1 / 2),   # ℓ_L  : (ν_L , e_L)
    (1,     -1),       # e_R  : charged-lepton singlet
]


# ──────────────────────────────────────────────────────────────────────────────
def anomaly_sums(n_gen: int = 3) -> tuple[float, float]:
    """
    Return Σ Y and Σ Y³ (including colour & SU(2) multiplicities) for *n_gen* families.
    They must both vanish in a consistent gauge theory.
    """
    sum_Y = 0.0
    sum_Y3 = 0.0
    for mult, Y in _SM_FIELD_LIST:
        sum_Y += n_gen * mult * Y
        sum_Y3 += n_gen * mult * Y**3
    return sum_Y, sum_Y3


def anomaly_cancelled(n_gen: int = 3) -> bool:
    """Convenience boolean with a tight numerical tolerance."""
    sY, sY3 = anomaly_sums(n_gen)
    return abs(sY) < 1e-12 and abs(sY3) < 1e-12
