"""
Gauge-anomaly checker for the Standard Model.
SM hypercharges are in units where Y = Q − T3.
"""

from typing import List, Tuple

# (name, electric charge Q, SU(2) doublet size)
_FERMIONS: List[Tuple[str, float, int]] = [
    # 3 colours of quarks × 3 families
    ("u_L",  +2/3, 2), ("d_L", -1/3, 2),
    ("u_R",  +2/3, 1), ("d_R", -1/3, 1),
    # leptons
    ("e_L",  -1. , 2), ("ν_L",  0. , 2),
    ("e_R",  -1. , 1),
]

def anomaly_sums() -> Tuple[float, float]:
    """
    Return (ΣQ, ΣQ³) over all SM fermions including colours.
    Both must cancel (=0) for gauge consistency.
    """
    sum_Q   = 0.0
    sum_Q3  = 0.0
    nc = 3  # QCD colours
    for name, Q, d2 in _FERMIONS:
        factor = nc if "u_" in name or "d_" in name else 1
        sum_Q  += factor * d2 * Q
        sum_Q3 += factor * d2 * Q**3
    return sum_Q, sum_Q3
