"""
Standard-Model gauge-anomaly bookkeeping.

We list the SM fermions (one generation) with their gauge
quantum numbers and compute
    Σ Q³  (hyper-charge cubic anomaly)
    Σ Q   (mixed U(1)–gravitational anomaly)
Both sums must vanish generation-by-generation → anomaly cancellation.
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Fermion:
    name: str
    Y: float   # weak hyper-charge
    Nc: int    # colour multiplicity
    chirality: str = "L"   # 'L' or 'R'


# --- one generation ---------------------------------------------------------

_fermions: List[Fermion] = [
    # quarks
    Fermion("q_L",  +1/6, 3, "L"),
    Fermion("u_R",  +2/3, 3, "R"),
    Fermion("d_R",  -1/3, 3, "R"),
    # leptons
    Fermion("l_L",  -1/2, 1, "L"),
    Fermion("e_R",  -1  , 1, "R"),
]

def anomaly_sums() -> Tuple[float, float]:
    """Return (Σ Y³, Σ Y) for one SM generation."""
    sum_Y3 = sum(f.Nc * f.Y**3 for f in _fermions)
    sum_Y  = sum(f.Nc * f.Y    for f in _fermions)
    return sum_Y3, sum_Y


# ---- convenience flag used by the tests ------------------------------------

sum_Y3, sum_Y = anomaly_sums()
anomaly_cancelled: bool = abs(sum_Y3) < 1e-12 and abs(sum_Y) < 1e-12
