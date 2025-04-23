"""
Electroweak symmetry breaking toy formulas.

We work in the SM tree-level potential
    V = μ² φ†φ + λ (φ†φ)²
with convention μ² < 0, λ > 0.

Given the physical Higgs mass m_H and vev v we can solve for
λ and μ².  Units: GeV.
"""

import math
from typing import Tuple

# electroweak vacuum expectation value (G_F → v)
HIGGS_VEV = 246.22  # GeV   (PDG 2024)

def higgs_vev() -> float:
    """Return the Higgs VEV v ≃ 246 GeV."""
    return HIGGS_VEV


def lam_mu2_from_mh(m_h: float, v: float = HIGGS_VEV) -> Tuple[float, float]:
    """
    Solve λ and μ² from the physical mass.

    m_H² = 2 λ v² ,     μ² = -λ v²
    """
    lam = m_h**2 / (2 * v**2)
    mu2 = -lam * v**2
    return lam, mu2


def higgs_mass(lam: float, v: float = HIGGS_VEV) -> float:
    """Return m_H for given λ and v (tree-level)."""
    return math.sqrt(2 * lam * v**2)
