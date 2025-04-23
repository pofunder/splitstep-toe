"""
Solve (λ, μ²) from the SM Higgs potential

    V(Φ) = μ² |Φ|² + λ |Φ|⁴

with vacuum expectation value  v ≈ 246 GeV and Higgs mass  m_H.
In tree level:

    λ = m_H² /(2 v²)      ,     μ² = −λ v²
"""

from typing import Tuple
import math

_VEV = 246.0  # GeV

def lam_mu2_from_mh(m_h: float = 125.0, v: float = _VEV) -> Tuple[float, float]:
    """Return (λ, μ²) from Higgs mass and vev."""
    lam = m_h**2 / (2 * v**2)
    mu2 = -lam * v**2
    return lam, mu2

def higgs_mass(lam: float, v: float = _VEV) -> float:
    """Return m_H computed from λ (tree-level)."""
    return math.sqrt(2 * lam) * v
