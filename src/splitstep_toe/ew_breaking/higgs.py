"""
Tree-level Higgs potential utilities
V(φ) = − μ² |φ|² + λ |φ|⁴
"""

from __future__ import annotations
import math

_VEV = 246.22  # GeV  (PDG central value)


def lam_mu2_from_mh(m_h: float = 125.0, v: float = _VEV) -> tuple[float, float]:
    """
    Solve λ and μ² from the physical Higgs mass (tree level).

    Returns (λ, μ²) such that:
        m_h² = 2 λ v²   and   μ² = λ v²
    """
    lam = m_h**2 / (2 * v**2)
    mu2 = m_h**2 / 2.0
    return lam, mu2


def higgs_mass(lam: float, v: float = _VEV) -> float:
    """Invert the above: m_h from a given λ and v."""
    return math.sqrt(2 * lam) * v


def higgs_vev() -> float:
    """Reference vev used by unit tests."""
    return _VEV
