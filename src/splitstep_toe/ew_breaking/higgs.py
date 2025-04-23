"""
Tree-level Higgs potential utilities
====================================

V(φ) = μ² φ†φ + λ (φ†φ)² with  
μ² < 0   (thus ⟨φ⟩ ≠ 0)  

Minimisation gives  
 v² = −μ² / λ, m_H² = 2 λ v².
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "higgs_vev",
    "lam_mu2_from_mh",
    "higgs_mass",
]

# electroweak vacuum expectation value in GeV
_VEV = 246.0


def higgs_vev() -> float:
    """Return v ≈ 246 GeV."""
    return _VEV


def lam_mu2_from_mh(m_h: float = 125.0) -> tuple[float, float]:
    """
    Solve tree-level relations for λ and μ² from an input Higgs mass.

    Returns
    -------
    λ : quartic coupling (dimensionless, > 0)  
    μ² : (negative) mass-squared parameter driving symmetry breaking
    """
    v = higgs_vev()
    lam = m_h**2 / (2.0 * v**2)
    mu2 = -m_h**2 / 2.0          # ***must be negative***
    return lam, mu2


def higgs_mass(lam: float, v: float | None = None) -> float:
    """Higgs mass from λ (tree-level)."""
    if v is None:
        v = higgs_vev()
    return np.sqrt(2.0 * lam) * v
