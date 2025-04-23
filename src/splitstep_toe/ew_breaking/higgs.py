"""
Tree-level Higgs-sector utilities.

The SM potential (φ†φ) has parameters

    V(φ) = μ²(φ†φ) + λ(φ†φ)²      with   μ² < 0   and   λ > 0.

The vacuum expectation value (VEV) v ≃ 246 GeV and physical Higgs
mass m_H ≃ 125 GeV fix (λ, μ²).

Relations
---------
λ   = m_H² / (2 v²)
μ² = −m_H² / 2
"""

from __future__ import annotations

import math

# Electroweak VEV (G_F → v)  ----------
V_SM: float = 246.0  # GeV


# ──────────────────────────────────────────────────────────────────────────────
def lam_mu2_from_mh(m_h: float = 125.0, v: float = V_SM) -> tuple[float, float]:
    """
    Compute λ and μ² from an input Higgs pole mass *m_h*.

    Returns
    -------
    (λ, μ²) with μ² < 0 signalling spontaneous symmetry breaking.
    """
    lam = m_h**2 / (2.0 * v**2)
    mu2 = -m_h**2 / 2.0
    return lam, mu2


def higgs_mass(lam: float, v: float = V_SM) -> float:
    """Invert the relation λ = m_H² / (2 v²)."""
    return math.sqrt(2.0 * lam * v**2)


def higgs_vev() -> float:
    """Return the SM Higgs vacuum expectation value v ≈ 246 GeV."""
    return V_SM
