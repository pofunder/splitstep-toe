"""
Electroweak-symmetry breaking helpers (placeholder).

Exports
-------
higgs_vev()  – returns 246 GeV
"""
from .higgs import higgs_vev

__all__ = ["higgs_vev"]
from .higgs import lam_mu2_from_mh, higgs_mass   # noqa: F401

