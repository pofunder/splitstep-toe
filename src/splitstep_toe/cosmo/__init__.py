"""
Minimal FLRW cosmology helpers.

Currently supports:
    • EdS (Ω_m = 1, Ω_Λ = 0)
    • Flat ΛCDM  (Ω_m + Ω_Λ = 1)

Exports
-------
scale_factor(t, ...)   : integrate Friedmann 1-D ODE a(t)
Hub  = Hubble parameter helper
"""

from .scale_factor import scale_factor, Hub      # noqa: F401

__all__ = ["scale_factor", "Hub"]

