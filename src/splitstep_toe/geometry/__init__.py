"""
Geometry sub‑package – minimal General‑Relativity helpers.

Public API
----------
schwarzschild_f  : lapse function 1 − rs/r
integrate_photon : weak‑field photon geodesic integrator
"""
from .metric    import schwarzschild_f
from .geodesic  import integrate_photon

__all__ = [
    "schwarzschild_f",
    "integrate_photon",
]
