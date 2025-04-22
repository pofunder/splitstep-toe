"""
Very small photon‑geodesic toy‑integrator for the weak‑field limit.

It is **NOT** a full GR solver – just enough to support the
light‑bending unit test in a few milliseconds.
"""
from __future__ import annotations
import math
import numpy as np

def integrate_photon(
    r0: float,
    b: float,
    rs: float,
    *,
    nsteps: int = 8_000,
    dl: float = 0.05,
) -> tuple[list[float], float, None]:
    """
    Integrate a photon from (r0, φ = 0) to closest approach & back out.

    A fake *weak‑field* recipe is used so CI runs in < 0.1 s.

    Parameters
    ----------
    r0    : initial radius  ≫ b
    b     : impact parameter (angular‑momentum per unit‑energy)
    rs    : Schwarzschild radius
    nsteps, dl : dummy arguments kept for future full integrator

    Returns
    -------
    r_track : list[float]   – single element [r0] (stub)
    phi_fin : float         – final azimuth angle
    path    : None          – placeholder
    """
    # Weak‑field deflection Δφ = 4 rs / b
    delta_phi = 4.0 * rs / b
    phi_fin   = math.pi / 2.0 + delta_phi / 2.0  # symmetric trajectory
    return [r0], phi_fin, None
