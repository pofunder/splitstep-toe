"""
Dark-matter playground.

Exports
-------
leapfrog_step  – symplectic 2-body Leapfrog
demo_two_body  – evolve two equal-mass halos, return separation vs time
"""

from .leapfrog import leapfrog_step, demo_two_body

__all__ = ["leapfrog_step", "demo_two_body"]

