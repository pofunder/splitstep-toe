# src/splitstep_toe/core/__init__.py
"""
Core numerical kernels.
"""

from .engine    import step_2d
from .laplacian import laplacian_2d

__all__ = ["step_2d", "laplacian_2d"]
from .fft import laplacian_fft        # noqa: F401
