"""
Schwarzschild metric helpers (geometrised units G = c = 1)
"""
def schwarzschild_f(r: float, rs: float) -> float:
    """
    Lapse function f(r) = 1 − rs / r

    Parameters
    ----------
    r  : radial coordinate ( > rs )
    rs : Schwarzschild radius (= 2 GM)

    Returns
    -------
    f : float
    """
    return 1.0 - rs / r
