"""Define helper functions for calculating equilibrium distribution values."""


def _feq2(r, u, v, s):
    """
    Compute the 2nd-order equilibrium distribution function.

    Parameters
    ----------
    r : np.ndarray or float
        Two-dimensional array containing the fluid density.
    u : np.ndarray or float
        Two-dimensional array containing the x-component of the fluid velocity.
    v : np.ndarray or float
        Two-dimensional array containing the y-component of the fluid velocity.
    s : Stencil
        The lattice stencil.

    Returns
    -------
    np.ndarray
        Three-dimensional array containing equilibrium distribution function values.
    """
    cu = s.cx[:, None, None] * u + s.cy[:, None, None] * v
    uu = u**2 + v**2
    return s.w[:, None, None] * r * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uu)


def _feq2_q(q, r, u, v, s):
    """
    Compute the 2nd-order equilibrium distribution function.

    Parameters
    ----------
    q : int
        Lattice vector index.
    r : np.ndarray or float
        Two-dimensional array containing the fluid density.
    u : np.ndarray or float
        Two-dimensional array containing the x-component of the fluid velocity.
    v : np.ndarray or float
        Two-dimensional array containing the y-component of the fluid velocity.
    s : Stencil
        The lattice stencil.

    Returns
    -------
    np.ndarray
        Three-dimensional array containing equilibrium distribution function values.
    """
    cu = s.cx[q] * u + s.cy[q] * v
    uu = u**2 + v**2
    return s.w[q] * r * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uu)
