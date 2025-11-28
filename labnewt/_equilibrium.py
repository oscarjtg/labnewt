"""Define helper functions for calculating equilibrium distribution values."""


def _feq2(r, u, v, s):
    """
    Compute the 2nd-order equilibrium distribution function.

    Parameters
    ----------
    r : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains fluid density.
    u : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains x-component of fluid velocity.
    v : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains y-component of fluid velocity.
    s : Stencil
        Stencil object defining lattice velocities and weights in arrays of length nq.

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
    Compute the 2nd-order equilibrium distribution function in lattice direction `q`.

    Parameters
    ----------
    q : int
        Integer lattice velocity index.
        Require q < nq, where nq is the length of the arrays in stencil.
    r : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains fluid density.
    u : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains x-component of fluid velocity.
    v : np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float.
        Contains y-component of fluid velocity.
    s : Stencil
        Stencil object defining lattice velocities and weights in arrays of length nq.

    Returns
    -------
    np.ndarray or float
        Two-dimensional array of floats of shape (ny, nx), or float
        giving equilibrium distribution function value(s) in lattice direction `q`.
    """
    cu = s.cx[q] * u + s.cy[q] * v
    uu = u**2 + v**2
    return s.w[q] * r * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uu)
