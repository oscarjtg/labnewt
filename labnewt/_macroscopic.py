"""Functions for calculating macroscopic variables from distribution functions"""

import numpy as np


def _density(f):
    """
    Calculates macroscopic density from distribution function `f`.

        r[y, x] = sum_q f[q, y, x]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    Returns
    -------
    r : np.ndarray
        Two-dimensional numpy array containing density values.
    """
    return np.sum(f, axis=0)


def _velocity_x(f, r, s):
    """
    Calculates the x-component of macroscopic velocity from distribution function `f`.

        u[y, x] = (sum_q f[q, y, x] * s.cx[q]) / r[y, x]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    r : np.narray
        Two-dimensional numpy array containing density values.

    s : Stencil
        Stencil instance containing attribute cx, a one-dimensional numpy array.

    Returns
    -------
    u : np.ndarray
        Two-dimensional numpy array containins x-component of velocity values.
    """
    return np.sum(f * s.cx[:, None, None], axis=0) / r


def _velocity_y(f, r, s):
    """
    Calculates the x-component of macroscopic velocity from distribution function `f`.

        v[y, x] = (sum_q f[q, y, x] * s.cy[q]) / r[y, x]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    r : np.narray
        Two-dimensional numpy array containing density values.

    s : Stencil
        Stencil instance containing attribute cy, a one-dimensional numpy array.

    Returns
    -------
    v : np.ndarray
        Two-dimensional numpy array containins y-component of velocity values.
    """
    return np.sum(f * s.cy[:, None, None], axis=0) / r
