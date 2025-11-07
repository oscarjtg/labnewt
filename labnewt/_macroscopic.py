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


def _velocity_x(f, s):
    """
    Calculates the x-component of macroscopic velocity from distribution function `f`.

        u[y, x] = sum_q f[q, y, x] * s.cx[q]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    s : Stencil
        Stencil instance, which contains attribute cx which is a one-dimensional numpy array.

    Returns
    -------
    u : np.ndarray
        Two-dimensional numpy array containins x-component of velocity values.
    """
    return np.sum(f * s.cx[:, None, None], axis=0)
