"""Functions for calculating macroscopic variables from distribution functions"""

import numpy as np


def _m0(f):
    """
    Calculates first moment of distribution function `f`.

        m0[y, x] = sum_q f[q, y, x]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    Returns
    -------
    m0 : np.ndarray
        Two-dimensional numpy array containing density values.
    """
    return np.sum(f, axis=0)


def _mx(f, s):
    """
    Calculates first moment mx.

        mx[y, x] = sum_q f[q, y, x] * s.cx[q]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    s : Stencil
        Stencil instance containing attribute cx, a one-dimensional numpy array.

    Returns
    -------
    mx : np.ndarray
        Two-dimensional numpy array containins x-component of velocity values.
    """
    return np.sum(f * s.cx[:, None, None], axis=0)


def _my(f, s):
    """
    Calculates first moment my.

        my[y, x] = sum_q f[q, y, x] * s.cy[q]

    Parameters
    ----------
    f : np.ndarray
        Three-dimensional numpy array containing distribution function values.

    s : Stencil
        Stencil instance containing attribute cy, a one-dimensional numpy array.

    Returns
    -------
    my : np.ndarray
        Two-dimensional numpy array containins y-component of velocity values.
    """
    return np.sum(f * s.cy[:, None, None], axis=0)


def _m1(dim, f, s):
    """
    Calculates first moment in direction dim.

        md[y, x] = sum_q f[q, y, x] * s.cd[q]

    where
        cd = cx if d = 0
        cd = cy if d = 1

    Parameters
    ----------
    dim : int
        Integer giving the direction along which to compute the moment.
        Should be 0 or 1 only.

    f : np.ndarray
        Three-dimensional numpy array containing distribution function values,
        f[q, y, x].

    s : Stencil
        Stencil instance containing attribute cy, a one-dimensional numpy array.
    """
    c = s.cx if dim == 0 else s.cy
    return np.sum(f * c[:, None, None], axis=0)
