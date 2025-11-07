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
