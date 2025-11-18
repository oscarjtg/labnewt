"""Functions for computing diagnostics from model output."""

import numpy as np
from numpy.typing import NDArray


def relative_error(X_approx, X_ref):
    """
    Computes the L2 error norm of `X_approx` comapred to `X_ref`.

    L2 error = sum_k (X_approx[k] - X_ref[k])^2 / sum_k X_ref[k]^2

    Parameters
    ----------
    X_approx : NDArray[np.float64]
        One-dimensional numpy array of floats containing approximate values
        of some quantity
    X_ref : NDArray[np.float64]
        One-dimensional numpy array of floats containing reference values
        to which we wish to compare the approximate values in `X_approx`

    Returns
    -------
    error : float
        The L2 error norm
    """
    return np.linalg.norm(X_approx - X_ref) / np.linalg.norm(X_ref)


def average_difference(X_approx, X_ref, mask=None):
    """
    Computes the average element-wise absolute difference in the input arrays.

    If `mask` is `None`:

        Difference = sum_k (X_approx[k] - X_ref[k]) / N

    where N is the number of elements in each array.

    If `mask` is not `None` (it should be a boolean array):

        Difference = sum_k (X_approx[k] - X_ref[k]) * mask[k] / N

    where N is the number of `True` values in the `mask`.
    
    Parameters
    ----------
    X_approx : NDArray[np.float64]
        Numpy array of floats containing approximate values of some quantity.
    X_ref : NDArray[np.float64]
        Numpy array of floats containing reference values
        to which we wish to compare the approximate values in `X_approx`.
        Must have the same shape as `X_approx`.
    mask : None or NDArray[np.bool_]
        Boolean array indicating the cells that should be included.
        Must have the same shape as `X_ref`.

    Returns
    -------
    difference : float
        The average element-wise absolute difference.
    """
    assert X_approx.shape == X_ref.shape
    N = X_approx.size
    if mask is not None:
        assert mask.shape == X_ref.shape
        X_approx *= mask
        X_ref *= mask
        N = np.sum(mask)
    return np.sum(X_approx - X_ref) / N
