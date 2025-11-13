"""Functions for computing diagnostics from model output."""

import numpy as np


def relative_error(X_approx, X_ref):
    """
    Computes the L2 error norm of `X_approx` comapred to `X_ref`.

    L2 error = sum_k (X_approx[k] - X_ref[k])^2 / sum_k X_ref[k]^2

    Parameters
    ----------
    X_approx : np.ndarray
        One-dimensional numpy array of floats containing approximate values
        of some quantity
    X_ref : np.ndarray
        One-dimensional numpy array of floats containing reference values
        to which we wish to compare the approximate values in `X_approx`

    Returns
    -------
    error : float
        The L2 error norm
    """
    return np.linalg.norm(X_approx - X_ref) / np.linalg.norm(X_ref)
