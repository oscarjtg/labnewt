"""Defines functions for calculating M^star."""

import numpy as np
from numpy.typing import NDArray


def _Mstar_inplace(M: NDArray[np.float64], dMq: NDArray[np.float64]) -> None:
    """
    Computes mass after mass exchange but before cell conversion.

    M^star = M + sum_q dM_q

    Writes M^star directly into array `M`.
    Modifies `M`. Does not modify `dMq`.

    Parameters
    ----------
    M : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell mass before mass exchange.
        Modified by the function to contain cell mass after mass exchange.
    dMq : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains incoming exchanged mass from each lattice direction q.

    Returns
    -------
    None
    """
    assert M.shape == dMq.shape[1:]
    np.add(M, np.sum(dMq, axis=0), out=M)
