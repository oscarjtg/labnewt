import numpy as np


def periodic_shift(arr, s, q):
    """
    Parameters
    ----------
    arr : np.ndarray
        Two dimensional numpy array

    s : Stencil
        Stencil

    q : int
        Lattice vector

    Returns
    -------
    arr_shifted : np.ndarray
        arr_shifted[y, x] = arr[(y - s.ey[q]) % Ny, (x - s.ex[q]) % Nx]
    """
    return np.roll(np.roll(arr, s.ey[q], axis=0), s.ex[q], axis=1)
