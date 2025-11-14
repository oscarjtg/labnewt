"""Volume of fluid (free surface) helper functions."""

import numpy as np


def _dMq_(dMq, fo, phi, F_mask, I_mask, G_mask, s):
    """
    Computes mass exchange in each lattice direction (q) at each cell (y, x).

    Modifies `dMq` in place.

    `fo`, `phi`, `F_mask`, `I_mask`, `G_mask`, and `s` are read-only
    and are not changed.

    Parameters
    ----------
    dMq : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains mass exchanged in each lattice direction.
    fo : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains outgoing distribution functions.
    phi : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell fill fractions.
    F_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks FLUID cells.
    I_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks INTERFACE cells.
    G_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks GAS cells.
    s : Stencil
        Lattice stencil object.

    Returns
    -------
    None
    """
    nq, ny, nx = fo.shape
    assert nq == s.nq
    assert phi.shape == (ny, nx)
    assert F_mask.shape == (ny, nx)
    assert I_mask.shape == (ny, nx)
    assert G_mask.shape == (ny, nx)
    assert (F_mask * G_mask).all() is np.False_
    assert (F_mask * I_mask).all() is np.False_
    assert (I_mask * G_mask).all() is np.False_

    y = np.arange(ny)
    x = np.arange(nx)

    # _nq means "neighbour in q direction"
    y_nq = (y[np.newaxis, :, np.newaxis] + s.ey[:, np.newaxis, np.newaxis]) % ny
    x_nq = (x[np.newaxis, np.newaxis, :] + s.ex[:, np.newaxis, np.newaxis]) % nx

    idx_qrev = s.q_rev[:, np.newaxis, np.newaxis]

    f_nq = fo[idx_qrev, y_nq, x_nq]
    f_here = fo

    phi_nq = phi[y_nq, x_nq]
    phi_here = phi[np.newaxis, ...]

    I_here = I_mask[np.newaxis, ...]
    I_nq = I_mask[y_nq, x_nq]
    G_nq = G_mask[y_nq, x_nq]

    A = np.where(
        ~I_here,
        0.0,
        np.where(G_nq, 0.0, np.where(I_nq, 0.5 * (phi_here + phi_nq), 1.0)),
    )

    np.multiply(A, (f_nq - f_here), out=dMq)
