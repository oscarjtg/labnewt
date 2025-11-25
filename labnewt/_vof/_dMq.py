"""Mass exchange functions, acting at FLUID and INTERFACE cells."""

import numpy as np
from numpy.typing import NDArray

from ..stencil import Stencil


def _compute_dMq_(
    dMq: NDArray[np.float64],
    f_in: NDArray[np.float64],
    f_out: NDArray[np.float64],
    phi: NDArray[np.float64],
    x_nq: NDArray[np.int64],
    y_nq: NDArray[np.int64],
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
):
    """
    Computes mass exchange in each lattice direction (q) at each cell.

    Modifies `dMq` in place.

    `fo`, `phi`, `x_nq`, `y_nq`, `F_mask`, and `I_mask` are read-only
    and are not changed.

    Parameters
    ----------
    dMq : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains mass exchanged in each lattice direction.
    f_in : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains incoming distribution functions from direction q.
    f_out : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains outgoing distribution functions, going to direction q.
    phi : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell fill fractions.
    x_nq : np.ndarray
        Three-dimensional numpy array of ints of shape (ny, nx).
        Contains x indices of neighbouring cells.
    y_nq : np.ndarray
        Three-dimensional numpy array of ints of shape (ny, nx).
        Contains y indices of neighbouring cells.
    F_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks FLUID cells.
    I_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks INTERFACE cells.

    Returns
    -------
    None
    """
    phi_nq = phi[y_nq, x_nq]
    phi_here = phi[np.newaxis, ...]

    G_mask = ~np.logical_or(F_mask, I_mask)

    I_here = I_mask[np.newaxis, ...]
    G_here = G_mask[np.newaxis, ...]

    I_nq = I_mask[y_nq, x_nq]
    G_nq = G_mask[y_nq, x_nq]

    A = np.where(
        G_here,
        0.0,
        np.where(G_nq, 0.0, np.where(I_here & I_nq, 0.5 * (phi_here + phi_nq), 1.0)),
    )

    np.multiply(A, (f_in - f_out), out=dMq)


def _dMq_(
    dMq: NDArray[np.float64],
    fo: NDArray[np.float64],
    phi: NDArray[np.float64],
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    s: Stencil,
) -> None:
    """
    Computes mass exchange in each lattice direction (q) at each cell.

    Modifies `dMq` in place.

    `fo`, `phi`, `F_mask`, `I_mask`, and `s` are read-only
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
    assert not np.any(F_mask & I_mask)

    y = np.arange(ny)
    x = np.arange(nx)

    # _nq means "neighbour in q direction"
    y_nq = _compute_y_nq(y, ny, s)
    x_nq = _compute_x_nq(x, nx, s)

    idx_qrev = s.q_rev[:, np.newaxis, np.newaxis]

    f_nq = fo[idx_qrev, y_nq, x_nq]
    f_here = fo

    _compute_dMq_(dMq, f_nq, f_here, phi, x_nq, y_nq, F_mask, I_mask)


def _dMq(
    fo: NDArray[np.float64],
    phi: NDArray[np.float64],
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    s: Stencil,
) -> NDArray[np.float64]:
    """
    Computes mass exchange in each lattice direction (q) at each cell.

    `fo`, `phi`, `F_mask`, `I_mask`, and `s` are read-only
    and are not changed.

    Parameters
    ----------
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
    s : Stencil
        Lattice stencil object.

    Returns
    -------
    dMq : np.ndarray
        Three-dimensional numpy array of floats of shape (nq, ny, nx).
        Contains mass exchanged in each lattice direction.
    """
    dMq = np.empty_like(fo)
    _dMq_(dMq, fo, phi, F_mask, I_mask, s)
    return dMq


def _compute_x_nq(x, nx, s):
    return (x[np.newaxis, np.newaxis, :] + s.ex[:, np.newaxis, np.newaxis]) % nx


def _compute_y_nq(y, ny, s):
    return (y[np.newaxis, :, np.newaxis] + s.ey[:, np.newaxis, np.newaxis]) % ny
