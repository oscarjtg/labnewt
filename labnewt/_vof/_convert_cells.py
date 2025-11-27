"""Cell type conversion functions."""

import numpy as np
from numpy.typing import NDArray


def _convert_cells_(
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    G_mask: NDArray[np.bool_],
    cIF: NDArray[np.bool_],
    cIG: NDArray[np.bool_],
) -> None:
    """
    Convert cell types of neighbours of over- or under-filled cells.

    Modifies `F_mask`, `I_mask`, and `G_mask` in-place.
    `cIF` and `cIG` are read-only and are not changed.

    Parameters
    ----------
    F_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at FLUID cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    G_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at GAS cells.
    cIF : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->F transition.
    cIG : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I-> G transition.

    Returns
    -------
    None
    """
    _convert_GI_(G_mask, I_mask, cIF)
    _convert_FI_(F_mask, I_mask, cIG)


def _convert_GI_(
    G_mask: NDArray[np.bool_], I_mask: NDArray[np.bool_], cIF: NDArray[np.bool_]
) -> None:
    """
    For cIF cells, convert neighbouring GAS cells to INTERFACE cells.

    Modifies `G_mask` and `I_mask` in-place.
    `cIF` is read-only and is not changed.

    Parameters
    ----------
    G_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at GAS cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    cIF : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->F transition.

    Returns
    -------
    None
    """
    pass


def _convert_FI_(
    F_mask: NDArray[np.bool_], I_mask: NDArray[np.bool_], cIG: NDArray[np.bool_]
) -> None:
    """
    For cIG cells, convert neighbouring FLUID cells to INTERFACE cells.

    Modifies `F_mask` and `I_mask` in-place.
    `cIG` is read-only and is not changed.

    Parameters
    ----------
    F_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at FLUID cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    cIG : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I-> G transition.

    Returns
    -------
    None
    """
    pass
