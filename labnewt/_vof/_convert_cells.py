"""Cell type conversion functions."""

import numpy as np
from numpy.typing import NDArray

from ..streamer import periodic_shift


def _convert_cells_(
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    G_mask: NDArray[np.bool_],
    cIF: NDArray[np.bool_],
    cIG: NDArray[np.bool_],
) -> None:
    """
    Convert cell types of over- or under-filled cells and their neighbours.

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
    I_mask[cIF] = False
    I_mask[cIG] = False
    F_mask[cIF] = True
    G_mask[cIG] = True


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
    assert _check_subset(cIF, I_mask)
    neighbour_transitioned = _check_neighbours_true(cIF)
    cells_to_convert = G_mask & neighbour_transitioned
    G_mask[cells_to_convert] = False
    I_mask[cells_to_convert] = True


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
    assert _check_subset(cIG, I_mask)
    neighbour_transitioned = _check_neighbours_true(cIG)
    cells_to_convert = F_mask & neighbour_transitioned
    F_mask[cells_to_convert] = False
    I_mask[cells_to_convert] = True


def _check_subset(subset: NDArray[np.bool_], full: NDArray[np.bool_]) -> bool:
    """
    Check that every cell that is `True` in `subset` is also `True` in `full`.

    Parameters
    ----------
    subset : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        Contains values that should only be `True` at a subset of the cells
        where `full` contains `True`.
    full : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).

    Returns
    -------
    None
    """
    return np.all(np.logical_or(~subset, full))


class Neighbours2D:
    number_of_neighbours = 8
    ex = np.array([1, -1, 0, 0, 1, -1, -1, 1], dtype=np.int64)
    ey = np.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int64)

    ex.flags.writeable = False
    ey.flags.writeable = False


def _check_neighbours_true(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Make boolean array with `True` if any neighbouring cells in `mask` are `True`.

    Neighbours are adjacent and diagonal, i.e. there are eight neighbours.

    `mask` is read-only and not changed.

    Parameters
    ----------
    neighbour : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells with at least one `True` neighbour in `mask`.
    mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).

    Returns
    -------
    None
    """
    neighbours = np.zeros_like(mask).astype(np.bool_)
    _check_neighbours_true_(neighbours, mask)
    return neighbours


def _check_neighbours_true_(
    neighbour: NDArray[np.bool_], mask: NDArray[np.bool_]
) -> None:
    """
    Set `neighbour[y, x]` to `True` if any neighbouring cells in `mask` are `True`.

    Neighbours are adjacent and diagonal, i.e. there are eight neighbours.

    Modifies `neighbour` in-place. `mask` is read-only and not changed.

    Parameters
    ----------
    neighbour : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells with at least one `True` neighbour in `mask`.
    mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).

    Returns
    -------
    None
    """
    assert neighbour.shape == mask.shape, "neighbour and mask must have same shape"

    neighbour[:] = False  # Reset array

    stencil = Neighbours2D()

    for direction in range(stencil.number_of_neighbours):
        neighbour[:] |= periodic_shift(mask, stencil, direction)
