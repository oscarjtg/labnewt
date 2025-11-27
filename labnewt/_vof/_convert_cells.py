"""Cell type conversion functions."""

import numpy as np
from numpy.typing import NDArray

from ..streamer import periodic_shift


def _convert_cells_(
    F_mask: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    G_mask: NDArray[np.bool_],
    to_fluid: NDArray[np.bool_],
    to_gas: NDArray[np.bool_],
) -> None:
    """
    Convert cell types of over- or under-filled cells and their neighbours.

    Modifies `F_mask`, `I_mask`, and `G_mask` in-place.
    `to_fluid` and `to_gas` are read-only and are not changed.

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
    to_fluid : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->F transition.
    to_gas : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->G transition.

    Returns
    -------
    None
    """
    _convert_GI_(G_mask, I_mask, to_fluid)
    _convert_FI_(F_mask, I_mask, to_gas)
    I_mask[to_fluid] = False
    I_mask[to_gas] = False
    F_mask[to_fluid] = True
    G_mask[to_gas] = True


def _convert_GI_(
    G_mask: NDArray[np.bool_], I_mask: NDArray[np.bool_], to_fluid: NDArray[np.bool_]
) -> None:
    """
    For to_fluid cells, convert neighbouring GAS cells to INTERFACE cells.

    Modifies `G_mask` and `I_mask` in-place.
    `to_fluid` is read-only and is not changed.

    Parameters
    ----------
    G_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at GAS cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    to_fluid : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->F transition.

    Returns
    -------
    None
    """
    assert _check_subset(to_fluid, I_mask)
    neighbour_transitioned = _check_neighbours_true(to_fluid)
    cells_to_convert = G_mask & neighbour_transitioned
    G_mask[cells_to_convert] = False
    I_mask[cells_to_convert] = True


def _convert_FI_(
    F_mask: NDArray[np.bool_], I_mask: NDArray[np.bool_], to_gas: NDArray[np.bool_]
) -> None:
    """
    For to_gas cells, convert neighbouring FLUID cells to INTERFACE cells.

    Modifies `F_mask` and `I_mask` in-place.
    `to_gas` is read-only and is not changed.

    Parameters
    ----------
    F_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at FLUID cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    to_gas : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->G transition.

    Returns
    -------
    None
    """
    assert _check_subset(to_gas, I_mask)
    neighbour_transitioned = _check_neighbours_true(to_gas)
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

    return


def _identify_underfull_(
    to_gas: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    M: NDArray[np.float64],
    eps: float = 1.0e-06,
) -> None:
    """
    Identify over-full INTERFACE cells; mark them in `to_gas`.

    An underfull cell is one where M < -eps.

    Modifies `to_gas` in-place.

    Parameters
    ----------
    to_gas : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        Modified in-place to contain `True` at underfull INTERFACE cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    M : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell mass (fill fraction * density).
    rho : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell density.
    eps : float
        Float giving a numerical regularisation on the "underfull" condition.

    Returns
    -------
    None
    """
    np.less(M, -eps, out=to_gas)
    np.logical_and(to_gas, I_mask, out=to_gas)


def _identify_overfull_(
    to_fluid: NDArray[np.bool_],
    I_mask: NDArray[np.bool_],
    M: NDArray[np.float64],
    rho: NDArray[np.float64],
    eps: float = 1.0e-06,
) -> None:
    """
    Identify over-full INTERFACE cells; mark them in `to_fluid`.

    An overfull cell is one where M > rho + eps.

    Modifies `to_fluid` in-place.

    Parameters
    ----------
    to_fluid : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        Modified in-place to contain `True` at overfull INTERFACE cells.
    I_mask : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells.
    M : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell mass (fill fraction * density).
    rho : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell density.
    eps : float
        Float giving a numerical regularisation on the "overfull" condition.

    Returns
    -------
    None
    """
    np.greater(M + eps, rho, out=to_fluid)
    np.logical_and(to_fluid, I_mask, out=to_fluid)
