import numpy as np
from numpy.typing import NDArray

from labnewt.stencil import Stencil
from labnewt.streamer import Streamer


def _distribute_Mex(
    M_star: NDArray[np.float64],
    rho: NDArray[np.float64],
    norm_x: NDArray[np.float64],
    norm_y: NDArray[np.float64],
    I_mask: NDArray[np.bool_],
    s: Stencil,
) -> NDArray[np.float64]:
    """
    Distribute excess mass. Returns array with cell mass after redistribution.

    All inputs are read-only and are not changed.

    Parameters
    ----------
    M_star : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains mass at each grid cell redistributed after streaming.
    rho : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains density at each grid cell after streaming.
    norm_x : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains x-component of unit surface normal.
    norm_y : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains y-component of unit surface normal.
    I_mask : NDArray[np.bool]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at INTERFACE cells, otherwise `False`.
    s : Stencil
        Lattice stencil.

    Returns
    -------
    M : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell mass after redistribution of excess mass
    """
    # Identify cells that are converting...
    to_gas = I_mask & (M_star < 0)
    to_fluid = I_mask & (M_star > rho)

    M = np.copy(M_star)
    _distribute_Mex_(M, rho, norm_x, norm_y, to_fluid, to_gas, s)

    return M


def _distribute_Mex_(
    M: NDArray[np.float64],
    rho: NDArray[np.float64],
    norm_x: NDArray[np.float64],
    norm_y: NDArray[np.float64],
    to_fluid: NDArray[np.bool_],
    to_gas: NDArray[np.bool_],
    s: Stencil,
) -> None:
    """
    Distribute excess mass.

    Modifies `M` in-place

    Parameters
    ----------
    M : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains mass at each grid cell redistributed after streaming.
    rho : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains density at each grid cell after streaming.
    norm_x : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains x-component of unit surface normal.
    norm_y : NDArray[np.float64]
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains y-component of unit surface normal.
    to_fluid : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->F transition.
    to_gas : NDArray[np.bool_]
        Two-dimensional numpy array of bools of shape (ny, nx).
        `True` at cells that have undergone I->G transition.
    s : Stencil
        Lattice stencil.

    Returns
    -------
    None
    """
    ny, nx = M.shape
    nq = s.nq
    dots = (
        s.ex[:, np.newaxis, np.newaxis] * norm_x[np.newaxis, :, :]
        + s.ey[:, np.newaxis, np.newaxis] * norm_y[np.newaxis, :, :]
    )

    # Excess mass
    M_ex_neg = np.where(to_gas, -M, 0.0)
    M_ex_pos = np.where(to_fluid, M - rho, 0.0)

    # Selectors
    sel_neg = (dots < 0) & to_gas[np.newaxis, :, :]
    sel_pos = (dots > 0) & to_fluid[np.newaxis, :, :]

    # Count selected neighbours of each source / sink
    count_neg = sel_neg.sum(axis=0, dtype=np.float64)
    count_pos = sel_pos.sum(axis=0, dtype=np.float64)

    eps = 1.0e-12
    count_neg += eps
    count_pos += eps

    Mqo = np.zeros((nq, ny, nx))
    Mqo -= sel_neg * M_ex_neg / count_neg
    Mqo += sel_pos * M_ex_pos / count_pos

    Mqi = np.empty_like(Mqo)

    Streamer()._stream(Mqi, Mqo, s)

    M[:] = M + np.sum(Mqi, axis=0) + M_ex_neg - M_ex_pos
