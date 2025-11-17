import numpy as np
from numpy.typing import NDArray

from labnewt import Streamer
from labnewt.stencil import Stencil


def _distribute_Mex(
    M_star: NDArray[np.float64],
    rho: NDArray[np.float64],
    norm_x: NDArray[np.float64],
    norm_y: NDArray[np.float64],
    I_mask: NDArray[np.bool_],
    s: Stencil,
) -> NDArray[np.float64]:
    """
    Distribute excess mass.

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
    ny, nx = M_star.shape
    nq = s.nq
    dots = (
        s.ex[:, np.newaxis, np.newaxis] * norm_x[np.newaxis, :, :]
        + s.ey[:, np.newaxis, np.newaxis] * norm_y[np.newaxis, :, :]
    )

    # Identify cells that are converting...
    to_gas = I_mask & (M_star < 0)
    to_fluid = I_mask & (M_star > rho)

    # Excess mass
    M_ex_neg = np.where(to_gas, -M_star, 0.0)
    M_ex_pos = np.where(to_fluid, M_star - rho, 0.0)

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

    Streamer().stream(Mqi, Mqo, s)

    return M_star + np.sum(Mqi, axis=0) + M_ex_neg - M_ex_pos
