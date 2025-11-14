from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _normals_(
    norm_x: NDArray[np.float64],
    norm_y: NDArray[np.float64],
    phi: NDArray[np.float64],
    I_mask: NDArray[np.bool_],
    dx: float = 1.0,
    dy: float = 1.0,
) -> None:
    """
    Fill `norm_x` and `norm_y` in place with the components of the unit normal.
    n = -grad(phi) / |grad(phi)| on interface cells (where `I_mask` is `True`).
    On non-interface cells (where `I_mask` is `False`),
    `nx` and `ny` are set to `np.nan`.

    Parameters
    ----------
    norm_x : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains x-components of interface unit normal vectors.
    norm_y : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains y-components of interface unit normal vectors.
    phi : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell fill fractions.
    I_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks INTERFACE cells.
    dx : float, optional
        Grid spacing in x direction. Default = 1.0.
    dy : float, optional
        Grid spacing in y direction. Default = 1.0.

    Returns
    -------
    None
    """
    assert norm_x.shape == norm_y.shape
    assert norm_x.shape == phi.shape
    assert norm_x.shape == I_mask.shape

    # Compute gradient of phi.
    grad_y, grad_x = np.gradient(phi, dy, dx)

    # Compute magnitude of gradient.
    mag = np.empty_like(phi)
    np.hypot(grad_x, grad_y, out=mag)

    norm_x.fill(np.nan)
    norm_y.fill(np.nan)

    norm_x[I_mask] = -grad_x[I_mask] / mag[I_mask]
    norm_y[I_mask] = -grad_y[I_mask] / mag[I_mask]


def _normals(
    phi: NDArray[np.float64],
    I_mask: NDArray[np.bool_],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculates the components of the surface unit normal.
    n = -grad(phi) / |grad(phi)| on interface cells (where `I_mask` is `True`).
    On non-interface cells (where `I_mask` is `False`),
    `nx` and `ny` are set to `np.nan`.

    Parameters
    ----------
    phi : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell fill fractions.
    I_mask : np.ndarray
        Two-dimensional numpy array of bools of shape (ny, nx).
        Marks INTERFACE cells.
    dx : float, optional
        Grid spacing in x direction. Default = 1.0.
    dy : float, optional
        Grid spacing in y direction. Default = 1.0.

    Returns
    -------
    norm_x : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains x-components of interface unit normal vectors.
    norm_y : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains y-components of interface unit normal vectors.
    """
    norm_x, norm_y = np.empty_like(phi), np.empty_like(phi)
    _normals_(norm_x, norm_y, phi, I_mask, dx, dy)
    return norm_x, norm_y
