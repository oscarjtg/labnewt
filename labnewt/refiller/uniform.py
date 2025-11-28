"""Implements UniformRefiller class"""

import numpy as np
from numpy.typing import NDArray

from .._equilibrium import _feq2
from ..model import Model
from .base import Refiller


class UniformRefiller(Refiller):
    def __init__(self, density: float, velocity_x: float, velocity_y: float):
        """
        Parameters
        ----------
        density : float
            Float giving the density of refilled cells.
        velocity_x : float
            Float giving the x-component of velocity of refilled cells.
        velocity_y : float
            Float giving the y-component of velocity of refilled cells.
        """
        self.r = density
        self.u = velocity_x
        self.v = velocity_y

    def fill(self, model: Model, needs_filling: NDArray[np.bool_]):
        """
        Fill cells where `needs_filling` is `True`.

        Modifies `model.fi`, `model.r`, `model.u`, and `model.v` in-place.
        Sets `model.r` to `self.r`, `model.u` to `self.u, `model.v` to `self.v`,
        and `model.fi` to equilibrium distribution with the above macroscopic values.

        Parameters
        ----------
        model : Model or FreeSurfaceModel
            Model object. Some arrays contained within `model` may be modified.
        needs_filling : NDArray[np.bool_]
            Two-dimensional numpy array of bools of shape (ny, nx).
            Contains `True` at cells that are to be refilled.

        Returns
        -------
        None
        """
        model.r[needs_filling] = self.r
        model.u[needs_filling] = self.u
        model.v[needs_filling] = self.v
        model.fi[:, needs_filling] = _feq2(self.r, self.u, self.v, model.stencil)
