"""Implemented LocalAverageRefiller class."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

from .base import Refiller
from .._equilibrium import _feq2_q


class LocalAverageRefiller(Refiller):
    def __init__(self, density=1.0, velocity_x=0.0, velocity_y=0.0, pad=1):
        self.pad = pad
        self.r = density
        self.u = velocity_x
        self.v = velocity_y

        self.neighbours = np.ones((1 + 2 * pad, 1 + 2 * pad))
        self.neighbours[pad, pad] = 0.0
        self.neighbours.flags.writeable = False

    def fill(self, model, needs_filling: NDArray[np.bool_]):
        """
        Fills `model.fi` at `needs_filling` with average of GAS neighbours.

        Modifies `model.fi` in-place. `needs_filling` is read-only and is not changed.

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
        assert needs_filling.shape == model.shape

        mask = model.vof.F_mask
        count = convolve2d(mask, self.neighbours, boundary="wrap")[
            self.pad : -self.pad, self.pad : -self.pad
        ]
        assert count.shape == model.shape
        local_average = np.empty_like(model.r)
        for q in range(model.stencil.nq):
            local_average[:] = convolve2d(
                model.fi[q, ...] * mask, self.neighbours, boundary="wrap"
            )[self.pad : -self.pad, self.pad : -self.pad]
            no_neighbours = count == 0
            local_average[no_neighbours] = _feq2_q(
                q, self.r, self.u, self.v, model.stencil
            )
            local_average[~no_neighbours] = (
                local_average[~no_neighbours] / count[~no_neighbours]
            )
            assert local_average.shape == model.shape
            model.fi[q, needs_filling] = local_average[needs_filling]
        model.macros.density(model)
        model.macros.velocity_x(model)
        model.macros.velocity_y(model)
