import numpy as np
from numpy.typing import NDArray

from ._shift import periodic_shift
from .stencil import Stencil


class Streamer:
    def stream(self, model):
        """
        Streams particles in `model.fo` to `model.fi`.

            fi[q, y, x] = fo[q, y - s.ey[q], x - s.ex[q]]

        Modifies `model.fi` in place.
        `model.fo` is read-only and remains unchanged.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        self._stream(model.fi, model.fo, model.stencil)

    def _stream(
        self, fi: NDArray[np.float64], fo: NDArray[np.float64], s: Stencil
    ) -> None:
        """
        Streams particles in array `fo` by directions `s.ex` and `s.ey` to array `fi`.

            fi[q, y, x] = fo[q, y - s.ey[q], x - s.ex[q]]

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of floats of shape (nq, ny, nx).
            Contains incoming particle distribution functions.
        fo : np.ndarray
            Three-dimensional numpy array of floats of shape (nq, ny, nx).
            Contains outgoing particle distribution functions.
        s : Stencil
            Lattice stencil

        Returns
        -------
        None
        """
        for q in range(s.nq):
            fi[q, :, :] = periodic_shift(fo[q, :, :], s, q)
