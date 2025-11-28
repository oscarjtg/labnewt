"""
Implements the Refiller base class.

This demonstrates the public interface for classes derived from Refiller.
"""

import numpy as np
from numpy.typing import NDArray

from ..model import Model


class Refiller:
    def fill(self, model: Model, needs_filling: NDArray[np.bool_]):
        raise NotImplementedError
