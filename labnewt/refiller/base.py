"""
Implements the Refiller base class.

This demonstrates the public interface for classes derived from Refiller.
"""

import numpy as np
from numpy.typing import NDArray


class Refiller:
    def fill(self, model, needs_filling: NDArray[np.bool_]):
        raise NotImplementedError
