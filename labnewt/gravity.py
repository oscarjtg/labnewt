"""
Gravity class:
stores gravity components and allows setting magnitude and direction
"""

import numpy as np


class Gravity:
    def __init__(self, gx=0.0, gy=9.81):
        self.gx = gx
        self.gy = gy

    def set_gravity(self, magnitude=None, direction=None):
        """
        Sets gravity vectors gx and gy.

        If only a magnitude is given,
        the existing gravity vector components `self.gx` and `self.gy`
        are scaled by `magnitude`.

        If only a direction `dir` tuple is given,
        the magnitude is preserved but the direction changes.

        If both are given, new gravity vector components
        are calculated in the given direction with the
        given vector magnitude.

        Parameters
        ----------
        magnitude : float, optional
            Float giving the desired magnitude of gravity.
        direction : tuple, optional
            Tuple of floats giving the direction to set gravity.

        Returns
        -------
        None
        """
        mag = np.hypot(self.gx, self.gy)
        if magnitude:
            mag = magnitude
        dir = (self.gx, self.gy)
        if direction:
            assert len(direction) == 2
            dir = direction
        self.gx, self.gy = mag * dir / np.hypot(*dir)
