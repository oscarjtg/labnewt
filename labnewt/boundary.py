"""
Boundary condition classes.
These are passed into Model through the add_boundary_condition() method.
"""

import numpy as np


class LeftWallNoSlip:
    def __init__(self):
        pass

    def apply(self, f, s):
        """
        Applies no slip BC to left wall, which is stationary.

        Parameters
        ----------
        f : np.ndarray
            three-dimensional numpy array containing particle distribution

        s : Stencil
            instance of Stencil
        """
        f2_out = f[2, :, -1]

        f6_out = np.empty_like(f2_out)
        f6_out[0] = f[6, -1, -1]
        f6_out[1:] = f[6, :-1, -1]

        f7_out = np.empty_like(f2_out)
        f7_out[:-1] = f[7, 1:, -1]
        f7_out[-1] = f[7, 0, -1]

        f[1, :, 0] = f2_out
        f[5, :, 0] = f6_out
        f[8, :, 0] = f7_out


class RightWallNoSlip:
    def __init__(self):
        pass

    def apply(self, f, s):
        """
        Applies no slip BC to right wall, which is stationary.

        Parameters
        ----------
        f : np.ndarray
            three-dimensional numpy array containing particle distribution

        s : Stencil
            instance of Stencil
        """
        f1_out = f[1, :, 0]

        f5_out = np.empty_like(f1_out)
        f5_out[:-1] = f[5, 1:, 0]
        f5_out[-1] = f[5, 0, 0]

        f8_out = np.empty_like(f1_out)
        f8_out[0] = f[8, -1, 0]
        f8_out[1:] = f[8, :-1, 0]

        f[2, :, -1] = f1_out
        f[6, :, -1] = f5_out
        f[7, :, -1] = f8_out
