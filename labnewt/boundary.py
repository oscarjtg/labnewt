"""
Boundary condition classes.
These are passed into Model through the add_boundary_condition() method.
"""

import numpy as np

from ._equilibrium import _feq2_q
from ._shift import periodic_shift


class BoundaryCondition:
    def apply(self, model):
        return NotImplementedError


class NoSlip(BoundaryCondition):
    def bounce_back(self, fi, fo, qi, qo, x, y):
        """
        Applies bounce back lattice Boltzmann boundary rule.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Contains incoming distrbution functions. Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Contains outgoing distribution functions. Not modified.
        qi : int
            Lattice index of incoming particles.
        qo : int
            Lattice index of outgoing particles.
        x : int
            Spatial index of x-coordinate of grid cell.
        y : int
            Spatial index of y-coordinate of grid cell.

        Returns
        -------
        None
        """
        fi[qi, y, x] = fo[qo, y, x]


class LeftWallNoSlip(NoSlip):
    def apply(self, model):
        """
        Applies no slip BC to left wall, which is stationary.

        Modifies `model.fi` in place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        qi = model.stencil.q_right
        qo = model.stencil.q_rev[qi]
        x = 0
        y = slice(None)
        self.bounce_back(model.fi, model.fo, qi, qo, x, y)


class RightWallNoSlip(NoSlip):
    def apply(self, model):
        """
        Applies no slip BC to right wall, which is stationary.

        Modifies `model.fi` in place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        qi = model.stencil.q_left
        qo = model.stencil.q_rev[qi]
        x = -1
        y = slice(None)
        self.bounce_back(model.fi, model.fo, qi, qo, x, y)


class BottomWallNoSlip(NoSlip):
    def apply(self, model):
        """
        Applies no slip BC to bottom wall, which is stationary.

        Modifies `model.fi` in place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        qi = model.stencil.q_up
        qo = model.stencil.q_rev[qi]
        x = slice(None)
        y = 0
        self.bounce_back(model.fi, model.fo, qi, qo, x, y)


class TopWallNoSlip(NoSlip):
    def apply(self, model):
        """
        Applies no slip BC to top wall, which is stationary.

        Modifies `model.fi` in place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        qi = model.stencil.q_down
        qo = model.stencil.q_rev[qi]
        x = slice(None)
        y = -1
        self.bounce_back(model.fi, model.fo, qi, qo, x, y)


class FreeSurface:
    def __init__(self, rho_G=1.0):
        self.rho_G = rho_G

    def apply(self, model):
        """
        Applies free surface boundary condition at interface cells.

        Modifies `model.fi` in place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel

        Returns
        -------
        None
        """
        for q in range(model.stencil.nq):
            self._free_surface_boundary_condition_q(
                q, model.fi, model.fo, model.stencil, model.u, model.v, model.vof.I_mask, model.vof.G_mask, self.rho_G
            )

    def _free_surface_boundary_condition_q(
        self, q, fi, fo, s, u, v, I_mask, G_mask, rho_G
    ):
        """
        Apply fluid-gas boundary condition in-place.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        q : int
            Lattice vector index
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Contains incoming distrbution functions. Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Contains outgoing distribution functions. Not modified.
        s : stencil
            Lattice stencil object.
        u : np.ndarray
            Two-dimensional numpy array of floats of shape (ny, nx).
            Contains velocity x-components.
        v : np.ndarray
            Two-dimensional numpy array of floats of shape (ny, nx).
            Contains velocity y-components.
        I_mask : np.ndarray
            Two-dimensional numpy array of booleans of shape (ny, nx).
            Contains `True` at INTERFACE cells, otherwise `False`.
        G_mask : np.ndarray
            Two-dimensional numpy array of booleans of shape (ny, nx).
            Contains `True` at GAS cells, otherwise `False`.
        rho_G : float
            Gas density, in lattice units. Default = 1.0

        Returns
        -------
        None
        """
        mask_gas_shifted = periodic_shift(G_mask, s, q)
        f_out_qrev = np.copy(fo[s.q_rev[q], :, :])
        feq_q = _feq2_q(q, rho_G, u, v, s)
        feq_qrev = _feq2_q(s.q_rev[q], rho_G, u, v, s)

        mask = I_mask * mask_gas_shifted
        fi[q, mask] = feq_q[mask] + feq_qrev[mask] - f_out_qrev[mask]
