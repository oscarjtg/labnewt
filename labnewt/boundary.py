"""
Boundary condition classes.
These are passed into Model through the add_boundary_condition() method.
"""

import numpy as np

from ._equilibrium import _feq2_q
from ._shift import periodic_shift


class NoSlip:
    def bounce_back(self, fi, fo, qi, qo, x, y):
        """
        Applies bounce back lattice Boltzmann boundary rule.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
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
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to left wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.

        Returns
        -------
        None
        """
        qi = s.q_right
        qo = s.q_rev[qi]
        x = 0
        y = slice(None)
        self.bounce_back(fi, fo, qi, qo, x, y)


class RightWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to right wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.

        Returns
        -------
        None
        """
        qi = s.q_left
        qo = s.q_rev[qi]
        x = -1
        y = slice(None)
        self.bounce_back(fi, fo, qi, qo, x, y)


class BottomWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to bottom wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.

        Returns
        -------
        None
        """
        qi = s.q_up
        qo = s.q_rev[qi]
        x = slice(None)
        y = 0
        self.bounce_back(fi, fo, qi, qo, x, y)


class TopWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to top wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.

        Returns
        -------
        None
        """
        qi = s.q_down
        qo = s.q_rev[qi]
        x = slice(None)
        y = -1
        self.bounce_back(fi, fo, qi, qo, x, y)


class FreeSurface:
    def __init__(self, rho_G=1.0):
        self.rho_G = rho_G

    def apply(self, fi, fo, s, u, v, I_mask, G_mask):
        """
        Applies free surface boundary condition at interface cells.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        u : np.ndarray
            Two-dimensional numpy array of floats containing x-component of velocity.
            u.shape = (ny, nx)
        v : np.ndarray
            Two-dimensional numpy array of floats containing y-component of velocity.
            v.shape = (ny, nx)
        I_mask : np.ndarray
            Two-dimensional numpy array of booleans. shape = (ny, nx)
        G_mask : np.ndarray
            Two-dimensional numpy array of booleans. shape = (ny, nx)

        Returns
        -------
        None
        """
        for q in range(s.nq):
            self._free_surface_boundary_condition_q(
                q, fi, fo, s, u, v, I_mask, G_mask, self.rho_G
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
        f : np.ndarray
            Three dimensional numpy array of floats containing distribution functions.
            f.shape = (nq, ny, nx)
        s : stencil
            Lattice stencil
        u : np.ndarray
            Two-dimensional numpy array of floats containing x-component of velocity.
            u.shape = (ny, nx)
        v : np.ndarray
            Two-dimensional numpy array of floats containing y-component of velocity.
            v.shape = (ny, nx)
        I_mask : np.ndarray
            Two-dimensional numpy array of booleans. shape = (ny, nx)
        G_mask : np.ndarray
            Two-dimensional numpy array of booleans. shape = (ny, nx)
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
