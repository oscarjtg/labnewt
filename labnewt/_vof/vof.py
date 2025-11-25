import os

import matplotlib.pyplot as plt
import numpy as np

from ._distribute_Mex import _distribute_Mex
from ._dMq import _dMq_
from ._Mstar import _Mstar_inplace
from ._normals import _normals_


class VolumeOfFluid:
    def __init__(self, shape, stencil):
        self.phi = np.zeros(shape, dtype=np.float64)
        self.M = np.zeros(shape, dtype=np.float64)
        self.norm_x = np.zeros(shape, dtype=np.float64)
        self.norm_y = np.zeros(shape, dtype=np.float64)
        self.F_mask = np.zeros(shape, dtype=np.bool_)
        self.I_mask = np.zeros(shape, dtype=np.bool_)
        self.G_mask = np.zeros(shape, dtype=np.bool_)
        self.dMq = np.zeros((stencil.nq, *shape), dtype=np.float64)
        self.x = np.arange(shape[1])
        self.y = np.arange(shape[0])

    def initialise(self, model):
        self._initialise(model.r)

    def update(self, model):
        self._step(model.fo, model.r, model.stencil)

    def _initialise(self, rho):
        self.M = self.phi * rho
        self._update_state(rho)

    def _set_masks(self, eps=1.0e-05):
        np.less_equal(self.phi, eps, out=self.G_mask)
        np.greater_equal(self.phi, 1.0 - eps, out=self.F_mask)
        np.logical_not(np.logical_or(self.G_mask, self.F_mask), out=self.I_mask)

    def _update_state(self, rho):
        # Update cell fill fraction.
        self.phi = self.M / rho

        # Update masks
        self._set_masks()

        # Update unit normal vectors.
        _normals_(self.norm_x, self.norm_y, self.phi, self.I_mask)

    def _step(self, fo, rho, stencil):
        # Compute mass exchange in each direction.
        _dMq_(self.dMq, fo, self.phi, self.F_mask, self.I_mask, stencil)

        # Add the exchanged masses to self.M.
        _Mstar_inplace(self.M, self.dMq)

        # Identify INTERFACE cells with excess mass and transfer to neighbours.
        def within_bounds(phi, eps=1.0e-05):
            return np.all((-eps <= phi) & (phi <= 1.0 + eps))

        counter = 0
        while not within_bounds(self.M / rho):
            self.M = _distribute_Mex(
                self.M, rho, self.norm_x, self.norm_y, self.I_mask, stencil
            )
            counter += 1
            if counter > 5:
                # print("Warning: _distribute_Mex iterations did not converge.")
                # print(f"M_min = {self.M.min()}")
                # print(f"M_max = {self.M.max()}")
                break

        # Update phi, masks, and normals.
        self._update_state(rho)

    def plot_fields(self, path=None):
        """
        Plots heatmaps of `self.r`, `self.u`, `self.v`, and `self.phi` arrays.
        Saves plot if a `path` is given.
        The plot can be displayed by calling `plt.show()`.

        Parameters
        ----------
        path : str or Path-like, optional
            File path to save the plot. If None (default), the plot is
            not saved.

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        X, Y = np.meshgrid(self.x, self.y)

        ax0 = ax[0][0]
        ax1 = ax[1][0]
        ax2 = ax[1][1]
        ax3 = ax[0][1]

        p0 = ax0.pcolormesh(X, Y, self.phi)
        p1 = ax1.pcolormesh(X, Y, self.F_mask)
        p2 = ax2.pcolormesh(X, Y, self.I_mask)
        p3 = ax3.pcolormesh(X, Y, self.G_mask)

        cbar0 = plt.colorbar(p0, ax=ax0)
        cbar1 = plt.colorbar(p1, ax=ax1)
        cbar2 = plt.colorbar(p2, ax=ax2)
        cbar3 = plt.colorbar(p3, ax=ax3)

        cbar0.set_label(r"$\phi$", fontsize=14)
        cbar1.set_label("FLUID", fontsize=14)
        cbar2.set_label("INTERFACE", fontsize=14)
        cbar3.set_label("GAS", fontsize=14)

        cbar0.ax.tick_params(labelsize=13)
        cbar1.ax.tick_params(labelsize=13)
        cbar2.ax.tick_params(labelsize=13)
        cbar3.ax.tick_params(labelsize=13)

        ax0.tick_params(labelsize=13)
        ax1.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        ax3.tick_params(labelsize=13)

        ax[0][0].set_ylabel(r"$y$", fontsize=14)
        ax[1][0].set_ylabel(r"$y$", fontsize=14)

        ax[1][0].set_xlabel(r"$x$", fontsize=14)
        ax[1][1].set_xlabel(r"$x$", fontsize=14)

        plt.tight_layout()
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close()
