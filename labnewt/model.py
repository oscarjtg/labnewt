import os

import matplotlib.pyplot as plt
import numpy as np

from .collider import ColliderSRT
from .stencil import StencilD2Q9
from .streamer import Streamer


class Model:
    def __init__(self, nx, ny, dx, dt, nu, stencil=None, streamer=None, collider=None):
        self.stencil = StencilD2Q9() if stencil is None else stencil
        self.streamer = Streamer() if streamer is None else streamer
        self.collider = ColliderSRT(nu, dx, dt) if collider is None else collider

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.nu = nu

        self.x = np.linspace(0.5 * dx, (nx - 0.5) * dx, nx)
        self.y = np.linspace(0.5 * dx, (ny - 0.5) * dx, ny)
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.r = np.ones((nx, ny))
        self.f = np.zeros((self.stencil.nq, nx, ny))

    def _set(self, data, source, *args):
        """
        Sets data to values in source.
        If source is callable (i.e. a function),
        assume it has signature (x, y, *args)
        and fill data with source(x, y, *args)
        for coordinates (x, y) in self.x and self.y.
        """
        if callable(source):
            X, Y = np.meshgrid(self.x, self.y)
            data[:] = source(X, Y, *args)
        else:
            assert data.shape == source.shape
            data[:] = source

    def set_u(self, source, *args):
        """
        Set x-component of velocity values, u.
        If source is an array, set u to array values.
        If source is a function with signature (x, y, *args),
        set u to values evaluated by the function.
        """
        self._set(self.u, source, *args)

    def set_v(self, source, *args):
        """
        Set y-component of velocity values, v.
        If source is an array, set v to array values.
        If source is a function with signature (x, y, *args),
        set v to values evaluated by the function.
        """
        self._set(self.v, source, *args)

    def set_r(self, source, *args):
        """
        Set density values, r.
        If source is an array, set r to array values.
        If source is a function with signature (x, y, *args),
        set r to values evaluated by the function.
        """
        self._set(self.r, source, *args)

    def plot_fields(self, path=None):
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        X, Y = np.meshgrid(self.x, self.y)

        p0 = ax[0].pcolormesh(X, Y, self.r)
        p1 = ax[1].pcolormesh(X, Y, self.u)
        p2 = ax[2].pcolormesh(X, Y, self.v)

        cbar0 = plt.colorbar(p0, ax=ax[0])
        cbar1 = plt.colorbar(p1, ax=ax[1])
        cbar2 = plt.colorbar(p2, ax=ax[2])

        cbar0.set_label(r"$\rho$", fontsize=14)
        cbar1.set_label(r"$u$", fontsize=14)
        cbar2.set_label(r"$v$", fontsize=14)

        cbar0.ax.tick_params(labelsize=13)
        cbar1.ax.tick_params(labelsize=13)
        cbar2.ax.tick_params(labelsize=13)

        ax[0].tick_params(labelsize=13)
        ax[1].tick_params(labelsize=13)
        ax[2].tick_params(labelsize=13)

        ax[0].set_ylabel(r"$y$", fontsize=14)
        ax[1].set_ylabel(r"$y$", fontsize=14)
        ax[2].set_ylabel(r"$y$", fontsize=14)

        ax[2].set_xlabel(r"$x$", fontsize=14)

        plt.tight_layout()
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
