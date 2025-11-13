import os

import matplotlib.pyplot as plt
import numpy as np

from ._equilibrium import _feq2
from .collider import ColliderSRT
from .macroscopic import Macroscopic
from .stencil import StencilD2Q9
from .streamer import Streamer


class Model:
    def __init__(
        self,
        nx,
        ny,
        dx,
        dt,
        nu,
        stencil=None,
        streamer=None,
        collider=None,
        macros=None,
        quiet=True,
    ):
        self.stencil = StencilD2Q9() if stencil is None else stencil
        self.streamer = Streamer() if streamer is None else streamer
        self.collider = ColliderSRT(nu, dx, dt) if collider is None else collider
        self.macros = Macroscopic() if macros is None else macros

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.nu = nu

        self.x = np.linspace(0.5 * dx, (nx - 0.5) * dx, nx)
        self.y = np.linspace(0.5 * dx, (ny - 0.5) * dx, ny)
        self.shape = (ny, nx)
        self.u = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.r = np.ones(self.shape)
        self.fi = np.zeros((self.stencil.nq, *self.shape))
        self.fo = np.zeros_like(self.fi)

        self.boundary_conditions = []
        self.forcings = []

        if not quiet:
            print("Model instance created with:")
            print(f"nx       = {self.nx}")
            print(f"ny       = {self.ny}")
            print(f"dx       = {self.dx}")
            print(f"dt       = {self.dt}")
            print(f"nu       = {self.nu}")
            print(f"tau_star = {1 / self.collider.omega:.3f}")

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

    def _step(self):
        """Perform one time step of lattice Boltzmann algorithm."""
        # Collision step
        self.collider.collide(self.fo, self.fi, self.r, self.u, self.v, self.stencil)

        # Apply forcing terms
        for force in self.forcings:
            force.apply(self.fo, self.stencil, self.macros)

        # Stream step
        self.streamer.stream(self.fi, self.fo, self.stencil)

        # Apply boundary conditions.
        for bc in self.boundary_conditions:
            bc.apply(self.fi, self.fo, self.stencil)

        # Compute new macroscopic variables
        self.macros.density(self.r, self.fi)
        self.macros.velocity_x(self.u, self.r, self.fi, self.stencil)
        self.macros.velocity_y(self.v, self.r, self.fi, self.stencil)

    def _initialise_feq2(self):
        """Initialise self.f with 2nd order equilibrium distribution."""
        self.fi = _feq2(self.r, self.u, self.v, self.stencil)

    def _initialise(self):
        """Initialise model."""
        self._initialise_feq2()

    def add_boundary_condition(self, bc):
        """Adds bc to self.boundary_conditions list."""
        self.boundary_conditions.append(bc)

    def add_forcing(self, force):
        """Adds force to self.forcings list."""
        self.forcings.append(force)

    def plot_fields(self, path=None):
        """
        Plots heatmaps of `self.r`, `self.u`, and `self.v` arrays.
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


class FreeSurfaceModel(Model):
    def __init__(
        self, nx, ny, dx, dt, nu, stencil=None, streamer=None, collider=None, quiet=True
    ):
        super().__init__(
            nx,
            ny,
            dx,
            dt,
            nu,
            stencil=stencil,
            streamer=streamer,
            collider=collider,
            quiet=quiet,
        )
        self.phi = np.ones(self.shape)

    def set_phi(self, source, *args):
        """
        Set fluid fraction values, phi.

        If source is an array, set phi to array values.
        If source is a function with signature (x, y, *args),
        set phi to values evaluated by the function.
        """
        self._set(self.phi, source, *args)

    def _phi_from_eta(self, eta_array, interface_width=1):
        """
        Set fill fraction `self.phi` from elevation values in `eta_array`.

        Parameters
        ----------
        eta_array : np.ndarray
            One-dimensional numpy array of floats containing eta[x].
        interface_width : float
            Interface half-width, in lattice units.

        Returns
        -------
        phi_array : np.ndarray
            Two dimensional numpy array of floats containing phi[y, x].
        """
        delta = interface_width * self.dx
        X, Y = np.meshgrid(self.x, self.y)
        eta_array_2d = eta_array[None, :] * np.ones(Y.shape)
        phi_array = np.empty_like(X)
        mask_fluid = Y <= eta_array[None, :] - delta
        mask_air = Y >= eta_array[None, :] + delta
        mask_interface = ~mask_fluid * ~mask_air
        phi_array[mask_fluid] = 1.0
        phi_array[mask_air] = 0.0
        phi_array[mask_interface] = (
            eta_array_2d[mask_interface] - Y[mask_interface] + delta
        ) / (2 * delta)
        return phi_array

    def set_phi_from_eta(self, eta_source, *args):
        """
        Set fill fraction `self.phi` from surface elevation `eta_source`.

        Modifies `self.phi`.

        Parameters
        ----------
        eta_source : callable or array
            Function with signature (x, *args) that generates surface elevations
            at y = y(x) = eta(x, *args),
            or an array with these values pre-computed.

        *args : any
            Optional arguments for `eta_source` if it is callable.

        Returns
        -------
        None
        """
        eta = np.empty_like(self.x)
        if callable(eta_source):
            eta[:] = eta_source(self.x, *args)
        else:
            assert eta_source.shape == self.x.shape
            eta[:] = eta_source
        self.phi = self._phi_from_eta(eta)
