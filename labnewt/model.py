import os

import matplotlib.pyplot as plt
import numpy as np

from ._equilibrium import _feq2
from ._vof import VolumeOfFluid
from .boundary import FreeSurface
from .collider import ColliderSRT
from .force import Force
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
        self.clock = 0.0

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
        self.initialised = False

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

    def _set_f(self, source, *args):
        """
        Sets fi and fo. For unit tests only. Should not be used.
        """
        self._set(self.fi, source, *args)
        self._set(self.fo, source, *args)

    def _step(self):
        """Perform one time step of lattice Boltzmann algorithm."""
        # Collision step
        self.collider.collide(self.fo, self.fi, self.r, self.u, self.v, self.stencil)

        # Apply forcing terms
        for force in self.forcings:
            force.apply(self)

        # Stream step
        self.streamer.stream(self.fi, self.fo, self.stencil)

        # Apply boundary conditions.
        for bc in self.boundary_conditions:
            bc.apply(self.fi, self.fo, self.stencil)

        # Compute new macroscopic variables
        self.macros.density(self.r, self.fi)
        self.macros.velocity_x(self.u, self.r, self.fi, self.stencil)
        self.macros.velocity_y(self.v, self.r, self.fi, self.stencil)

        # Update time
        self.clock += self.dt

    def _initialise_feq2(self):
        """Initialise self.f with 2nd order equilibrium distribution."""
        self.fi = _feq2(self.r, self.u, self.v, self.stencil)

    def _initialise(self):
        """Initialise model."""
        self._initialise_feq2()
        self.clock = 0.0
        self.initialised = True

    def add_boundary_condition(self, bc):
        """Adds bc to self.boundary_conditions list."""
        self.boundary_conditions.append(bc)

    def add_forcing(self, force: Force):
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

        plt.suptitle(f"time = {self.clock:.3f} s")

        plt.tight_layout()
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)


class FreeSurfaceModel(Model):
    def __init__(
        self,
        nx,
        ny,
        dx,
        dt,
        nu,
        rho_G=1.0,
        stencil=None,
        streamer=None,
        collider=None,
        macros=None,
        quiet=True,
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
            macros=macros,
            quiet=quiet,
        )
        self.vof = VolumeOfFluid(self.shape, self.stencil)
        self.fsbc = FreeSurface(rho_G)

    def set_phi(self, source, *args):
        """
        Set fluid fraction values, phi.

        If source is an array, set phi to array values.
        If source is a function with signature (x, y, *args),
        set phi to values evaluated by the function.
        """
        self._set(self.vof.phi, source, *args)

    def _phi_from_eta(self, eta_array, interface_width=0.9):
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
        self.vof.phi = self._phi_from_eta(eta)

    def print_integrals(self):
        print(f"sum_(y,x) density[y, x]    = {np.sum(self.r):.3f}")
        print(f"sum_(y,x) velocity_X[y, x] = {np.sum(self.u):.3f}")
        print(f"sum_(y,x) velocity_Y[y, x] = {np.sum(self.v):.3f}")
        print(f"sum_(y,x) phi[y, x]        = {np.sum(self.vof.phi):.3f}")
        print(f"sum_(y,x) M[y, x]          = {np.sum(self.vof.M):.3f}")
        print(f"# of FLUID cells           = {np.sum(self.vof.F_mask)}")
        print(f"# of INTERFACE cells       = {np.sum(self.vof.I_mask)}")
        print(f"# of GAS cells             = {np.sum(self.vof.G_mask)}")

    def print_means(self):
        print(f"mean density[y, x]    = {np.mean(self.r):.6f}")
        print(f"mean velocity_X[y, x] = {np.mean(self.u):.6f}")
        print(f"mean velocity_Y[y, x] = {np.mean(self.v):.6f}")
        print(f"mean phi[y, x]        = {np.mean(self.vof.phi):.6f}")
        print(f"mean M[y, x]          = {np.mean(self.vof.M):.6f}")
        print(f"% FLUID cells         = {np.mean(self.vof.F_mask)*100:.4f}")
        print(f"% of INTERFACE cells  = {np.mean(self.vof.I_mask)*100:.4f}")
        print(f"% of GAS cells        = {np.mean(self.vof.G_mask)*100:.4f}")

    def _initialise(self, do_mei=False):
        """Initialise model."""
        self._initialise_feq2()
        self.vof.initialise(self.r)

        # Mei's method: timestep but without evolving velocity
        # until fi stabilises.

        fi_old = np.empty_like(self.fi)
        number_of_iterations = 0

        while do_mei and not np.allclose(self.fi, fi_old, atol=1.0e-12):
            number_of_iterations += 1
            fi_old[:] = self.fi

            # Collision step
            self.collider.collide(
                self.fo, self.fi, self.r, self.u, self.v, self.stencil
            )

            # Apply forcing terms
            for force in self.forcings:
                force.apply(self)

            # Stream step
            self.streamer.stream(self.fi, self.fo, self.stencil)

            # Apply boundary conditions.
            self.fsbc.apply(
                self.fi,
                self.fo,
                self.stencil,
                self.u,
                self.v,
                self.vof.I_mask,
                self.vof.G_mask,
            )
            for bc in self.boundary_conditions:
                bc.apply(self.fi, self.fo, self.stencil)

            # Compute new density, but not velocity.
            self.macros.density(self.r, self.fi)

            # Do not update free surface!
            self.vof.M = self.vof.phi * self.r

        self.clock = 0.0
        self.initialised = True
        return number_of_iterations

    def _step(self):
        """Perform one time step of lattice Boltzmann algorithm."""
        # Collision step
        self.collider.collide(self.fo, self.fi, self.r, self.u, self.v, self.stencil)

        # Apply forcing terms
        for force in self.forcings:
            force.apply(self)

        # Stream step
        self.streamer.stream(self.fi, self.fo, self.stencil)

        # Apply boundary conditions.
        self.fsbc.apply(
            self.fi,
            self.fo,
            self.stencil,
            self.u,
            self.v,
            self.vof.I_mask,
            self.vof.G_mask,
        )
        for bc in self.boundary_conditions:
            bc.apply(self.fi, self.fo, self.stencil)

        # Compute new macroscopic variables
        self.macros.density(self.r, self.fi)
        self.macros.velocity_x(self.u, self.r, self.fi, self.stencil)
        self.macros.velocity_y(self.v, self.r, self.fi, self.stencil)

        # Update free surface
        self.vof.step(self.fo, self.r, self.stencil)

        # Update time
        self.clock += self.dt

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
        fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        X, Y = np.meshgrid(self.x, self.y)

        ax0 = ax[0][0]
        ax1 = ax[1][0]
        ax2 = ax[1][1]
        ax3 = ax[0][1]
        ax4 = ax[0][2]
        ax5 = ax[1][2]
        ax6 = ax[0][3]
        ax7 = ax[1][3]

        p0 = ax0.pcolormesh(X, Y, self.r)
        p1 = ax1.pcolormesh(X, Y, self.u)
        p2 = ax2.pcolormesh(X, Y, self.v)
        p3 = ax3.pcolormesh(X, Y, self.vof.M)
        p4 = ax4.pcolormesh(X, Y, self.vof.phi)
        p5 = ax5.pcolormesh(X, Y, self.vof.F_mask)
        p6 = ax6.pcolormesh(X, Y, self.vof.I_mask)
        p7 = ax7.pcolormesh(X, Y, self.vof.G_mask)

        cbar0 = plt.colorbar(p0, ax=ax0)
        cbar1 = plt.colorbar(p1, ax=ax1)
        cbar2 = plt.colorbar(p2, ax=ax2)
        cbar3 = plt.colorbar(p3, ax=ax3)
        cbar4 = plt.colorbar(p4, ax=ax4)
        cbar5 = plt.colorbar(p5, ax=ax5)
        cbar6 = plt.colorbar(p6, ax=ax6)
        cbar7 = plt.colorbar(p7, ax=ax7)

        cbar0.set_label(r"$\rho$", fontsize=14)
        cbar1.set_label(r"$u$", fontsize=14)
        cbar2.set_label(r"$v$", fontsize=14)
        cbar3.set_label(r"$M$", fontsize=14)
        cbar4.set_label(r"$\phi$", fontsize=14)
        cbar5.set_label("FLUID", fontsize=14)
        cbar6.set_label("INTERFACE", fontsize=14)
        cbar7.set_label("GAS", fontsize=14)

        ax0.set_title(r"$\rho$", fontsize=16)
        ax1.set_title(r"$u$", fontsize=16)
        ax2.set_title(r"$v$", fontsize=16)
        ax3.set_title(r"$M$", fontsize=16)
        ax4.set_title(r"$\phi$", fontsize=16)
        ax5.set_title("FLUID", fontsize=16)
        ax6.set_title("INTERFACE", fontsize=16)
        ax7.set_title("GAS", fontsize=16)

        cbar0.ax.tick_params(labelsize=13)
        cbar1.ax.tick_params(labelsize=13)
        cbar2.ax.tick_params(labelsize=13)
        cbar3.ax.tick_params(labelsize=13)
        cbar4.ax.tick_params(labelsize=13)
        cbar5.ax.tick_params(labelsize=13)
        cbar6.ax.tick_params(labelsize=13)
        cbar7.ax.tick_params(labelsize=13)

        ax0.tick_params(labelsize=13)
        ax1.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        ax3.tick_params(labelsize=13)
        ax4.tick_params(labelsize=13)
        ax5.tick_params(labelsize=13)
        ax6.tick_params(labelsize=13)
        ax7.tick_params(labelsize=13)

        ax[0][0].set_ylabel(r"$y$", fontsize=14)
        ax[1][0].set_ylabel(r"$y$", fontsize=14)

        ax[1][0].set_xlabel(r"$x$", fontsize=14)
        ax[1][1].set_xlabel(r"$x$", fontsize=14)
        ax[1][2].set_xlabel(r"$x$", fontsize=14)
        ax[1][3].set_xlabel(r"$x$", fontsize=14)

        plt.suptitle(f"time = {self.clock:.3f} s")

        plt.tight_layout()
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close()
