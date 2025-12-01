import os

import matplotlib.pyplot as plt
import numpy as np

from ._equilibrium import _feq2
from ._vof import VolumeOfFluid
from .boundary import BoundaryCondition, FreeSurface
from .collider import Collider, ColliderSRT
from .force import Force
from .macroscopic import Macroscopic, MacroscopicStandard
from .refiller.base import Refiller
from .stencil import Stencil, StencilD2Q9
from .streamer import Streamer


class Model:
    """
    Single-phase lattice Boltzmann model, solved on a uniform rectangular grid.

    Attributes
    ----------
    stencil : Stencil
    streamer : Streamer
    collider : Collider
    macros : Macroscopic
    nx : int
        Integer number of grid cells in x direction.
    ny : int
        Integer number of grid cells in y direction.
    dx : float
        Grid spacing, in metres (square grid).
    dt : float
        Time step, in seconds.
    nu : float
        Kinematic viscosity, in m^2/s.
    clock : float
        Model time, in seconds.
    x : ndarray
        One-dimensional numpy array of floats of shape (`nx`,).
        Contains x-coordinates of cell centres, in metres.
    y : ndarray
        One-dimensional numpy array of floats of shape (`ny`,).
        Contains y-coordinates of cell centres, in metres.
    shape : tuple of ints
        Tuple of integers (`ny`, `nx`).
    u : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-component of velocity at each cell centre, in lattice units.
    v : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-component of velocity at each cell centre, in lattice units.
    r : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains fluid density at each cell centre, in lattice units.
    fi : ndarray
        Three-dimensional numpy array of floats of shape (`stencil.nq`, `ny`, `nx`).
        Contains incoming distribution functions, in lattice units.
    fo : ndarray
        Three-dimensional numpy array of floats of shape (`stencil.nq`, `ny`, `nx`).
        Contains outgoing distribution functions, in lattice units.
    Fx : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-components of body force on fluid at cell centres, in lattice units.
    Fy : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-components of body force on fluid at cell centres, in lattice units.
    uc : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-components of velocity for model collision step, in lattice units.
    vc : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-components of velocity for model collision step, in lattice units.
    boundary_conditions : list[BoundaryCondition]
        List of BoundaryCondition objects. These are all called during model `step()`.
    forcings : list[Force]
        List of Force objects. These are all called during model `step()`.
    initialised : bool
        Boolean flag that tracks whether model has been initialised.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        nu: float,
        stencil: Stencil = None,
        streamer: Streamer = None,
        collider: Collider = None,
        macros: Macroscopic = None,
        quiet=True,
    ):
        """
        Construct `Model` object.

        Parameters
        ----------
        nx : int
            Integer number of grid cells in the x direction.
        ny : int
            Integer number of grid cells in the y direction.
        dx : float
            Grid spacing between cells (grid cells are uniform squares).
        dt : float
            Time step.
        nu : float
            Kinematic viscosity of the fluid.
        stencil : Stencil, optional
            Lattice Stencil object. Default is `StencilD2Q9`
        streamer : Streamer, optional
            Streamer object, which performs the lattice Boltzmann streaming step.
            Default is `Streamer`
        collider : Collider, optional
            Collider object, which performs the lattice Boltzmann collision step.
            Default is `ColliderSRT`
        macros : Macroscopic, optional
            Macroscopic object, which contains methods for computing fluid properties.
            Default is `MacroscopicStandard`
        quiet : bool, optional
            If `False`, print progress while model runs. Otherwise, don't.
            Default is `True`

        Returns
        -------
        None
        """
        self.stencil = StencilD2Q9() if stencil is None else stencil
        self.streamer = Streamer() if streamer is None else streamer
        self.collider = ColliderSRT(nu, dx, dt) if collider is None else collider
        self.macros = MacroscopicStandard() if macros is None else macros

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
        self.Fx = np.zeros(self.shape)
        self.Fy = np.zeros(self.shape)
        self.uc = np.zeros(self.shape)
        self.vc = np.zeros(self.shape)

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

    def add_boundary_condition(self, bc: BoundaryCondition):
        """
        Add boundary condition `bc` to model.

        Parameters
        ----------
        bc : BoundaryCondition
            BoundaryCondition object which is to be added to the model.

        Returns
        -------
        None
        """
        self.boundary_conditions.append(bc)

    def add_forcing(self, force: Force):
        """
        Add force term `force` to model.

        Parameters
        ----------
        force : Force
            Force object which is to be added to model.

        Returns
        -------
        None
        """
        self.forcings.append(force)

    def set_r(self, source, *args):
        """
        Set `self.r` (fluid density) array values.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have the same shape as `data`.

        Parameters
        ----------
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        self._set(self.r, source, *args)

    def set_u(self, source, *args):
        """
        Set `self.u` (x-component of velocity) array values.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have the same shape as `data`.

        Parameters
        ----------
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        self._set(self.u, source, *args)

    def set_v(self, source, *args):
        """
        Set `self.v` (y-component of velocity) array values.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have the same shape as `data`.

        Parameters
        ----------
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        self._set(self.v, source, *args)

    def initialise(self):
        """
        Initialise model.

        Modifies `self.fi`, `self.clock`, and `self.initialised`.
        """
        self._initialise_feq2()
        self.clock = 0.0
        self.initialised = True

    def step(self):
        """
        Perform one time step of lattice Boltzmann algorithm.

        Updates `self.fi`, `self.fo`, `self.r`, `self.u`, `self.v`, 
        `self.Fx`, `self.Fy`, `self.uc`, `self.vc`, and `self.clock`.
        `Force` and `BoundaryCondition` internals may also change.
        """
        # Collision step
        self.macros.velocity_x_coll(self)
        self.macros.velocity_y_coll(self)
        self.collider.collide(self)

        # Compute force arrays.
        self.Fx.fill(0.0)
        self.Fy.fill(0.0)
        for force in self.forcings:
            force.apply(self)

        # Apply forcing source term to distribution functions
        self.macros.forcing(self)

        # Stream step
        self.streamer.stream(self)

        # Apply boundary conditions.
        for bc in self.boundary_conditions:
            bc.apply(self)

        # Compute new macroscopic variables
        self.macros.density(self)
        self.macros.velocity_x(self)
        self.macros.velocity_y(self)

        # Update time
        self.clock += self.dt

    def plot_fields(self, path=None):
        """
        Plot heatmaps of `self.r`, `self.u`, and `self.v` arrays.

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

    def _set_f(self, source, *args):
        """
        Set `self.fi` and `self.fo` (incoming and outgoing distribution) array values.

        This is a convenience method for unit tests. Should not be used otherwise.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have shape (nq, ny, nx).

        Parameters
        ----------
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        self._set(self.fi, source, *args)
        self._set(self.fo, source, *args)

    def _set(self, data, source, *args):
        """
        Set `data` array to `source` values.

        Array `data` is modified in-place.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have the same shape as `data`.

        Parameters
        ----------
        data : np.ndarray
            Array to be filled
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        if callable(source):
            X, Y = np.meshgrid(self.x, self.y)
            data[:] = source(X, Y, *args)
        else:
            assert data.shape == source.shape
            data[:] = source

    def _initialise_feq2(self):
        """Initialise `self.fi` with 2nd order equilibrium distribution."""
        self.fi = _feq2(self.r, self.u, self.v, self.stencil)


class FreeSurfaceModel(Model):
    """
    Two-phase lattice Boltzmann model, solved on a uniform rectangular grid.

    Attributes
    ----------
    stencil : Stencil
    streamer : Streamer
    collider : Collider
    macros : Macroscopic
    refiller : Refiller
    nx : int
        Integer number of grid cells in x direction.
    ny : int
        Integer number of grid cells in y direction.
    dx : float
        Grid spacing, in metres (square grid).
    dt : float
        Time step, in seconds.
    nu : float
        Kinematic viscosity, in m^2/s.
    rho_G : float
        Gas density, in lattice units.
    clock : float
        Model time, in seconds.
    x : ndarray
        One-dimensional numpy array of floats of shape (`nx`,).
        Contains x-coordinates of cell centres, in metres.
    y : ndarray
        One-dimensional numpy array of floats of shape (`ny`,).
        Contains y-coordinates of cell centres, in metres.
    shape : tuple of ints
        Tuple of integers (`ny`, `nx`).
    u : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-component of velocity at each cell centre, in lattice units.
    v : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-component of velocity at each cell centre, in lattice units.
    r : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains fluid density at each cell centre, in lattice units.
    fi : ndarray
        Three-dimensional numpy array of floats of shape (`stencil.nq`, `ny`, `nx`).
        Contains incoming distribution functions, in lattice units.
    fo : ndarray
        Three-dimensional numpy array of floats of shape (`stencil.nq`, `ny`, `nx`).
        Contains outgoing distribution functions, in lattice units.
    Fx : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-components of body force on fluid at cell centres, in lattice units.
    Fy : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-components of body force on fluid at cell centres, in lattice units.
    uc : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains x-components of velocity for model collision step, in lattice units.
    vc : ndarray
        Two-dimensional numpy array of floats of shape (`ny`, `nx`).
        Contains y-components of velocity for model collision step, in lattice units.
    boundary_conditions : list[BoundaryCondition]
        List of BoundaryCondition objects. These are all called during model `step()`.
    forcings : list[Force]
        List of Force objects. These are all called during model `step()`.
    initialised : bool
        Boolean flag that tracks whether model has been initialised.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        nu: float,
        rho_G: float = 1.0,
        stencil: Stencil = None,
        streamer: Streamer = None,
        collider: Collider = None,
        macros: Macroscopic = None,
        refiller: Refiller = None,
        quiet: bool = True,
    ):
        """
        Construct `FreeSurfaceModel` object.

        Parameters
        ----------
        nx : int
            Integer number of grid cells in the x direction.
        ny : int
            Integer number of grid cells in the y direction.
        dx : float
            Grid spacing between cells (grid cells are uniform squares).
        dt : float
            Time step.
        nu : float
            Kinematic viscosity of the fluid.
        rho_G : float, optional.
            Gas density, in lattice units. Default = 1.0
        stencil : Stencil, optional
            Lattice Stencil object. Default is `StencilD2Q9`
        streamer : Streamer, optional
            Streamer object, which performs the lattice Boltzmann streaming step.
            Default is `Streamer`
        collider : Collider, optional
            Collider object, which performs the lattice Boltzmann collision step.
            Default is `ColliderSRT`
        macros : Macroscopic, optional
            Macroscopic object, which contains methods for computing fluid properties.
            Default is `MacroscopicStandard`
        refiller : Refiller, optional
            Refiller objects, which provides the refilling scheme for `VolumeOfFluid`.
            Default is `UniformRefiller(1.0, 0.0, 0.0)`
        quiet : bool, optional
            If `False`, print progress while model runs. Otherwise, don't.
            Default is `True`

        Returns
        -------
        None
        """
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
        self.vof = VolumeOfFluid(self.shape, self.stencil, refiller=refiller)
        self.fsbc = FreeSurface(rho_G)

    def set_phi(self, source, *args):
        """
        Set `self.vof.phi` (cell fluid fraction) array values.

        - If `source` is callable, it must have signature (x, y, *args).
        - If `source` is an array, it must have the same shape as `data`.

        Parameters
        ----------
        source : callable or np.ndarray
            Callable or array to use to fill `data`
        *args : Any, optional
            Optional arguments for source if it is callable

        Returns
        -------
        None
        """
        self._set(self.vof.phi, source, *args)

    def set_phi_from_eta(self, eta_source, *args):
        """
        Set fill fraction `self.vof.phi` from surface elevation `eta_source`.

        Modifies `self.vof.phi`.

        Parameters
        ----------
        eta_source : callable or array
            Function with signature (x, *args) that generates surface elevations
            at `y = y(x) = eta(x, *args)`,
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
        """Print fluid properties summed over the entire grid."""
        print(f"sum_(y,x) density[y, x]    = {np.sum(self.r):.3f}")
        print(f"sum_(y,x) velocity_X[y, x] = {np.sum(self.u):.3f}")
        print(f"sum_(y,x) velocity_Y[y, x] = {np.sum(self.v):.3f}")
        print(f"sum_(y,x) phi[y, x]        = {np.sum(self.vof.phi):.3f}")
        print(f"sum_(y,x) M[y, x]          = {np.sum(self.vof.M):.3f}")
        print(f"# of FLUID cells           = {np.sum(self.vof.F_mask)}")
        print(f"# of INTERFACE cells       = {np.sum(self.vof.I_mask)}")
        print(f"# of GAS cells             = {np.sum(self.vof.G_mask)}")

    def print_means(self):
        """Print mean (average) fluid properties."""
        print(f"mean density[y, x]    = {np.mean(self.r):.6f}")
        print(f"mean velocity_X[y, x] = {np.mean(self.u):.6f}")
        print(f"mean velocity_Y[y, x] = {np.mean(self.v):.6f}")
        print(f"mean phi[y, x]        = {np.mean(self.vof.phi):.6f}")
        print(f"mean M[y, x]          = {np.mean(self.vof.M):.6f}")
        print(f"% FLUID cells         = {np.mean(self.vof.F_mask)*100:.4f}")
        print(f"% of INTERFACE cells  = {np.mean(self.vof.I_mask)*100:.4f}")
        print(f"% of GAS cells        = {np.mean(self.vof.G_mask)*100:.4f}")

    def initialise(self, do_mei: bool = False):
        """
        Initialise model.

        Modifies `self.fi`, `self.vof.M`, `self.vof.F_mask`, `self.vof.I_mask`,
        `self.vof.G_mask`, `self.clock`, and `self.initialised`

        Parameters
        ----------
        do_mei : bool, optional
            If `True`, do Mei's iterative initialisation method (untested).
            Default is `False`

        Returns
        -------
        None
        """
        self._initialise_feq2()
        self.vof.initialise(self)

        # Mei's method: iterate model step at fixed velocity until `self.fi` converges.

        fi_old = np.empty_like(self.fi)
        number_of_iterations = 0

        while do_mei and not np.allclose(self.fi, fi_old, atol=1.0e-12):
            number_of_iterations += 1
            fi_old[:] = self.fi

            # Collision step
            self.macros.velocity_x_coll(self)
            self.macros.velocity_y_coll(self)
            self.collider.collide(self)

            # Compute force arrays.
            self.Fx.fill(0.0)
            self.Fy.fill(0.0)
            for force in self.forcings:
                force.apply(self)

            # Apply forcing source term to distribution functions
            self.macros.forcing(self)

            # Stream step
            self.streamer.stream(self)

            # Apply boundary conditions.
            self.fsbc.apply(self)
            for bc in self.boundary_conditions:
                bc.apply(self)

            # Compute new density, but not velocity.
            self.macros.density(self)

            # Do not update free surface!
            self.vof.M = self.vof.phi * self.r

        self.clock = 0.0
        self.initialised = True
        return number_of_iterations

    def step(self):
        """
        Perform one time step of the two-phase lattice Boltzmann algorithm.

        Updates `self.fi`, `self.fo`, `self.r`, `self.u`, `self.v`, 
        `self.vof.phi`, `self.vof.M`, `self.vof.F_mask`, `self.vof.I_mask`,
        `self.vof.G_mask`, `self.Fx`, `self.Fy`, `self.uc`, `self.vc` and `self.clock`.
        `Force` and `BoundaryCondition` internals may also change.
        """
        # Collision step
        self.macros.velocity_x_coll(self)
        self.macros.velocity_y_coll(self)
        self.collider.collide(self)

        # Compute force arrays.
        self.Fx.fill(0.0)
        self.Fy.fill(0.0)
        for force in self.forcings:
            force.apply(self)

        # Apply forcing source term to distribution functions
        self.macros.forcing(self)

        # Stream step
        self.streamer.stream(self)

        # Apply boundary conditions.
        self.fsbc.apply(self)
        for bc in self.boundary_conditions:
            bc.apply(self)

        # Compute new macroscopic variables
        self.macros.density(self)
        self.macros.velocity_x(self)
        self.macros.velocity_y(self)

        # Update free surface
        self.vof.update(self)

        # Update time
        self.clock += self.dt

    def plot_fields(self, path=None):
        """
        Plot heatmaps of `self.r`, `self.u`, `self.v`, and `self.phi` arrays.

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

    def _phi_from_eta(self, eta_array, interface_width=0.9):
        """
        Compute fill fraction array `phi_array` from elevation values in `eta_array`

        Parameters
        ----------
        eta_array : np.ndarray
            One-dimensional numpy array of floats of shape (nx,).
            Contains surface elevation at each grid column.
            Assumes eta is continuous and injective (one-one).
        interface_width : float
            Half-width of interface, in lattice units.

        Returns
        -------
        phi_array : np.ndarray
            Two dimensional numpy array of floats of shape (ny, nx).
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
