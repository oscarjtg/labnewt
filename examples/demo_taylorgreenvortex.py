"""Taylor Green Vortex flow in two dimensions"""

import numpy as np

from labnewt import Model, Simulation


def tgv_ux(x, y, t, kx, ky, td, u0):
    """
    Computes x-component of velocity for 2D Taylor Green Vortex.

        ux = -u0 * sqrt(ky/kx) * cos(kx*x) * sin(ky*y) * exp(-t/td)

    Parameters
    ----------
    x : float
        x-coordinate

    y : float
        y-coordinate

    t : float
        time

    kx : float
        x-component of wave vector

    ky : float
        y-component of wave vector

    td : float
        decay timescale

    u0 : float
        initial amplitude of u

    Returns
    ux : float
        x-component of velocity of Taylor Green Vortex at position (x, y) at time t.
    """
    return -u0 * np.sqrt(ky / kx) * np.cos(kx * x) * np.sin(ky * y) * np.exp(-t / td)


def tgv_uy(x, y, t, kx, ky, td, u0):
    """
    Computes y-component of velocity for 2D Taylor Green Vortex.

        uy = u0 * sqrt(kx/ky) * sin(kx*x) * cos(ky*y) * exp(-t/td)

    Parameters
    ----------
    x : float
        x-coordinate

    y : float
        y-coordinate

    t : float
        time

    kx : float
        x-component of wave vector

    ky : float
        y-component of wave vector

    td : float
        decay timescale

    u0 : float
        initial amplitude of u

    Returns
    uy : float
        y-component of velocity of Taylor Green Vortex at position (x, y) at time t.
    """
    return u0 * np.sqrt(kx / ky) * np.sin(kx * x) * np.cos(ky * y) * np.exp(-t / td)


def tgv_p(x, y, t, kx, ky, td, u0, p0, rho0):
    """
    Computes pressure for 2D Taylor Green Vortex.

        p = p0 - (1/4) * rho0 * u0^2 * (
            ky/kx * cos(2*kx*x) + kx/ky * cos(2*ky*y)
        ) * exp(-2*t/td)

    Parameters
    ----------
    x : float
        x-coordinate

    y : float
        y-coordinate

    t : float
        time

    kx : float
        x-component of wave vector

    ky : float
        y-component of wave vector

    td : float
        decay timescale

    u0 : float
        initial amplitude of u

    p0 : float
        reference pressure

    rho0 : float
        reference density

    Returns
    p : float
        pressure of Taylor Green Vortex at position (x, y) at time t.
    """
    return p0 - 0.25 * rho0 * u0 * u0 * (
        (ky / kx) * np.cos(2 * kx * x) + (kx / ky) * np.cos(2 * ky * y)
    ) * np.exp(-2 * t / td)


def convert_pressure_to_density(p, p0, rho0, cs=1.0 / np.sqrt(3)):
    """
    Converts pressure to density for a weakly compressible lattice Boltzmann model.

    Parameters
    ----------
    p : float
        pressure

    p0 : float
        reference pressure

    rho0 : float
        reference density (usually unity in lattice Boltzmann model)

    cs : float
        lattice speed of sound in a lattice Boltzmann model

    Returns
    -------
    rho : float
        density used in a weakly compressible lattice Boltzmann model
    """
    return rho0 + (p - p0) / cs**2


def convert_density_to_pressure(rho, rho0, p0, cs=1.0 / np.sqrt(3)):
    """
    Converts lattice Boltzmann density to pressure.

    Parameters
    ----------
    rho : float
        density from a weakly compressible lattice Boltzmann model

    rho0 : float
        reference density (usually unity)

    p0 : float
        reference pressure

    cs : float
        lattice speed of sound in a lattice Boltzmann model

    Returns
    -------
    p : float
        pressure
    """
    return p0 + (rho - rho0) * cs**2


def tgv_r(x, y, t, kx, ky, td, u0, p0, rho0):
    """
    Computes lattice Boltzmann density for 2D Taylor Green Vortex.

    Parameters
    ----------
    x : float
        x-coordinate

    y : float
        y-coordinate

    t : float
        time

    kx : float
        x-component of wave vector

    ky : float
        y-component of wave vector

    td : float
        decay timescale

    u0 : float
        initial amplitude of u

    p0 : float
        reference pressure

    rho0 : float
        reference density

    Returns
    p : float
        pressure of Taylor Green Vortex at position (x, y) at time t.
    """
    p = tgv_p(x, y, t, kx, ky, td, u0, p0, rho0)
    return convert_pressure_to_density(p, p0, rho0)


def relative_error(X_approx, X_true):
    return np.linalg.norm(X_approx - X_true) / np.linalg.norm(X_true)


if __name__ == "__main__":
    nx = 72  # number of grid points in x direction
    ny = 96  # number of grid points in y direction
    u0 = 0.03  # vortex amplitude
    rho0 = 1.0  # reference density
    p0 = 0.0  # reference pressure
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    kx = 2 * np.pi / nx  # wavevector x component
    ky = 2 * np.pi / ny  # wavevector y component
    td = 1.0 / (nu * (kx**2 + ky**2))  # decay timescale
    ti = 0.0  # initial time
    tf = 1000.0  # end time

    model = Model(nx, ny, dx, dt, nu)
    model.set_r(tgv_r, ti, kx, ky, td, u0, p0, rho0)
    model.set_u(tgv_ux, ti, kx, ky, td, u0)
    model.set_v(tgv_uy, ti, kx, ky, td, u0)

    simulation = Simulation(model, stop_time=tf)
    simulation.run()

    X, Y = np.meshgrid(model.x, model.y)
    u_ref = tgv_ux(X, Y, tf, kx, ky, td, u0)
    v_ref = tgv_uy(X, Y, tf, kx, ky, td, u0)
    p_ref = tgv_p(X, Y, tf, kx, ky, td, u0, p0, rho0)

    u_err = relative_error(u_ref, model.u)
    v_err = relative_error(v_ref, model.v)
    p_err = relative_error(p_ref, convert_density_to_pressure(model.r, rho0, p0))

    print(f"Relative error in u: {u_err*100:.2f}%")
    print(f"Relative error in v: {v_err*100:.2f}%")
    print(f"Relative error in p: {p_err*100:.2f}%")

    model.plot_fields("./examples/plots/taylorgreenvortex.png")
