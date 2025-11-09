"""Gravity-driven flow of a viscous fluid in a 2d rectangular channel"""

import numpy as np

from labnewt import ConstantGravityForce, LeftRightWallsNoSlip, Model, Simulation


def velocity_profile(x, g, L, nu):
    """
    Calculates the Poiseuille velocity at position x
    along a channel of width L.

    Parameters
    ---------
    x : float
        Position along channel, in m.

    g : float
        Gravity, in m/s^2.

    L : float
        Channel width, in m.

    nu : float
        Kinematic viscosity, in m^2/s.

    Returns
    -------
    u : float
        Flow velocity.
    """
    return g / (2.0 * nu) * (L - x) * x


def relative_error(X_approx, X_exact):
    return np.linalg.norm(X_approx - X_exact) / np.linalg.norm(X_exact)


def test_poiseuille_vertical():
    gx = 0.0  # gravity x-component
    gy = -0.8  # gravity y-component
    nu = 0.1  # kinematic viscosity
    L = 1.0  # channel width
    dx = 0.05  # grid spacing
    dt = 0.005  # time step
    tf = 10.0  # end time

    nx = int(L / dx)
    ny = 1

    model = Model(nx, ny, dx, dt, nu, quiet=False)
    gravity = ConstantGravityForce(dx, dt, abs(gy), gx, gy)
    model.add_forcing(gravity)
    model.add_boundary_condition(LeftRightWallsNoSlip())

    simulation = Simulation(model, stop_time=tf)
    simulation.run(print_progress=False)

    U_analytic = velocity_profile(model.x, gy, L, nu)
    U_numeric = model.v[0, :] * dx / dt

    error = relative_error(U_numeric, U_analytic)
    assert error * 100 < 0.05
