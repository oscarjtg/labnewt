"""Gravity-driven flow of a viscous fluid in a 2d rectangular channel"""

import pytest

from labnewt import (
    BottomWallNoSlip,
    ConstantGravityForce,
    LeftWallNoSlip,
    MacroscopicGuo,
    MacroscopicStandard,
    Model,
    RightWallNoSlip,
    Simulation,
    TopWallNoSlip,
)
from labnewt.diagnostics import relative_error


def velocity_profile(x, g, L, nu):
    """
    Calculates the fluid velocity in a two-dimensional channel.

    Assumes laminar (Poiseuille) flow.

    Parameters
    ---------
    x : float
        Position across channel, in m.
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


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_poiseuille_vertical_down(Macroscopic):
    gx = 0.0  # gravity x-component
    gy = -0.8  # gravity y-component
    nu = 0.1  # kinematic viscosity
    L = 1.0  # channel width
    dx = 0.05  # grid spacing
    dt = 0.005  # time step
    tf = 20.0  # end time

    nx = int(L / dx)
    ny = 1

    model = Model(nx, ny, dx, dt, nu, macros=Macroscopic(), quiet=False)
    gravity = ConstantGravityForce(dx, dt, abs(gy), (gx, gy))
    model.add_forcing(gravity)
    model.add_boundary_condition(LeftWallNoSlip())
    model.add_boundary_condition(RightWallNoSlip())

    simulation = Simulation(model, stop_time=tf)
    simulation.run_to_steady_state(int(tf / dt), rtol=1.0e-08)

    U_analytic = velocity_profile(model.x, gy, L, nu)
    U_numeric = model.v[0, :] * dx / dt

    error = relative_error(U_numeric, U_analytic)
    percent_error = error * 100
    assert (0.3125 < percent_error) & (percent_error < 0.3127)


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_poiseuille_vertical_up(Macroscopic):
    gx = 0.0  # gravity x-component
    gy = 0.8  # gravity y-component
    nu = 0.1  # kinematic viscosity
    L = 1.0  # channel width
    dx = 0.05  # grid spacing
    dt = 0.005  # time step
    tf = 20.0  # end time

    nx = int(L / dx)
    ny = 1

    model = Model(nx, ny, dx, dt, nu, macros=Macroscopic(), quiet=False)
    gravity = ConstantGravityForce(dx, dt, abs(gy), (gx, gy))
    model.add_forcing(gravity)
    model.add_boundary_condition(LeftWallNoSlip())
    model.add_boundary_condition(RightWallNoSlip())

    simulation = Simulation(model, stop_time=tf)
    simulation.run_to_steady_state(int(tf / dt), rtol=1.0e-08)

    U_analytic = velocity_profile(model.x, gy, L, nu)
    U_numeric = model.v[0, :] * dx / dt

    error = relative_error(U_numeric, U_analytic)
    percent_error = error * 100
    assert (0.3125 < percent_error) & (percent_error < 0.3127)


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_poiseuille_horizontal(Macroscopic):
    gx = 0.8  # gravity x-component
    gy = 0.0  # gravity y-component
    nu = 0.1  # kinematic viscosity
    L = 1.0  # channel width
    dx = 0.05  # grid spacing
    dt = 0.005  # time step
    tf = 20.0  # end time

    nx = 1
    ny = int(L / dx)

    model = Model(nx, ny, dx, dt, nu, macros=Macroscopic(), quiet=False)
    gravity = ConstantGravityForce(dx, dt, abs(gx), (gx, gy))
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())

    simulation = Simulation(model, stop_time=tf)
    simulation.run_to_steady_state(int(tf / dt), rtol=1.0e-08)

    U_analytic = velocity_profile(model.y, gx, L, nu)
    U_numeric = model.u[:, 0] * dx / dt

    error = relative_error(U_numeric, U_analytic)
    percent_error = error * 100
    assert (0.3125 < percent_error) & (percent_error < 0.3127)
