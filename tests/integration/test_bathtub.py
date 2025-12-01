"""Test script for checking that FreeSurfaceModel reaches steady state"""

import numpy as np
import pytest

from labnewt import (
    BottomWallNoSlip,
    FreeSurfaceModel,
    GravityForce,
    MacroscopicGuo,
    MacroscopicStandard,
    Simulation,
    TopWallNoSlip,
)
from labnewt.diagnostics import average_difference


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_bathtub_periodic(Macroscopic):
    nx = 2  # number of grid points in x direction
    ny = 20  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 100000.0  # end time
    g = 0.001  # gravitational acceleration

    model = FreeSurfaceModel(nx, ny, dx, dt, nu, macros=Macroscopic())
    gravity = GravityForce(dx, dt, g)
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())

    # Set fill fraction, phi
    phi = np.zeros((ny, nx))
    j_surf = ny // 2
    phi[:j_surf, :] = 1.0
    phi[j_surf, :] = 0.5
    model.set_phi(phi)

    # Set density due to hydrostatic pressure
    rho0 = 1.0
    rho = np.full_like(phi, rho0)
    II, JJ = np.meshgrid(np.arange(nx), np.arange(ny))
    rho -= 3 * g * (JJ - j_surf) * dx
    model.set_r(rho)

    u0 = np.copy(model.u)
    v0 = np.copy(model.v)
    r0 = np.copy(model.r)

    sim = Simulation(model, stop_time=tf)
    sim.run_to_steady_state(int(tf), rtol=1.0e-6)

    model.print_integrals()
    model.print_means()

    phi_err = average_difference(model.vof.phi, phi)
    u_err = average_difference(model.vof.phi * model.u, phi * u0)
    v_err = average_difference(model.vof.phi * model.v, phi * v0)
    r_err = average_difference(model.vof.phi * model.r, phi * r0)

    assert abs(phi_err) < 1.0e-03
    assert abs(u_err) < 1.0e-10
    assert abs(v_err) < 2.1e-06
    assert abs(r_err) < 1.0e-10
