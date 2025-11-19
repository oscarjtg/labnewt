"""Demo script for checking that FreeSurfaceModel reaches steady state"""

import matplotlib.pyplot as plt
import numpy as np

from labnewt import (
    BottomWallNoSlip,
    ConstantGravityForce,
    FreeSurfaceModel,
    GravityForce,
    Simulation,
    TopWallNoSlip,
)
from labnewt.diagnostics import average_difference

if __name__ == "__main__":
    nx = 2  # number of grid points in x direction
    ny = 20  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 100000.0  # end time
    g = 0.001  # gravitational acceleration

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
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
    # model.set_r(rho)

    model._initialise()
    model.print_means()
    model.plot_fields()
    plt.show()

    u0 = np.copy(model.u)
    v0 = np.copy(model.v)
    r0 = np.copy(model.r)

    sim = Simulation(model, stop_time=tf)
    sim.run_to_steady_state(int(tf), rtol=1.0e-8)

    model.print_means()

    phi_err = average_difference(model.vof.phi, phi)
    u_err = average_difference(model.vof.phi * model.u, phi * u0)
    v_err = average_difference(model.vof.phi * model.v, phi * v0)
    r_err = average_difference(model.vof.phi * model.r, phi * r0)

    error_dict = {
        "phi    ": phi_err,
        "u      ": u_err,
        "v      ": v_err,
        "density": r_err,
    }

    for key, val in error_dict.items():
        print(f"Average element-wise change in {key} = {val:.12f}")

    model.plot_fields()
    plt.show()
