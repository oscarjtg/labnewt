"""Demo script for prototyping the FreeSurfaceModel."""

import matplotlib.pyplot as plt
import numpy as np

from labnewt import (
    BottomWallNoSlip,
    ConstantGravityForce,
    FreeSurfaceModel,
    Simulation,
    TopWallNoSlip,
)


def eta(x, height, amplitude, wavelength, phase):
    return height + amplitude * np.sin(2 * np.pi * x / wavelength + phase)


if __name__ == "__main__":
    nx = 100  # number of grid points in x direction
    ny = 50  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 100.0  # end time
    g = 0.001  # gravitational acceleration

    eta_args = (ny / 2, ny / 10, nx / 2, 0.0)

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    model.set_phi_from_eta(eta, *eta_args)
    gravity = ConstantGravityForce(dx, dt, g)
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())
    model._initialise()
    model.print_integrals()

    sim = Simulation(model, stop_time=tf)
    sim.run()

    model.print_integrals()

    model.plot_fields()
    plt.show()
    model.vof.plot_fields()
    plt.show()

    # X, Y = np.meshgrid(model.x, model.y)
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].plot(model.x, eta(model.x, *eta_args))
    # p1 = ax[1].pcolormesh(X, Y, model.vof.phi)
    # cbar1 = plt.colorbar(p1, ax=ax[1])
    # cbar1.set_label(r"$\phi$", fontsize=14)
    # plt.show()
