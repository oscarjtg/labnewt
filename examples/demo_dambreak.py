"""Demo script for prototyping the FreeSurfaceModel."""

import matplotlib.pyplot as plt
import numpy as np
from otgraph.video import pngs_to_mp4

from labnewt import (
    BottomWallNoSlip,
    ConstantGravityForce,
    FreeSurfaceModel,
    Simulation,
    TopWallNoSlip,
)

if __name__ == "__main__":
    nx = 100  # number of grid points in x direction
    ny = 100  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 5000.0  # end time
    g = 0.0001  # gravitational acceleration

    eta_args = (ny / 2, ny / 10, nx / 2, 0.0)

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    phi = np.zeros((ny, nx))
    y, x = np.arange(ny), np.arange(nx)
    X, Y = np.meshgrid(x, y)
    y_upper = ny // 2
    x_left = nx // 4
    x_right = 3 * (nx // 4)
    phi[(Y < y_upper) & (X > x_left) & (X < x_right)] = 1.0
    phi[(Y == y_upper) & ((X >= x_left) & (X <= x_right))] = 0.5
    phi[(X == x_left) & (Y <= y_upper)] = 0.5
    phi[(X == x_right) & (Y <= y_upper)] = 0.5
    model.set_phi(phi)
    gravity = ConstantGravityForce(dx, dt, g)
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())
    model._initialise()
    model.print_means()
    model.plot_fields()
    plt.show()
    # model.vof.plot_fields()
    # plt.show()

    sim = Simulation(model, stop_time=tf)
    sim.run(save_frames=True)

    model.print_means()
    # model.plot_fields()
    # plt.show()

    pngs_to_mp4("./frames", "./examples/videos/demo_dambreak.mp4", fps=20)

    # X, Y = np.meshgrid(model.x, model.y)
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].plot(model.x, eta(model.x, *eta_args))
    # p1 = ax[1].pcolormesh(X, Y, model.vof.phi)
    # cbar1 = plt.colorbar(p1, ax=ax[1])
    # cbar1.set_label(r"$\phi$", fontsize=14)
    # plt.show()
