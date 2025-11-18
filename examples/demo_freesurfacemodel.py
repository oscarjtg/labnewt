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


def eta(x, height, amplitude, wavelength, phase):
    return height + amplitude * np.sin(2 * np.pi * x / wavelength + phase)


if __name__ == "__main__":
    nx = 100  # number of grid points in x direction
    ny = 50  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 1000.0  # end time
    g = 0.0001  # gravitational acceleration

    eta_args = (ny / 2, ny / 10, nx / 2, 0.0)

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    model.set_phi_from_eta(eta, *eta_args)
    gravity = ConstantGravityForce(dx, dt, g)
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())
    model._initialise()
    model.print_means()
    model.plot_fields()
    plt.show()

    sim = Simulation(model, stop_time=tf)
    sim.run(save_frames=True)

    model.print_means()

    pngs_to_mp4("./frames", "./examples/demo_freesurfacemodel.mp4", fps=5)

    model.plot_fields()
    plt.show()
