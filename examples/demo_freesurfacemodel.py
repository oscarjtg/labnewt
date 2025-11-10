"""Demo script for prototyping the FreeSurfaceModel."""

import matplotlib.pyplot as plt
import numpy as np

from labnewt import FreeSurfaceModel


def eta(x, height, amplitude, wavelength, phase):
    return height + amplitude * np.sin(2 * np.pi * x / wavelength + phase)


if __name__ == "__main__":
    nx = 48  # number of grid points in x direction
    ny = 48  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 1000.0  # end time

    eta_args = (ny / 2, ny / 10, nx / 2, 0.0)

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    model.set_phi_from_eta(eta, *eta_args)

    X, Y = np.meshgrid(model.x, model.y)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].plot(model.x, eta(model.x, *eta_args))
    p1 = ax[1].pcolormesh(X, Y, model.phi)
    cbar1 = plt.colorbar(p1, ax=ax[1])
    cbar1.set_label(r"$\phi$", fontsize=14)
    plt.show()
