"""Gravity-driven flow of a viscous fluid in a 2d rectangular channel"""

import os

import matplotlib.pyplot as plt
import numpy as np

from labnewt import (
    ConstantGravityForce,
    LeftWallNoSlip,
    Model,
    RightWallNoSlip,
    Simulation,
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


if __name__ == "__main__":
    gx = 0.0  # gravity x-component
    gy = -0.8  # gravity y-component
    nu = 0.1  # kinematic viscosity
    L = 1.0  # channel width
    dx = 0.05  # grid spacing
    dt = 0.005  # time step
    tf = 20.0  # end time

    nx = int(L / dx)
    ny = 1

    model = Model(nx, ny, dx, dt, nu, quiet=False)
    gravity = ConstantGravityForce(dx, dt, abs(gy), (gx, gy))
    model.add_forcing(gravity)
    model.add_boundary_condition(LeftWallNoSlip())
    model.add_boundary_condition(RightWallNoSlip())

    simulation = Simulation(model, stop_time=tf)
    simulation.run_to_steady_state(int(tf / dt), rtol=1.0e-05)

    U_analytic = velocity_profile(model.x, gy, L, nu)
    U_numeric = model.v[0, :] * dx / dt

    error = relative_error(U_numeric, U_analytic)
    print(f"Relative error = {error*100:.4f}%")

    x_fine = np.linspace(0, L)
    U_fine = velocity_profile(x_fine, gy, L, nu)

    plt.plot(x_fine, U_fine, color="black", label="analytic")
    plt.plot(model.x, U_numeric, "ko", label="numeric")
    plt.grid(True)
    plt.xlabel("x", fontsize=14)
    plt.ylabel(r"$u_z$", fontsize=14)
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=13)
    path = "./examples/plots/poiseuille.png"
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    plt.savefig("./examples/plots/poiseuille.png", dpi=150)
    plt.show()
