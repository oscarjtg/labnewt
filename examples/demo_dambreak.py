"""Demo script for prototyping the FreeSurfaceModel."""

import numpy as np

from labnewt import (
    BottomWallNoSlip,
    ConstantGravityForce,
    FreeSurfaceModel,
    NetCDFWriter,
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
    g = 0.0002  # gravitational acceleration

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

    sim = Simulation(model, stop_time=tf)
    save_fields = ["r", "u", "v", "vof.phi"]
    save_path = "./examples/data/demo_dambreak.nc"
    sim.callbacks["netcdfwriter"] = NetCDFWriter(save_fields, save_path, 25)
    sim.add_callback(
        lambda m: m.print_means(), "print_means", np.floor(tf / dt), on_init=True
    )

    sim.run()

    model.print_means()
