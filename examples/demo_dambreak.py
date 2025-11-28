"""Demo script for prototyping the FreeSurfaceModel."""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import netCDF4 as nc
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
    nx = 101  # number of grid points in x direction
    ny = 50  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 5000.0  # end time
    g = 0.0001  # gravitational acceleration

    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    phi = np.zeros((ny, nx))
    y, x = np.arange(ny), np.arange(nx)
    X, Y = np.meshgrid(x, y)
    y_upper = 4 * ny // 5
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
    save_fields = ["vof.F_mask", "vof.I_mask", "vof.G_mask"]
    data_path = "./examples/data/demo_dambreak.nc"
    sim.callbacks["netcdfwriter"] = NetCDFWriter(save_fields, data_path, 25)
    sim.add_callback(
        lambda m: m.print_means(), "print_means", np.floor(tf / dt), on_init=True
    )
    sim.run()

    # Post-processing
    with nc.Dataset(data_path, "r") as ncfile:
        x = np.array(ncfile.variables["x"][:])
        y = np.array(ncfile.variables["y"][:])
        t = np.array(ncfile.variables["time"][:])
        F_mask = np.array(ncfile.variables["vof.F_mask"][...], dtype=bool)
        I_mask = np.array(ncfile.variables["vof.I_mask"][...], dtype=bool)
        G_mask = np.array(ncfile.variables["vof.G_mask"][...], dtype=bool)

    # Check that no cell has two different types at the same time
    assert not np.any(np.logical_and(F_mask, I_mask))
    assert not np.any(np.logical_and(F_mask, G_mask))
    assert not np.any(np.logical_and(I_mask, G_mask))

    # Check that no FLUID cell has a neighbouring GAS cell at any time.
    neighbours = [[1, 1], [1, 0], [1, -1], [0, 1], [1, -1], [-1, 1], [-1, 0], [-1, -1]]
    nt, ny, nx = G_mask.shape
    for dx, dy in neighbours:
        # Choose slices so that for each (t, y, x) in F_slice,
        # the corresponding G element is at (t, y+dy, x+dx)
        if dx >= 0:
            f_x = slice(0, nx - dx)
            g_x = slice(dx, nx)
        else:
            f_x = slice(-dx, nx)
            g_x = slice(0, nx + dx)

        if dy >= 0:
            f_y = slice(0, ny - dy)
            g_y = slice(dy, ny)
        else:
            f_y = slice(-dy, ny)
            g_y = slice(0, ny + dy)

        # boolean overlap array of shape (nt, ny_overlap, nx_overlap)
        forbidden_neighbours = F_mask[:, f_y, f_x] & G_mask[:, g_y, g_x]

        assert not forbidden_neighbours.any()

    print("Passed all cell type checks!")

    # Animation: show cell type evolution over time.
    print("Making animation")
    FLUID_VALUE = 1
    INTERFACE_VALUE = 0.5
    GAS_VALUE = 0.0

    ct = np.full_like(F_mask, FLUID_VALUE, dtype=np.float64)
    ct[I_mask] = INTERFACE_VALUE
    ct[G_mask] = GAS_VALUE

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x, y, ct[0, ...], vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_title(f"t = {t[0]:.2f}")

    def update(i):
        mesh.set_array(ct[i, ...].ravel())
        ax.set_title(f"t = {t[i]:.2f}")
        return mesh,

    anim_dir = "./examples/videos/"
    os.makedirs(anim_dir, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=t.shape[0], interval=50)
    anim.save(anim_dir + "demo_dambreak.mp4", fps=20)
