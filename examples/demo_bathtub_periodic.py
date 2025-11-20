"""Demo script for checking that FreeSurfaceModel reaches steady state"""

import os

import matplotlib.pyplot as plt
import numpy as np

from labnewt import (
    BottomWallNoSlip,
    FreeSurfaceModel,
    GravityForce,
    Simulation,
    TopWallNoSlip,
)


def run_periodic_bathtub_model(g, nx, ny, nu, dx, dt, tf, j_surf):
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)
    gravity = GravityForce(dx, dt, g)
    model.add_forcing(gravity)
    model.add_boundary_condition(BottomWallNoSlip())
    model.add_boundary_condition(TopWallNoSlip())

    # Set fill fraction, phi
    phi = np.zeros((ny, nx))
    phi[:j_surf, :] = 1.0
    phi[j_surf, :] = 0.5
    model.set_phi(phi)

    model._initialise()

    sim = Simulation(model, stop_time=tf)
    sim.run_to_steady_state(int(tf), rtol=1.0e-10)

    return model


def hydrostatic_density(phi, g):
    """
    Calculate LB density due to hydrostatic pressure.

    Calculated by direct integration from top to bottom of domain.

    We use the density at the level above,

        rho[y] = rho[y + 1] + rho[y + 1] * g * phi[y + 1] * dx / cs^2

    This gives better agreement to model than

        rho[y] = rho[y + 1] + rho0 * g * phi[y + 1] * dx / cs^2,

    with constant rho0

    Parameters
    ----------
    phi : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains cell fill fractions.
    g : float
        Gravitational acceleration.

    Returns
    -------
    rho : np.ndarray
        Two-dimensional numpy array of floats of shape (ny, nx).
        Contains local lattice Boltzmann densities,
        which are related to pressure by the relationship
        Delta p = Delta rho * cs^2,
        where Delta p is pressure difference from reference,
        Delta rho is density difference from reference,
        and cs = 1/sqrt(3) is the lattice speed of sound.
    """
    ny, nx = phi.shape
    rho0 = 1.0
    rho = np.full(model.shape, rho0)
    for j in np.arange(ny - 2, -1, -1):
        rho[j, :] = rho[j + 1, :] + 3.0 * rho[j + 1, :] * g * phi[j + 1, :] * dx
    return rho


def plot_denisty_and_velocity_profiles(model, rho_hydrostatic):
    """
    Plots vertical density and velocity profiles.

    Parameters
    ----------
    model : Model
        Model instance
    rho_hydrostatic : np.ndarray
        Two-dimensional array of floats of shape (ny, nx).
        Contains hydrostatic density.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].plot(np.mean(model.r, axis=1), model.y, label="model")
    ax[0].plot(np.mean(rho_hydrostatic, axis=1), model.y, label="hydrostatic")
    ax[0].plot(
        1 + 0.05 * np.mean(model.vof.phi, axis=1),
        model.y,
        color="black",
        ls="dashed",
        label="phi",
    )
    ax[0].set_xlabel(r"$\rho$", fontsize=16)
    ax[0].set_ylabel("y", fontsize=16)
    ax[0].tick_params(labelsize=14)
    ax[0].legend(fontsize=14)

    ax[1].plot(np.mean(model.v, axis=1), model.y, label="model")
    ax[1].plot(np.zeros(ny), model.y, label="hydrostatic")
    ax[1].set_xlabel("v", fontsize=16)
    ax[1].set_ylabel("y", fontsize=16)
    ax[1].tick_params(labelsize=14)
    ax[1].legend(fontsize=14)


def run_gravity_sweep(gs, params, savedir):
    v_errs = np.zeros_like(gs)
    for i, g in enumerate(gs):
        model = run_periodic_bathtub_model(g, *params)
        v_err = np.max(np.abs(model.v))
        print(f"g = {g}, v_err = {v_err}, ratio = {v_err/g}")
        v_errs[i] = v_err
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "gs.npy", gs)
    np.save(savedir + "v_errs.npy", v_errs)


def plot_gravity_sweep(savedir):
    gs = np.load(savedir + "gs.npy")
    v_errs = np.load(savedir + "v_errs.npy")

    print(gs)
    print(v_errs)
    print(v_errs / gs)

    plt.plot(gs, v_errs / gs, "ko")
    plt.xlabel(r"$g$", fontsize=16)
    plt.ylabel(r"$v / g$", fontsize=16)
    plt.xscale("log")
    plt.grid(True)
    plt.ylim((0.4, 0.6))
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("./examples/plots/bathtub_velocity_vs_gravity.png", dpi=150)


if __name__ == "__main__":
    nx = 2  # number of grid points in x direction
    ny = 20  # number of grid points in y direction
    nu = 0.1  # kinematic viscosity
    dx = 1  # grid spacing
    dt = 1  # time step
    tf = 100000.0  # end time
    g = 0.001  # gravitational acceleration
    j_surf = ny // 2  # j index of free surface
    params = (nx, ny, nu, dx, dt, tf, j_surf)
    savedir = "./examples/data/bathtub/"

    SINGLE = False
    SWEEP = True
    SWEEP_PLOT_ONLY = True

    if SINGLE:
        model = run_periodic_bathtub_model(g, *params)
        rho_hydrostatic = hydrostatic_density(model.vof.phi, g)
        plot_denisty_and_velocity_profiles(model, rho_hydrostatic)
        plt.show()

    if SWEEP:
        gs = np.logspace(-5, -3, 10)
        if not SWEEP_PLOT_ONLY:
            run_gravity_sweep(gs, params, savedir)
        plot_gravity_sweep(savedir)
        plt.show()
