import numpy as np

from ._equilibrium import _feq2_q
from ._shift import periodic_shift


def _delta_M_q(q, f, s, phi, mask_gas, mask_fluid, mask_interface):
    """
    Compute Delta M for lattice vector q.

    Parameters
    ----------
    q : int
        Lattice vector index

    f : np.ndarray
        Three dimensional numpy array f[q, y, x]

    s : Stencil
        Stencil

    phi : np.ndarray
        Two dimensional numpy array of floats, shape = (y, x)

    mask_gas : np.ndarray
        Two dimensional numpy array of booleans, shape = (y, x)

    mask_fluid : np.ndarray
        Two dimensional numpy array of booleans, shape = (y, x)

    mask_interface : np.ndarray
        Two dimensional numpy array of booleans, shape = (y, x)

    Returns
    --------
    delta_M_q : np.ndarray
        Two dimensional numpy array of floats, shape = (y, x)
    """
    assert mask_gas.shape == mask_fluid.shape
    assert mask_gas.shape == mask_interface.shape
    delta_M_q = np.zeros(mask_gas.shape, dtype=np.float64)

    # mask_gas_shifted = periodic_shift(mask_gas, s, q)
    mask_fluid_shifted = periodic_shift(mask_fluid, s, s.q_rev[q])
    mask_interface_shifted = periodic_shift(mask_interface, s, s.q_rev[q])

    f_q_out = np.copy(f[q, :, :])
    f_qbar_out = np.copy(f[s.q_rev[q], :, :])
    f_qbar_out_shifted = periodic_shift(f_qbar_out, s, s.q_rev[q])

    mask_if = mask_interface * mask_fluid_shifted
    mask_ii = mask_interface * mask_interface_shifted
    assert (mask_if * mask_ii == False).all

    # Interface <-> Fluid mass transfer.
    delta_M_q[mask_if] += f_qbar_out_shifted[mask_if] - f_q_out[mask_if]

    # Interface <-> Interface mass transfer.
    phi_shifted = periodic_shift(phi, s, s.q_rev[q])
    delta_M_q[mask_ii] += (
        0.5
        * (phi[mask_ii] + phi_shifted[mask_ii])
        * (f_qbar_out_shifted[mask_ii] - f_q_out[mask_ii])
    )
    return delta_M_q


def _free_surface_boundary_condition_q(q, f, s, u, v, mask_gas, mask_interface, rho_G=1.0):
    """
    Apply fluid-gas boundary condition in-place.

    NB modifies distribution functions that have just been streamed
    into an interface cell from a neighbouring gas cells.

    This method should only touch interface cells

    Parameters
    ----------
    q : int
        Lattice vector index

    f : np.ndarray
        Three dimensional numpy array of floats containing distribution functions.
        f.shape = (nq, ny, nx)

    s : stencil
        Lattice stencil

    u : np.ndarray
        Two dimenaional numpy array of floats containing x-component of velocity.
        u.shape = (ny, nx)

    v : np.ndarray
        Two dimensional numpy array of floats containing y-component of velocity.
        v.shape = (ny, nx)

    mask_gas : np.ndarray
        Two dimensional numpy array of booleans. shape = (ny, nx)

    mask_interface : np.ndarray
        Two dimenaionl numpy array of booleans. shape = (ny, nx)

    rho_G : float
        Gas density, in lattice units. Default = 1.0
    """
    mask_gas_shifted = periodic_shift(mask_gas, s, q)
    f_out_qrev = np.copy(f[s.q_rev[q], :, :])
    feq_q = _feq2_q(q, rho_G, u, v, s)
    feq_qrev = _feq2_q(s.q_rev[q], rho_G, u, v, s)

    mask = mask_interface * mask_gas_shifted
    f[q, mask] = feq_q[mask] + feq_qrev[mask] - f_out_qrev[mask]
