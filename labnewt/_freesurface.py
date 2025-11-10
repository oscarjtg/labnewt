import numpy as np

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
