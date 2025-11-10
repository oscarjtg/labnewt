import numpy as np

from labnewt import StencilD2Q9
from labnewt._equilibrium import _feq2
from labnewt._freesurface import _delta_M_q, _free_surface_boundary_condition_q


def test_delta_M_q_basic():
    s = StencilD2Q9()
    nx = 3
    ny = 3
    shape = (ny, nx)
    rho = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    f = _feq2(rho, u, v, s)

    phi = np.zeros(shape)
    mask_gas = np.zeros(shape, dtype=bool)
    mask_fluid = np.zeros(shape, dtype=bool)
    mask_interface = np.zeros(shape, dtype=bool)

    # Bottom row
    phi[0, :] = 1.0
    mask_fluid[0, :] = True

    # Middle row
    phi[1, :] = 0.5
    mask_interface[1, :] = True

    # Top row
    phi[2, :] = 0.0
    mask_gas[2, :] = True

    for q in range(s.nq):
        deltaMq = _delta_M_q(q, f, s, phi, mask_gas, mask_fluid, mask_interface)
        assert np.allclose(deltaMq, np.zeros(shape), atol=1.0e-12)


def test_delta_M_q_basic_with_density_gradient():
    s = StencilD2Q9()
    nx = 3
    ny = 3
    shape = (ny, nx)
    rho = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)

    phi = np.zeros(shape)
    mask_gas = np.zeros(shape, dtype=bool)
    mask_fluid = np.zeros(shape, dtype=bool)
    mask_interface = np.zeros(shape, dtype=bool)

    delta_rho = 0.01
    # Bottom row
    rho[0, :] += delta_rho
    phi[0, :] = 1.0
    mask_fluid[0, :] = True

    # Middle row
    phi[1, :] = 0.5
    mask_interface[1, :] = True

    # Top row
    rho[2, :] -= delta_rho
    phi[2, :] = 0.0
    mask_gas[2, :] = True

    f = _feq2(rho, u, v, s)

    for q in [0, 1, 2, 3, 5, 7]:
        deltaMq = _delta_M_q(q, f, s, phi, mask_gas, mask_fluid, mask_interface)
        assert np.allclose(deltaMq, np.zeros(shape), atol=1.0e-12)

    deltaMq = _delta_M_q(4, f, s, phi, mask_gas, mask_fluid, mask_interface)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],  # bottom row
            [delta_rho / 9, delta_rho / 9, delta_rho / 9],  # middle row
            [0.0, 0.0, 0.0],
        ]
    )  # top row
    assert np.allclose(deltaMq, expected, atol=1.0e-12)

    for q in [6, 8]:
        deltaMq = _delta_M_q(q, f, s, phi, mask_gas, mask_fluid, mask_interface)
        expected = expected = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom row
                [delta_rho / 36, delta_rho / 36, delta_rho / 36],  # middle row
                [0.0, 0.0, 0.0],
            ]
        )  # top row
        assert np.allclose(deltaMq, expected, atol=1.0e-12)


def test_free_surface_boundary_condition_q():
    s = StencilD2Q9()
    nx = 3
    ny = 3
    shape = (ny, nx)
    rho = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    f = _feq2(rho, u, v, s)
    f0 = np.copy(f)

    phi = np.zeros(shape)
    mask_gas = np.zeros(shape, dtype=bool)
    mask_fluid = np.zeros(shape, dtype=bool)
    mask_interface = np.zeros(shape, dtype=bool)

    # Bottom row
    phi[0, :] = 1.0
    mask_fluid[0, :] = True

    # Middle row
    phi[1, :] = 0.5
    mask_interface[1, :] = True

    # Top row
    phi[2, :] = 0.0
    mask_gas[2, :] = True

    for q in range(s.nq):
        _free_surface_boundary_condition_q(q, f, s, u, v, mask_gas, mask_interface)
    assert np.allclose(f, f0, atol=1.0e-12)