import numpy as np

from labnewt._vof._convert_cells import (
    _convert_cells_,
    _convert_FI_,
    _convert_GI_,
    _identify_overfull_,
    _identify_underfull_,
)


def test_convert_GI_linear_interface():
    shape = (3, 3)
    I_mask = np.zeros(shape, dtype=bool)
    G_mask = np.zeros(shape, dtype=bool)

    I_mask[1, :] = True
    G_mask[2, :] = True

    cIF = np.zeros(shape, dtype=bool)
    cIF[1, 1] = True
    cIF_copy = np.copy(cIF)

    I_expected = np.array(
        [[False, False, False], [True, False, True], [True, True, True]]
    )
    G_expected = np.zeros(shape, dtype=bool)

    _convert_GI_(G_mask, I_mask, cIF)

    # Check correct values obtained
    assert np.array_equal(G_mask[~cIF], G_expected[~cIF])
    assert np.array_equal(I_mask[~cIF], I_expected[~cIF])

    # Check no unintended side effects
    assert np.array_equal(cIF, cIF_copy)


def test_convert_FI_linear_interface():
    shape = (3, 3)
    F_mask = np.zeros(shape, dtype=bool)
    I_mask = np.zeros(shape, dtype=bool)

    F_mask[0, :] = True
    I_mask[1, :] = True

    cIG = np.zeros(shape, dtype=bool)
    cIG[1, 1] = True
    cIG_copy = np.copy(cIG)

    I_expected = np.array(
        [[True, True, True], [True, False, True], [False, False, False]]
    )
    F_expected = np.zeros(shape, dtype=bool)

    _convert_FI_(F_mask, I_mask, cIG)

    # Check correct values obtained
    assert np.array_equal(F_mask[~cIG], F_expected[~cIG])
    assert np.array_equal(I_mask[~cIG], I_expected[~cIG])

    # Check no unintended side effects
    assert np.array_equal(cIG, cIG_copy)


def test_convert_cells_linear_interface_to_fluid():
    shape = (3, 3)
    F_mask = np.zeros(shape, dtype=bool)
    I_mask = np.zeros(shape, dtype=bool)
    G_mask = np.zeros(shape, dtype=bool)

    F_mask[0, :] = True
    I_mask[1, :] = True
    G_mask[2, :] = True

    cIF = np.zeros(shape, dtype=bool)
    cIF[1, 1] = True
    cIF_copy = np.copy(cIF)

    cIG = np.zeros(shape, dtype=bool)
    cIG_copy = np.copy(cIG)

    F_expected = np.array(
        [[True, True, True], [False, True, False], [False, False, False]]
    )
    I_expected = np.array(
        [[False, False, False], [True, False, True], [True, True, True]]
    )
    G_expected = np.zeros(shape, dtype=bool)

    _convert_cells_(F_mask, I_mask, G_mask, cIF, cIG)

    # Check correct values obtained
    assert np.array_equal(F_mask, F_expected)
    assert np.array_equal(I_mask, I_expected)
    assert np.array_equal(G_mask, G_expected)

    # Check no unintended side effects
    assert np.array_equal(cIF, cIF_copy)
    assert np.array_equal(cIG, cIG_copy)


def test_convert_cells_linear_interface_to_gas():
    shape = (3, 3)
    F_mask = np.zeros(shape, dtype=bool)
    I_mask = np.zeros(shape, dtype=bool)
    G_mask = np.zeros(shape, dtype=bool)

    F_mask[0, :] = True
    I_mask[1, :] = True
    G_mask[2, :] = True

    cIF = np.zeros(shape, dtype=bool)
    cIF_copy = np.copy(cIF)

    cIG = np.zeros(shape, dtype=bool)
    cIG[1, 1] = True
    cIG_copy = np.copy(cIG)

    F_expected = np.zeros(shape, dtype=bool)
    I_expected = np.array(
        [[True, True, True], [True, False, True], [False, False, False]]
    )
    G_expected = np.array(
        [[False, False, False], [False, True, False], [True, True, True]]
    )

    _convert_cells_(F_mask, I_mask, G_mask, cIF, cIG)

    # Check correct values obtained
    assert np.array_equal(F_mask, F_expected)
    assert np.array_equal(I_mask, I_expected)
    assert np.array_equal(G_mask, G_expected)

    # Check no unintended side effects
    assert np.array_equal(cIF, cIF_copy)
    assert np.array_equal(cIG, cIG_copy)


def test_convert_cells_corner_to_fluid():
    shape = (3, 3)
    F_mask = np.zeros(shape, dtype=bool)
    I_mask = np.zeros(shape, dtype=bool)
    G_mask = np.zeros(shape, dtype=bool)

    F_mask[0, 0] = True
    I_mask[0, 1] = True
    I_mask[1, :2] = True
    G_mask[0, 2] = True
    G_mask[1, 2] = True
    G_mask[2, :] = True

    cIF = np.zeros(shape, dtype=bool)
    cIF[1, 1] = True
    cIF_copy = np.copy(cIF)

    cIG = np.zeros(shape, dtype=bool)
    cIG_copy = np.copy(cIG)

    F_expected = np.array(
        [[True, False, False], [False, True, False], [False, False, False]]
    )
    I_expected = np.array(
        [[False, True, True], [True, False, True], [True, True, True]]
    )
    G_expected = np.zeros(shape, dtype=bool)

    _convert_cells_(F_mask, I_mask, G_mask, cIF, cIG)

    # Check correct values obtained
    assert np.array_equal(F_mask, F_expected)
    assert np.array_equal(I_mask, I_expected)
    assert np.array_equal(G_mask, G_expected)

    # Check no unintended side effects
    assert np.array_equal(cIF, cIF_copy)
    assert np.array_equal(cIG, cIG_copy)


def test_convert_cells_corner_to_gas():
    shape = (3, 3)
    F_mask = np.zeros(shape, dtype=bool)
    I_mask = np.zeros(shape, dtype=bool)
    G_mask = np.zeros(shape, dtype=bool)

    F_mask[0, 0] = True
    I_mask[0, 1] = True
    I_mask[1, :2] = True
    G_mask[0, 2] = True
    G_mask[1, 2] = True
    G_mask[2, :] = True

    cIF = np.zeros(shape, dtype=bool)
    cIF_copy = np.copy(cIF)

    cIG = np.zeros(shape, dtype=bool)
    cIG[1, 1] = True
    cIG_copy = np.copy(cIG)

    F_expected = np.zeros(shape, dtype=bool)
    I_expected = np.array(
        [[True, True, False], [True, False, False], [False, False, False]]
    )
    G_expected = np.array(
        [[False, False, True], [False, True, True], [True, True, True]]
    )

    _convert_cells_(F_mask, I_mask, G_mask, cIF, cIG)

    # Check correct values obtained
    assert np.array_equal(F_mask, F_expected)
    assert np.array_equal(I_mask, I_expected)
    assert np.array_equal(G_mask, G_expected)

    # Check no unintended side effects
    assert np.array_equal(cIF, cIF_copy)
    assert np.array_equal(cIG, cIG_copy)


def test_identify_underfull_random_():
    shape = (3, 4)
    rng = np.random.default_rng(42)
    to_gas = rng.random(shape) > 0.5
    I_mask = rng.random(shape) > 0.5
    M = rng.normal(0, 1, size=shape)
    eps = 1.0e-04

    to_gas_id = id(to_gas)
    I_mask_id = id(I_mask)
    M_id = id(M)

    I_mask_init = np.copy(I_mask)
    M_init = np.copy(M)

    _identify_underfull_(to_gas, I_mask, M, eps)

    def slow_version(I_mask, M, eps):
        ny, nx = I_mask.shape
        to_gas = np.zeros_like(I_mask, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                if I_mask[j, i] and M[j, i] < eps:
                    to_gas[j, i] = True
                else:
                    to_gas[j, i] = False
        return to_gas

    expected = slow_version(I_mask, M, eps)

    # Check correct values
    assert np.allclose(to_gas, expected)

    # Check no unintended side effects
    assert np.allclose(I_mask, I_mask_init)
    assert np.allclose(M, M_init)
    assert id(to_gas) == to_gas_id
    assert id(I_mask) == I_mask_id
    assert id(M) == M_id


def test_identify_overfull_random_():
    shape = (3, 4)
    rng = np.random.default_rng(42)
    to_fluid = rng.random(shape) > 0.5
    I_mask = rng.random(shape) > 0.5
    M = rng.normal(0, 1, size=shape)
    rho = rng.random(shape)
    eps = 1.0e-04

    to_fluid_id = id(to_fluid)
    I_mask_id = id(I_mask)
    M_id = id(M)

    I_mask_init = np.copy(I_mask)
    M_init = np.copy(M)

    _identify_overfull_(to_fluid, I_mask, M, rho, eps)

    def slow_version(I_mask, M, rho, eps):
        ny, nx = I_mask.shape
        to_fluid = np.zeros_like(I_mask, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                if I_mask[j, i] and M[j, i] > rho[j, i] + eps:
                    to_fluid[j, i] = True
                else:
                    to_fluid[j, i] = False
        return to_fluid

    expected = slow_version(I_mask, M, rho, eps)

    # Check correct values
    assert np.allclose(to_fluid, expected)

    # Check no unintended side effects
    assert np.allclose(I_mask, I_mask_init)
    assert np.allclose(M, M_init)
    assert id(to_fluid) == to_fluid_id
    assert id(I_mask) == I_mask_id
    assert id(M) == M_id
