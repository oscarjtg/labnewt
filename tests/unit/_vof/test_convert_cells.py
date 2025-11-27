import numpy as np

from labnewt._vof import _convert_cells_, _convert_FI_, _convert_GI_


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
    assert np.array_equal(G_mask, G_expected)
    assert np.array_equal(I_mask, I_expected)

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
    assert np.array_equal(F_mask, F_expected)
    assert np.array_equal(I_mask, I_expected)

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
