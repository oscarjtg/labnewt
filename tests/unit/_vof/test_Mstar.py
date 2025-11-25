import numpy as np
import numpy.testing as npt

from labnewt._vof import _Mstar_inplace


def compute_Mstar_explicit_loop(M, dMq):
    """
    Computes Mstar with an explicit Python loop.
    `M` and `dMq` are read-only.
    """
    assert M.shape == dMq.shape[1:]

    nq, ny, nx = dMq.shape

    Mstar = np.full_like(M, -1111)

    for y in range(ny):
        for x in range(nx):
            mass = M[y, x]
            for q in range(nq):
                mass += dMq[q, y, x]
            Mstar[y, x] = mass

    return Mstar


def test_Mstar_inplace_zeros():
    nq = 9
    ny = 5
    nx = 6
    rng = np.random.default_rng(42)
    M = rng.random((ny, nx))
    M0 = np.copy(M)

    dMq = np.zeros((nq, ny, nx))

    _Mstar_inplace(M, dMq)

    npt.assert_allclose(M, M0, atol=1.0e-12)


def test_Mstar_inplace_basic():
    nq = 9
    ny = 5
    nx = 6
    rng = np.random.default_rng(42)
    M = rng.random((ny, nx))
    M0 = np.copy(M)

    dMq = np.ones((nq, ny, nx)) / nq

    _Mstar_inplace(M, dMq)
    expected = M0 + np.ones_like(M0)

    npt.assert_allclose(M, expected, atol=1.0e-12)


def test_Mstar_inplace_behaviour():
    nq = 9
    ny = 5
    nx = 6
    rng = np.random.default_rng(42)
    M = rng.random((ny, nx))

    dMq = np.zeros((nq, ny, nx))
    dMq0 = np.copy(dMq)

    id_M = id(M)
    id_dMq = id(dMq)

    _Mstar_inplace(M, dMq)

    # Check that dMq has not changed.
    npt.assert_allclose(dMq, dMq0, atol=1.0e-12)

    # Check that M and dMq are the same objects.
    assert id_M == id(M)
    assert id_dMq == id(dMq)


def test_Mstar_inplace_compare_implementations():
    nq = 9
    ny = 5
    nx = 6

    rng = np.random.default_rng(42)
    M = rng.random((ny, nx))
    dMq = rng.random((nq, ny, nx))

    M0 = np.copy(M)

    expected = compute_Mstar_explicit_loop(M0, dMq)

    _Mstar_inplace(M, dMq)

    # Check implementations.
    npt.assert_allclose(M, expected, atol=1.0e-12)

    # Check mass conservation.
    npt.assert_approx_equal(np.sum(M0) + np.sum(dMq), np.sum(M))
