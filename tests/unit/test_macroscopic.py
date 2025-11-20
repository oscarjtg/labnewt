import numpy as np

from labnewt import Model, StencilD2Q9
from labnewt.macroscopic import MacroscopicStandard


def test_density_d2q9_against_known_values():
    macros = MacroscopicStandard()
    nq = 9
    shape = (2, 2)
    f = np.empty((nq, *shape))
    for q in range(nq):
        f[q, 0, 0] = 0.1
        f[q, 0, 1] = 0.1 * (q + 1)
        f[q, 1, 0] = 0.1 if q % 2 == 0 else -0.1
        f[q, 1, 1] = 0.1 * (q + 1) if q % 2 == 0 else -0.1 * (q + 1)

    f0 = np.copy(f)

    r_exp = np.empty(shape)
    r_exp[0, 0] = 0.9  # 9 * 0.1
    r_exp[0, 1] = 4.5  # 0.5 * 9 * 10 / 10
    r_exp[1, 0] = 0.1
    r_exp[1, 1] = 0.5

    params = (*shape, 1, 1, 1)
    model = Model(*params)
    model.fi = f

    macros.density(model)

    # Check model.r was computed correctly.
    assert np.allclose(model.r, r_exp, atol=1.0e-12)
    # Check that model.fi was not changed.
    assert np.allclose(model.fi, f0, atol=1.0e-12)


def test_velocity_x_d2q9_against_known_values():
    macros = MacroscopicStandard()
    s = StencilD2Q9
    shape = (2, 2)
    f = np.empty((s.nq, *shape))
    for q in range(s.nq):
        f[q, 0, 0] = 0.0
        f[q, 0, 1] = 0.1
        f[q, 1, 0] = 0.1 if s.ex[q] > 0 else 0.0
        f[q, 1, 1] = 0.1 if s.ex[q] > 0 else -0.1

    f0 = np.copy(f)

    u_exp = np.empty(shape)
    u_exp[0, 0] = 0.0  # all f = 0
    u_exp[0, 1] = 0.0  # f same in all directions, so no net speed
    u_exp[1, 0] = 0.3  # only three vectors in positive x direction
    u_exp[1, 1] = 0.6  # three +ve vals in +ve x, three -ve vals in -ve x

    params = (*shape, 1, 1, 1)
    model = Model(*params)
    model.fi = f
    model.r = np.ones(shape)

    macros.velocity_x(model)

    # Check model.u was computed currectly.
    assert np.allclose(model.u, u_exp, atol=1.0e-12)
    # Check that model.fi was not changed.
    assert np.allclose(model.fi, f0, atol=1.0e-12)


def test_velocity_y_d2q9_against_known_values():
    macros = MacroscopicStandard()
    s = StencilD2Q9
    shape = (2, 2)
    f = np.empty((s.nq, *shape))
    for q in range(s.nq):
        f[q, 0, 0] = 0.0
        f[q, 0, 1] = 0.1
        f[q, 1, 0] = 0.1 if s.ey[q] > 0 else 0.0
        f[q, 1, 1] = 0.1 if s.ey[q] > 0 else -0.1

    f0 = np.copy(f)

    v_exp = np.empty(shape)
    v_exp[0, 0] = 0.0  # all f = 0
    v_exp[0, 1] = 0.0  # f same in all directions, so no net speed
    v_exp[1, 0] = 0.3  # only three vectors in positive x direction
    v_exp[1, 1] = 0.6  # three +ve vals in +ve x, three -ve vals in -ve x

    params = (*shape, 1, 1, 1)
    model = Model(*params)
    model.fi = f
    model.r = np.ones(shape)

    macros.velocity_y(model)

    # Check that model.v was computed correctly.
    assert np.allclose(model.v, v_exp, atol=1.0e-12)
    # Check that model.fi was not changed.
    assert np.allclose(model.fi, f0, atol=1.0e-12)


def test_force_distribution_array_with_zeros():
    macros = MacroscopicStandard()
    s = StencilD2Q9()
    shape = (5, 5)
    params = (*shape, 1, 1, 1)
    model = Model(*params)

    rng = np.random.default_rng(42)
    model.fo = rng.random((s.nq, *shape))
    f0 = np.copy(model.fo)
    model.Fx = np.zeros(shape)
    model.Fy = np.zeros(shape)

    macros.forcing(model)

    # With zero force, expect no change to f
    assert np.allclose(model.fo, f0, atol=1.0e-12)

    # Check for unintended side effects
    assert np.allclose(model.Fx, np.zeros(shape), atol=1.0e-12)
    assert np.allclose(model.Fy, np.zeros(shape), atol=1.0e-12)


def test_force_distribution_constant_with_zeros():
    macros = MacroscopicStandard()
    s = StencilD2Q9()
    shape = (5, 5)
    params = (*shape, 1, 1, 1)
    model = Model(*params)

    rng = np.random.default_rng(42)
    model.fo = rng.random((s.nq, *shape))
    f0 = np.copy(model.fo)
    model.Fx = 0.0
    model.Fy = 0.0
    # NB this should never be the case, as model.Fx and model.Fy
    # are supposed to be two-dimensional numpy arrays.

    macros.forcing(model)

    # With zero force, expect no change to f
    assert np.allclose(model.fo, f0, atol=1.0e-12)
