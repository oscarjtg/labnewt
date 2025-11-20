import numpy as np

from labnewt import StencilD2Q9
from labnewt.macroscopic import Macroscopic


def test_density_d2q9_against_known_values():
    macros = Macroscopic()
    nq = 9
    shape = (2, 2)
    f = np.empty((nq, *shape))
    for q in range(nq):
        f[q, 0, 0] = 0.1
        f[q, 0, 1] = 0.1 * (q + 1)
        f[q, 1, 0] = 0.1 if q % 2 == 0 else -0.1
        f[q, 1, 1] = 0.1 * (q + 1) if q % 2 == 0 else -0.1 * (q + 1)

    r_exp = np.empty(shape)
    r_exp[0, 0] = 0.9  # 9 * 0.1
    r_exp[0, 1] = 4.5  # 0.5 * 9 * 10 / 10
    r_exp[1, 0] = 0.1
    r_exp[1, 1] = 0.5

    r_com = np.empty(shape)
    f0 = np.copy(f)
    macros.density(r_com, f)

    # Check r_com was computed correctly.
    assert np.allclose(r_exp, r_com, atol=1.0e-12)
    # Check that f was not changed.
    assert np.allclose(f0, f, atol=1.0e-12)


def test_velocity_x_d2q9_against_known_values():
    macros = Macroscopic()
    s = StencilD2Q9
    shape = (2, 2)
    f = np.empty((s.nq, *shape))
    for q in range(s.nq):
        f[q, 0, 0] = 0.0
        f[q, 0, 1] = 0.1
        f[q, 1, 0] = 0.1 if s.ex[q] > 0 else 0.0
        f[q, 1, 1] = 0.1 if s.ex[q] > 0 else -0.1

    u_exp = np.empty(shape)
    u_exp[0, 0] = 0.0  # all f = 0
    u_exp[0, 1] = 0.0  # f same in all directions, so no net speed
    u_exp[1, 0] = 0.3  # only three vectors in positive x direction
    u_exp[1, 1] = 0.6  # three +ve vals in +ve x, three -ve vals in -ve x

    u_com = np.empty(shape)
    r = np.ones(shape)
    f0 = np.copy(f)
    macros.velocity_x(u_com, r, f, s)

    # Check u_com was computed currectly.
    assert np.allclose(u_exp, u_com, atol=1.0e-12)
    # Check that f was not changed.
    assert np.allclose(f, f0, atol=1.0e-12)


def test_velocity_y_d2q9_against_known_values():
    macros = Macroscopic()
    s = StencilD2Q9
    shape = (2, 2)
    f = np.empty((s.nq, *shape))
    for q in range(s.nq):
        f[q, 0, 0] = 0.0
        f[q, 0, 1] = 0.1
        f[q, 1, 0] = 0.1 if s.ey[q] > 0 else 0.0
        f[q, 1, 1] = 0.1 if s.ey[q] > 0 else -0.1

    v_exp = np.empty(shape)
    v_exp[0, 0] = 0.0  # all f = 0
    v_exp[0, 1] = 0.0  # f same in all directions, so no net speed
    v_exp[1, 0] = 0.3  # only three vectors in positive x direction
    v_exp[1, 1] = 0.6  # three +ve vals in +ve x, three -ve vals in -ve x

    v_com = np.empty(shape)
    r = np.ones(shape)
    f0 = np.copy(f)
    macros.velocity_y(v_com, r, f, s)

    # Check that v_com was computed correctly.
    assert np.allclose(v_exp, v_com, atol=1.0e-12)
    # Check that f was not changed.
    assert np.allclose(f, f0, atol=1.0e-12)


def test_force_distribution_array_with_zeros():
    macros = Macroscopic()
    s = StencilD2Q9()
    shape = (5, 5)
    f = np.random.rand(s.nq, *shape)
    f0 = np.copy(f)
    Fx = np.zeros(shape)
    Fy = np.zeros(shape)
    macros.forcing(f, Fx, Fy, s)

    # With zero force, expect no change to f
    assert np.allclose(f, f0, atol=1.0e-12)

    # Check for unintended side effects
    assert np.allclose(Fx, np.zeros(shape), atol=1.0e-12)
    assert np.allclose(Fy, np.zeros(shape), atol=1.0e-12)


def test_force_distribution_constant_with_zeros():
    macros = Macroscopic()
    s = StencilD2Q9()
    shape = (5, 5)
    f = np.random.rand(s.nq, *shape)
    f0 = np.copy(f)
    Fx = 0.0
    Fy = 0.0
    macros.forcing(f, Fx, Fy, s)

    # With zero force, expect no change to f
    assert np.allclose(f, f0, atol=1.0e-12)
