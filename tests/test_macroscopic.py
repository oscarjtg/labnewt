import numpy as np

from labnewt import StencilD2Q9
from labnewt._macroscopic import _density, _velocity_x


def test_density_d2q9_against_known_values():
    nq = 9
    f = np.empty((nq, 2, 2))
    for q in range(nq):
        f[q, 0, 0] = 0.1
        f[q, 0, 1] = 0.1 * (q + 1)
        f[q, 1, 0] = 0.1 if q % 2 == 0 else -0.1
        f[q, 1, 1] = 0.1 * (q + 1) if q % 2 == 0 else -0.1 * (q + 1)

    r_exp = np.empty((2, 2))
    r_exp[0, 0] = 0.9  # 9 * 0.1
    r_exp[0, 1] = 4.5  # 0.5 * 9 * 10 / 10
    r_exp[1, 0] = 0.1
    r_exp[1, 1] = 0.5

    r_com = _density(f)

    assert np.allclose(r_exp, r_com, atol=1.0e-12)


def test_velocity_x_d2q9_against_known_values():
    s = StencilD2Q9
    nq = 9
    f = np.empty((nq, 2, 2))
    for q in range(nq):
        f[q, 0, 0] = 0.0
        f[q, 0, 1] = 0.1
        f[q, 1, 0] = 0.1 if s.ex[q] > 0 else 0.0
        f[q, 1, 1] = 0.1 if s.ex[q] > 0 else -0.1

    u_exp = np.empty((2, 2))
    u_exp[0, 0] = 0.0  # all f = 0
    u_exp[0, 1] = 0.0  # f same in all directions, so no net speed
    u_exp[1, 0] = 0.3  # only three vectors in positive x direction
    u_exp[1, 1] = 0.6  # three +ve vals in +ve x, three -ve vals in -ve x

    u_com = _velocity_x(f, s)

    assert np.allclose(u_exp, u_com, atol=1.0e-12)
