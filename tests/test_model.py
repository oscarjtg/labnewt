import numpy as np

from labnewt import Model


def test_model_set_u_func():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return a * x + b * y

    a = 1.0
    b = 10.0
    model.set_u(func, a, b)

    u_expected = np.empty_like(model.u)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            u_expected[j, i] = func(x, y, a, b)

    assert np.allclose(model.u, u_expected, atol=1.0e-12)


def test_model_set_u_array():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return a * x + b * y

    a = 1.0
    b = 10.0

    u_expected = np.empty_like(model.u)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            u_expected[j, i] = func(x, y, a, b)

    model.set_u(u_expected)

    assert np.allclose(model.u, u_expected, atol=1.0e-12)
