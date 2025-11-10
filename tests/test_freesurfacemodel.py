import numpy as np

from labnewt import FreeSurfaceModel


def test_free_surface_model_set_phi_func():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return np.where((x < a) * (y < b), 1.0, 0.0)

    a = 0.2
    b = 0.5
    model.set_phi(func, a, b)

    phi_expected = np.empty_like(model.phi)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            phi_expected[j, i] = func(x, y, a, b)

    assert np.allclose(model.phi, phi_expected, atol=1.0e-12)


def test_free_surface_model_set_phi_array():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return np.where((x < a) * (y < b), 1.0, 0.0)

    a = 0.2
    b = 0.5

    phi_expected = np.empty_like(model.u)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            phi_expected[j, i] = func(x, y, a, b)

    model.set_phi(phi_expected)

    assert np.allclose(model.phi, phi_expected, atol=1.0e-12)


def test_free_surface_set_phi_with_eta_func():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    def eta_func(x, a, b):
        return np.where(x < a, b, 0)

    a = 0.5
    b = 0.5

    model.set_phi_from_eta(eta_func, a, b)

    X, Y = np.meshgrid(model.x, model.y)

    mask = (X < 0.5) * (Y < 0.5)
    assert np.allclose(model.phi[mask], 1.0, atol=0.5)
    assert np.allclose(model.phi[~mask], 0.0, atol=0.5)


def test_free_surface_set_phi_with_eta_array():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    def eta_func(x, a, b):
        return np.where(x < a, b, 0)

    a = 0.5
    b = 0.5

    X, Y = np.meshgrid(model.x, model.y)

    mask = (X < 0.5) * (Y < 0.5)

    eta_array = np.empty(X.shape[1])
    eta_array = eta_func(X[0, :], a, b)
    model.set_phi_from_eta(eta_array, a, b)
    assert np.allclose(model.phi[mask], 1.0, atol=0.5)
    assert np.allclose(model.phi[~mask], 0.0, atol=0.5)
