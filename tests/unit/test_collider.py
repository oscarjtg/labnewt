import numpy as np

from labnewt import ColliderSRT, Model, StencilD2Q9


def test_collider_srt_unit_omega():
    # These values make collider.omega = 1.0
    nu = 0.5 / 3.0
    dx = 1.0
    dt = 1.0
    collider = ColliderSRT(nu, dx, dt)
    assert np.isclose(collider.omega, 1.0, rtol=1.0e-12)


def test_collider_srt_unit_omega_stationary_fluid():
    # These values make collider.omega = 1.0
    nu = 0.5 / 3.0
    dx = 1.0
    dt = 1.0
    collider = ColliderSRT(nu, dx, dt)
    collider.omega = 1.0
    s = StencilD2Q9()

    # Create arrays
    nx = 10
    ny = 10
    shape = (nx, ny)

    model = Model(nx, ny, dx, dt, nu)

    model.fi = np.random.rand(s.nq, *shape)
    model.fo = np.empty_like(model.fi)
    model.r = np.ones(shape)
    model.u = np.zeros(shape)
    model.v = np.zeros(shape)

    fi0 = np.copy(model.fi)
    r0 = np.copy(model.r)
    u0 = np.copy(model.u)
    v0 = np.copy(model.v)

    collider.collide(model)

    # Check no unintended side effects.
    assert np.allclose(model.fi, fi0, atol=1.0e-12)
    assert np.allclose(model.r, r0, atol=1.0e-12)
    assert np.allclose(model.u, u0, atol=1.0e-12)
    assert np.allclose(model.v, v0, atol=1.0e-12)

    # Check physics.
    assert np.allclose(model.fo, s.w[:, None, None], atol=1.0e-12)


def test_collider_srt_conserves_moments():
    collider = ColliderSRT(0.01, 0.1, 0.1)
    s = StencilD2Q9()

    nx = 10
    ny = 10
    shape = (ny, nx)

    model = Model(nx, ny, 1, 1, 1)
    model.fi = np.random.rand(s.nq, *shape)
    fi0 = np.copy(model.fi)

    model.macros.density(model)
    model.macros.velocity_x(model)
    model.macros.velocity_y(model)

    r0, u0, v0 = np.copy(model.r), np.copy(model.u), np.copy(model.v)

    collider.collide(model)

    model.macros.density(model)
    model.macros.velocity_x(model)
    model.macros.velocity_y(model)

    # Check no unintended side effects.
    assert np.allclose(model.fi, fi0, atol=1.0e-12)

    # Check correct physics.
    assert np.allclose(model.r, r0, atol=1.0e-12)
    assert np.allclose(model.u, u0, atol=1.0e-12)
    assert np.allclose(model.v, v0, atol=1.0e-12)
