import numpy as np

from labnewt import ColliderSRT, Macroscopic, StencilD2Q9


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

    fi = np.random.rand(s.nq, *shape)
    fo = np.empty_like(fi)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)

    fi0 = np.copy(fi)
    r0 = np.copy(r)
    u0 = np.copy(u)
    v0 = np.copy(v)

    collider.collide(fo, fi, r, u, v, s)

    # Check no unintended side effects.
    assert np.allclose(fi, fi0, atol=1.0e-12)
    assert np.allclose(r, r0, atol=1.0e-12)
    assert np.allclose(u, u0, atol=1.0e-12)
    assert np.allclose(v, v0, atol=1.0e-12)

    # Check physics.
    assert np.allclose(fo, s.w[:, None, None], atol=1.0e-12)


def test_collider_srt_conserves_moments():
    collider = ColliderSRT(0.01, 0.1, 0.1)
    s = StencilD2Q9()
    macros = Macroscopic()

    nx = 10
    ny = 10
    shape = (nx, ny)

    fi = np.random.rand(s.nq, *shape)
    fi0 = np.copy(fi)
    fo = np.empty_like(fi)
    r_pre, u_pre, v_pre = np.empty(shape), np.empty(shape), np.empty(shape)
    macros.density(r_pre, fi)
    macros.velocity_x(u_pre, r_pre, fi, s)
    macros.velocity_y(v_pre, r_pre, fi, s)

    r0, u0, v0 = np.copy(r_pre), np.copy(u_pre), np.copy(v_pre)

    collider.collide(fo, fi, r0, u0, v0, s)
    r_post, u_post, v_post = np.empty(shape), np.empty(shape), np.empty(shape)
    macros.density(r_post, fo)
    macros.velocity_x(u_post, r_post, fo, s)
    macros.velocity_y(v_post, r_post, fo, s)

    # Check no unintended side effects.
    assert np.allclose(fi, fi0, atol=1.0e-12)
    assert np.allclose(r_pre, r0, atol=1.0 - 12)
    assert np.allclose(u_pre, u0, atol=1.0 - 12)
    assert np.allclose(v_pre, v0, atol=1.0 - 12)

    # Check correct physics.
    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)
