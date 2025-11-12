import numpy as np

from labnewt import ColliderSRT, StencilD2Q9
from labnewt._moments import _density, _velocity_x, _velocity_y


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
    collider.collide(fi, fo, r, u, v, s)
    assert np.allclose(fo, s.w[:, None, None], atol=1.0e-12)


def test_collider_srt_conserves_moments():
    collider = ColliderSRT(0.01, 0.1, 0.1)
    s = StencilD2Q9()

    nx = 10
    ny = 10
    shape = (nx, ny)

    fi = np.random.rand(s.nq, *shape)
    fo = np.empty_like(fi)
    r_pre = _density(fi)
    u_pre = _velocity_x(fi, r_pre, s)
    v_pre = _velocity_y(fi, r_pre, s)

    collider.collide(fi, fo, r_pre, u_pre, v_pre, s)
    r_post = _density(fo)
    u_post = _velocity_x(fo, r_post, s)
    v_post = _velocity_y(fo, r_post, s)

    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)
