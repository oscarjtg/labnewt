import numpy as np

from labnewt import StencilD2Q9, ColliderSRT


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

    # Create arrays
    nx = 10
    ny = 10
    shape = (nx, ny)
    f = np.random.rand(*shape)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    s = StencilD2Q9()
    f_coll = collider.collide(f, r, u, v, s)
    assert np.allclose(f_coll, s.w[:, None, None], atol=1.0e-12)
