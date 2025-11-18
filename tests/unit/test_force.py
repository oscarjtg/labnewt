import numpy as np

from labnewt import ConstantGravityForce, Macroscopic, StencilD2Q9
from labnewt._moments import _m0, _mx, _my


def test_constant_gravity_force_default():
    dx = 1.0
    dt = 1.0
    force = ConstantGravityForce(dx, dt)
    assert np.isclose(force.Fx, 0.0, atol=1.0e-12)
    assert np.isclose(force.Fy, -9.81, atol=1.0e-12)


def test_constant_gravity_force_set_magnitude_on_init():
    dx = 0.1
    dt = 0.1
    g = 10.0
    force = ConstantGravityForce(dx, dt, g)
    assert np.isclose(force.Fx, 0.0, atol=1.0e-12)
    assert np.isclose(force.Fy, -1.0, atol=1.0e-12)


def test_constant_gravity_force_set_magnitude_after_init():
    dx = 0.1
    dt = 0.1
    g = 10.0
    force = ConstantGravityForce(dx, dt)
    force.set_gravity_magnitude(g)
    assert np.isclose(force.Fx, 0.0, atol=1.0e-12)
    assert np.isclose(force.Fy, -1.0, atol=1.0e-12)


def test_constant_gravity_force_set_direction_on_init():
    dx = 0.1
    dt = 0.1
    g = 10.0
    ex = -3.0
    ey = 4.0
    force = ConstantGravityForce(dx, dt, g, ex, ey)
    assert np.isclose(force.Fx, -0.6, atol=1.0e-12)
    assert np.isclose(force.Fy, 0.8, atol=1.0e-12)


def test_constant_gravity_force_set_direction_after_init():
    dx = 0.1
    dt = 0.1
    g = 10.0
    ex = -3.0
    ey = 4.0
    force = ConstantGravityForce(dx, dt)
    force.set_gravity_direction(ex, ey)
    force.set_gravity_magnitude(g)
    assert np.isclose(force.Fx, -0.6, atol=1.0e-12)
    assert np.isclose(force.Fy, 0.8, atol=1.0e-12)


def test_constant_gravity_force_conserves_moments():
    dx = 0.1
    dt = 0.1
    force = ConstantGravityForce(dx, dt)
    macros = Macroscopic()
    s = StencilD2Q9()

    nx = 10
    ny = 10
    shape = (nx, ny)

    f = np.random.rand(s.nq, *shape)
    r_pre = _m0(f)
    u_pre = _mx(f, s) / r_pre
    v_pre = _my(f, s) / r_pre

    force.apply(f, s, macros)
    r_post = _m0(f)
    u_post = (_mx(f, s) - force.Fx) / r_post
    v_post = (_my(f, s) - force.Fy) / r_post

    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)
