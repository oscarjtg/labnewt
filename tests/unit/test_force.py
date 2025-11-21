import numpy as np
import pytest

from labnewt import (
    ConstantGravityForce,
    FreeSurfaceModel,
    GravityForce,
    MacroscopicStandard,
    Model,
    StencilD2Q9,
)
from labnewt._moments import _m0, _mx, _my
from labnewt.force import Force


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
    force = ConstantGravityForce(dx, dt, g, (ex, ey))
    assert np.isclose(force.Fx, -0.6, atol=1.0e-12)
    assert np.isclose(force.Fy, 0.8, atol=1.0e-12)


def test_constant_gravity_force_set_direction_after_init():
    dx = 0.1
    dt = 0.1
    g = 10.0
    ex = -3.0
    ey = 4.0
    force = ConstantGravityForce(dx, dt)
    force.set_gravity_direction((ex, ey))
    force.set_gravity_magnitude(g)
    assert np.isclose(force.Fx, -0.6, atol=1.0e-12)
    assert np.isclose(force.Fy, 0.8, atol=1.0e-12)


def test_constant_gravity_force_conserves_moments():
    nx = 10
    ny = 10
    shape = (nx, ny)
    dx = 0.1
    dt = 0.1
    nu = 0.1
    force = ConstantGravityForce(dx, dt)
    macros = MacroscopicStandard()
    s = StencilD2Q9()
    model = Model(nx, ny, dx, dt, nu, stencil=s, macros=macros)

    rng = np.random.default_rng(42)
    f = rng.random((s.nq, *shape))
    r_pre = _m0(f)
    u_pre = _mx(f, s) / r_pre
    v_pre = _my(f, s) / r_pre

    model.set_r(r_pre)
    model.set_u(u_pre)
    model.set_v(v_pre)
    model._set_f(f)

    force.apply(model)
    model.macros.forcing(model)

    f[:] = model.fo
    r_post = _m0(f)
    u_post = (_mx(f, s) - model.Fx) / r_post
    v_post = (_my(f, s) - model.Fy) / r_post

    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)


def test_gravity_force_conserves_moments():
    nx = 10
    ny = 10
    shape = (nx, ny)
    dx = 0.1
    dt = 0.1
    nu = 0.1
    force = GravityForce(dx, dt)
    macros = MacroscopicStandard()
    s = StencilD2Q9()
    model = Model(nx, ny, dx, dt, nu, stencil=s, macros=macros)

    rng = np.random.default_rng(42)
    f = rng.random((s.nq, *shape))
    r_pre = _m0(f)
    u_pre = _mx(f, s) / r_pre
    v_pre = _my(f, s) / r_pre

    model.set_r(r_pre)
    model.set_u(u_pre)
    model.set_v(v_pre)
    model._set_f(f)

    force.apply(model)
    model.macros.forcing(model)

    f[:] = model.fo
    r_post = _m0(f)
    u_post = (_mx(f, s) - model.Fx) / r_post
    v_post = (_my(f, s) - model.Fy) / r_post

    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)


def test_gravity_force_free_surface_model_conserves_moments():
    nx = 10
    ny = 10
    shape = (nx, ny)
    dx = 0.1
    dt = 0.1
    nu = 0.1
    force = GravityForce(dx, dt)
    macros = MacroscopicStandard()
    s = StencilD2Q9()
    model = FreeSurfaceModel(nx, ny, dx, dt, nu, stencil=s, macros=macros)

    rng = np.random.default_rng(42)
    f = rng.random((s.nq, *shape))
    r_pre = _m0(f)
    u_pre = _mx(f, s) / r_pre
    v_pre = _my(f, s) / r_pre
    phi = np.random.random(shape)

    model.set_r(r_pre)
    model.set_u(u_pre)
    model.set_v(v_pre)
    model.set_phi(phi)
    model._set_f(f)

    force.apply(model)
    model.macros.forcing(model)

    f[:] = model.fo
    r_post = _m0(f)
    u_post = (_mx(f, s) - model.Fx) / r_post
    v_post = (_my(f, s) - model.Fy) / r_post

    assert np.allclose(r_pre, r_post, atol=1.0e-12)
    assert np.allclose(u_pre, u_post, atol=1.0e-12)
    assert np.allclose(v_pre, v_post, atol=1.0e-12)


def test_force_apply_raises_notimplemented():
    force = Force()

    class DummyModel:
        pass

    model = DummyModel()
    with pytest.raises(NotImplementedError):
        force.apply(model)
