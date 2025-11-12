import numpy as np

from labnewt import StencilD2Q9
from labnewt._equilibrium import _feq2
from labnewt._moments import _density, _velocity_x, _velocity_y

np.random.seed(42)


def test_feq2_array_zero_velocity():
    shape = (10, 10)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    s = StencilD2Q9()

    feq_computed = _feq2(r, u, v, s)
    feq_expected = s.w[:, None, None] * np.ones((s.nq, *shape))

    assert np.allclose(feq_computed, feq_expected)


def test_feq2_macroscopic_properties():
    shape = (10, 10)
    r = np.random.beta(1.0, 1.0, shape)
    u = np.random.normal(0.0, 0.1, shape)
    v = np.random.normal(0.0, 0.2, shape)
    s = StencilD2Q9()

    feq = _feq2(r, u, v, s)
    r1 = _density(feq)
    u1 = _velocity_x(feq, r1, s)
    v1 = _velocity_y(feq, r1, s)

    assert np.allclose(r, r1, atol=1.0e-12)
    assert np.allclose(u, u1, atol=1.0e-12)
    assert np.allclose(v, v1, atol=1.0e-12)
