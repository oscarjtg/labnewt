import numpy as np

from labnewt import Macroscopic, StencilD2Q9
from labnewt._equilibrium import _feq2, _feq2_q

np.random.seed(42)


def test_feq2_array_zero_velocity():
    shape = (2, 3)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    s = StencilD2Q9()

    feq_computed = _feq2(r, u, v, s)
    feq_expected = s.w[:, None, None] * np.ones((s.nq, *shape))

    assert np.allclose(feq_computed, feq_expected)


def test_feq2_macroscopic_properties():
    shape = (4, 5)
    r = np.random.beta(1.0, 1.0, shape)
    u = np.random.normal(0.0, 0.1, shape)
    v = np.random.normal(0.0, 0.2, shape)
    s = StencilD2Q9()
    feq = _feq2(r, u, v, s)
    r1, u1, v1 = np.empty(shape), np.empty(shape), np.empty(shape)
    macros = Macroscopic()
    macros.density(r1, feq)
    macros.velocity_x(u1, r1, feq, s)
    macros.velocity_y(v1, r1, feq, s)

    assert np.allclose(r, r1, atol=1.0e-12)
    assert np.allclose(u, u1, atol=1.0e-12)
    assert np.allclose(v, v1, atol=1.0e-12)


def test_feq2q_against_feq2():
    shape = (4, 5)
    r = np.random.beta(1.0, 1.0, shape)
    u = np.random.normal(0.0, 0.1, shape)
    v = np.random.normal(0.0, 0.2, shape)
    s = StencilD2Q9()
    feq = _feq2(r, u, v, s)

    for q in range(s.nq):
        assert np.allclose(feq[q, ...], _feq2_q(q, r, u, v, s), atol=1.0e-12)
