import numpy as np
import pytest

from labnewt import MacroscopicGuo, MacroscopicStandard, Model, StencilD2Q9
from labnewt._equilibrium import _feq2, _feq2_q


def test_feq2_float_zero_velocity():
    r = 1.0
    u = 0.0
    v = 0.0
    s = StencilD2Q9()

    computed = _feq2(r, u, v, s).ravel()
    expected = r * s.w

    assert np.allclose(computed, expected, rtol=1.0e-12)


def test_feq2_array_zero_velocity():
    shape = (2, 3)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    s = StencilD2Q9()

    feq_computed = _feq2(r, u, v, s)
    feq_expected = s.w[:, None, None] * np.ones((s.nq, *shape))

    assert np.allclose(feq_computed, feq_expected)


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_feq2_macroscopic_properties(Macroscopic):
    nx, ny = 5, 4
    shape = (ny, nx)
    model = Model(nx, ny, 1, 1, 1, macros=Macroscopic())

    rng = np.random.default_rng(42)
    r0 = rng.beta(1.0, 1.0, shape)
    u0 = rng.normal(0.0, 0.1, shape)
    v0 = rng.normal(0.0, 0.2, shape)

    model.fi = _feq2(r0, u0, v0, model.stencil)

    model.macros.density(model)
    model.macros.velocity_x(model)
    model.macros.velocity_y(model)

    assert np.allclose(model.r, r0, atol=1.0e-12)
    assert np.allclose(model.u, u0, atol=1.0e-12)
    assert np.allclose(model.v, v0, atol=1.0e-12)


def test_feq2q_against_feq2():
    shape = (4, 5)
    rng = np.random.default_rng(42)
    r = rng.beta(1.0, 1.0, shape)
    u = rng.normal(0.0, 0.1, shape)
    v = rng.normal(0.0, 0.2, shape)
    s = StencilD2Q9()
    feq = _feq2(r, u, v, s)

    for q in range(s.nq):
        assert np.allclose(feq[q, ...], _feq2_q(q, r, u, v, s), atol=1.0e-12)
