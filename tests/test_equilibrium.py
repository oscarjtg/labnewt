import numpy as np

from labnewt import StencilD2Q9
from labnewt._equilibrium import feq2


def test_feq2_array_zero_velocity():
    shape = (10, 10)
    r = np.ones(shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    s = StencilD2Q9()

    feq_computed = feq2(r, u, v, s)
    feq_expected = s.w[:, None, None] * np.ones((9, *shape))

    assert np.allclose(feq_computed, feq_expected)
