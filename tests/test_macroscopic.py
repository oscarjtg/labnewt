import numpy as np

from labnewt._macroscopic import _density


def test_density_d2q9_against_known_values():
    nq = 9
    f = np.empty((nq, 2, 2))
    for q in range(nq):
        f[q, 0, 0] = 0.1
        f[q, 1, 0] = 0.1 * (q + 1)
        f[q, 0, 1] = 0.1 if q % 2 == 0 else -0.1
        f[q, 1, 1] = 0.1 * (q + 1) if q % 2 == 0 else -0.1 * (q + 1)

    r_exp = np.empty((2, 2))
    r_exp[0, 0] = 0.9  # 9 * 0.1
    r_exp[1, 0] = 4.5  # 0.5 * 9 * 10 / 10
    r_exp[0, 1] = 0.1
    r_exp[1, 1] = 0.5

    r_com = _density(f)

    assert np.allclose(r_exp, r_com, atol=1.0e-12)
