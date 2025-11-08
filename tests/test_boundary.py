import numpy as np

from labnewt import LeftWallNoSlip, RightWallNoSlip, StencilD2Q9, Streamer

np.random.seed(42)


def test_left_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_left
    y_idxs = slice(None)
    x_idxs = 0
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    LeftWallNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_right_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_right
    y_idxs = slice(None)
    x_idxs = -1
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    RightWallNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)
