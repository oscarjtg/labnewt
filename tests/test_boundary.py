import numpy as np

from labnewt import (
    AllWallsNoSlip,
    BottomTopWallsNoSlip,
    BottomWallNoSlip,
    LeftRightWallsNoSlip,
    LeftWallNoSlip,
    RightWallNoSlip,
    StencilD2Q9,
    Streamer,
    TopWallNoSlip,
)

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


def test_left_right_wall_left_no_slip():
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
    LeftRightWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_all_walls_left_no_slip():
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
    AllWallsNoSlip().apply(f, s)

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


def test_left_right_wall_right_no_slip():
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
    LeftRightWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_all_walls_right_no_slip():
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
    AllWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_bottom_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_down
    y_idxs = 0
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    BottomWallNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_bottom_top_walls_bottom_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_down
    y_idxs = 0
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    BottomTopWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_all_walls_bottom_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_down
    y_idxs = 0
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    AllWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_top_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_up
    y_idxs = -1
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    TopWallNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_bottom_top_walls_top_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_up
    y_idxs = -1
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    BottomTopWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_all_walls_top_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    f = np.random.rand(s.nq, *shape)

    q_idxs = s.q_up
    y_idxs = -1
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(f[q, y_idxs, x_idxs])

    f = Streamer().stream(f, s)
    AllWallsNoSlip().apply(f, s)

    for q in q_idxs:
        f_in = f[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)
