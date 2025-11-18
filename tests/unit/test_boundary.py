import numpy as np

from labnewt import (
    BottomWallNoSlip,
    LeftWallNoSlip,
    RightWallNoSlip,
    StencilD2Q9,
    TopWallNoSlip,
)

np.random.seed(42)


def test_left_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    fi = np.random.rand(s.nq, *shape)
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_left
    y_idxs = slice(None)
    x_idxs = 0
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    LeftWallNoSlip().apply(fi, fo, s)

    # Test for unwanted side effects
    assert np.allclose(fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_right_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    fi = np.random.rand(s.nq, *shape)
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_right
    y_idxs = slice(None)
    x_idxs = -1
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    RightWallNoSlip().apply(fi, fo, s)

    # Test for unwanted side effects
    assert np.allclose(fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_bottom_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    fi = np.random.rand(s.nq, *shape)
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_down
    y_idxs = 0
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    BottomWallNoSlip().apply(fi, fo, s)

    # Test for unwanted side effects
    assert np.allclose(fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_top_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    fi = np.random.rand(s.nq, *shape)
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_up
    y_idxs = -1
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    TopWallNoSlip().apply(fi, fo, s)

    # Test for unwanted side effects
    assert np.allclose(fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)
