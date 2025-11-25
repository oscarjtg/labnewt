import numpy as np
import pytest

from labnewt import (
    BottomWallNoSlip,
    LeftWallNoSlip,
    RightWallNoSlip,
    StencilD2Q9,
    TopWallNoSlip,
)
from labnewt.boundary import BoundaryCondition


def test_boundary_condition_protocol():
    bc = BoundaryCondition()

    class EmptyModel:
        pass

    model = EmptyModel()
    with pytest.raises(NotImplementedError):
        bc.apply(model)


class DummyModel:
    def __init__(self, fi, fo, s):
        self.fi = fi
        self.fo = fo
        self.stencil = s


def test_left_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    rng = np.random.default_rng(42)
    fi = rng.random((s.nq, *shape))
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_left
    y_idxs = slice(None)
    x_idxs = 0
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    model = DummyModel(fi, fo, s)
    LeftWallNoSlip().apply(model)

    # Test for unwanted side effects
    assert np.allclose(model.fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = model.fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_right_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    rng = np.random.default_rng(42)
    fi = rng.random((s.nq, *shape))
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_right
    y_idxs = slice(None)
    x_idxs = -1
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    model = DummyModel(fi, fo, s)
    RightWallNoSlip().apply(model)

    # Test for unwanted side effects
    assert np.allclose(model.fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = model.fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_bottom_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    rng = np.random.default_rng(42)
    fi = rng.random((s.nq, *shape))
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_down
    y_idxs = 0
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    model = DummyModel(fi, fo, s)
    BottomWallNoSlip().apply(model)

    # Test for unwanted side effects
    assert np.allclose(model.fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = model.fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)


def test_top_wall_no_slip():
    s = StencilD2Q9()
    nx = 5
    ny = 5
    shape = (ny, nx)
    rng = np.random.default_rng(42)
    fi = rng.random((s.nq, *shape))
    fo = np.copy(fi)
    fo0 = np.copy(fo)

    q_idxs = s.q_up
    y_idxs = -1
    x_idxs = slice(None)
    f_out = {}
    for q in q_idxs:
        f_out[q] = np.copy(fo[q, y_idxs, x_idxs])

    model = DummyModel(fi, fo, s)
    TopWallNoSlip().apply(model)

    # Test for unwanted side effects
    assert np.allclose(model.fo, fo0, atol=1.0e-12)

    # Test for correct boundary condition
    for q in q_idxs:
        f_in = model.fi[s.q_rev[q], y_idxs, x_idxs]
        assert np.allclose(f_in, f_out[q], atol=1.0e-12)
