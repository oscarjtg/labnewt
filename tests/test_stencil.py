import numpy as np

from labnewt import StencilD2Q9


def test_stencil_d2q9_zeroth_moment():
    """Tests that sum_i w_i = 1"""
    s = StencilD2Q9()
    assert np.isclose(np.sum(s.w), 1.0, atol=1.0e-12)


def test_stencil_d2q9_first_moments():
    """Tests that sum_i w_i c_ia = 0"""
    s = StencilD2Q9()
    mx = np.sum(s.w * s.cx)
    my = np.sum(s.w * s.cy)
    assert np.isclose(mx, 0.0, atol=1.0e-12)
    assert np.isclose(my, 0.0, atol=1.0e-12)


def test_stencil_d2q9_second_moments():
    """Tests that sum_i w_i c_ia c_ib = cs^2 delta_ab"""
    s = StencilD2Q9()
    mxx = np.sum(s.w * s.cx * s.cx)
    mxy = np.sum(s.w * s.cx * s.cy)
    myy = np.sum(s.w * s.cy * s.cy)
    assert np.isclose(mxx, 1.0 / 3.0, atol=1.0e-12)
    assert np.isclose(mxy, 0.0, atol=1.0e-12)
    assert np.isclose(myy, 1.0 / 3.0, atol=1.0e-12)


def test_stencil_d2q9_third_moments():
    """Tests that sum_i w_i c_ia c_ib c_ic = 0"""
    s = StencilD2Q9()
    mxxx = np.sum(s.w * s.cx * s.cx * s.cx)
    mxxy = np.sum(s.w * s.cx * s.cx * s.cy)
    mxyy = np.sum(s.w * s.cx * s.cy * s.cy)
    myyy = np.sum(s.w * s.cy * s.cy * s.cy)
    assert np.isclose(mxxx, 0.0, atol=1.0e-12)
    assert np.isclose(mxxy, 0.0, atol=1.0e-12)
    assert np.isclose(mxyy, 0.0, atol=1.0e-12)
    assert np.isclose(myyy, 0.0, atol=1.0e-12)


def test_stencil_d2q9_fourth_moments():
    """
    Tests that
    sum_i w_i c_ia c_ib c_ic c_id
    =
    cs^4 * (d_ab * d_cd + d_ac * d_bd + d_ad * d_bc)
    """
    s = StencilD2Q9()
    mxxxx = np.sum(s.w * s.cx * s.cx * s.cx * s.cx)
    mxxxy = np.sum(s.w * s.cx * s.cx * s.cx * s.cy)
    mxxyy = np.sum(s.w * s.cx * s.cx * s.cy * s.cy)
    mxyyy = np.sum(s.w * s.cx * s.cy * s.cy * s.cy)
    myyyy = np.sum(s.w * s.cy * s.cy * s.cy * s.cy)
    assert np.isclose(mxxxx, 1.0 / 3.0, atol=1.0e-12)
    assert np.isclose(mxxxy, 0.0, atol=1.0e-12)
    assert np.isclose(mxxyy, 1.0 / 9.0, atol=1.0e-12)
    assert np.isclose(mxyyy, 0.0, atol=1.0e-12)
    assert np.isclose(myyyy, 1.0 / 3.0, atol=1.0e-12)


def test_stencil_d2q9_fifth_moments():
    """Tests that sum_i w_id c_ia c_ib c_ic c_id c_ie = 0"""
    s = StencilD2Q9()
    mxxxxx = np.sum(s.w * s.cx * s.cx * s.cx * s.cx * s.cx)
    mxxxxy = np.sum(s.w * s.cx * s.cx * s.cx * s.cx * s.cy)
    mxxxyy = np.sum(s.w * s.cx * s.cx * s.cx * s.cy * s.cy)
    mxxyyy = np.sum(s.w * s.cx * s.cx * s.cy * s.cy * s.cy)
    mxyyyy = np.sum(s.w * s.cx * s.cy * s.cy * s.cy * s.cy)
    myyyyy = np.sum(s.w * s.cy * s.cy * s.cy * s.cy * s.cy)
    assert np.isclose(mxxxxx, 0.0, atol=1.0e-12)
    assert np.isclose(mxxxxy, 0.0, atol=1.0e-12)
    assert np.isclose(mxxxyy, 0.0, atol=1.0e-12)
    assert np.isclose(mxxyyy, 0.0, atol=1.0e-12)
    assert np.isclose(mxyyyy, 0.0, atol=1.0e-12)
    assert np.isclose(myyyyy, 0.0, atol=1.0e-12)


def test_stencil_d2q9_qrev():
    s = StencilD2Q9()
    for q in range(s.nq):
        assert s.ex[q] + s.ex[s.q_rev[q]] == 0
        assert s.ey[q] + s.ey[s.q_rev[q]] == 0


def test_stencil_d2q9_qleft():
    s = StencilD2Q9()
    for q in range(s.nq):
        if q in s.q_left:
            assert s.ex[q] == -1
        else:
            assert s.ex[q] != -1


def test_stencil_d2q9_qright():
    s = StencilD2Q9()
    for q in range(s.nq):
        if q in s.q_right:
            assert s.ex[q] == 1
        else:
            assert s.ex[q] != 1


def test_stencil_d2q9_qdown():
    s = StencilD2Q9()
    for q in range(s.nq):
        if q in s.q_down:
            assert s.ey[q] == -1
        else:
            assert s.ey[q] != -1


def test_stencil_d2q9_qup():
    s = StencilD2Q9()
    for q in range(s.nq):
        if q in s.q_up:
            assert s.ey[q] == 1
        else:
            assert s.ey[q] != 1
