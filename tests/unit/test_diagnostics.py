import numpy as np

from labnewt.diagnostics import relative_error, average_difference


def test_relative_error_no_difference():
    X_ref = np.ones(10)
    X_com = np.ones(10)
    error = relative_error(X_com, X_ref)
    assert np.isclose(error, 0.0, atol=1.0e-12)


def test_relative_error_large_difference_one_element():
    x_ref = 2.0
    x_com = 4.0
    error = relative_error(x_com, x_ref)
    assert np.isclose(error, 1.0, atol=1.0e-12)


def test_average_difference_no_difference():
    X_ref = np.ones(10)
    X_com = np.ones(10)
    error = average_difference(X_com, X_ref)
    assert np.isclose(error, 0.0, atol=1.0e-12)


def test_average_difference_no_difference_with_mask():
    X_ref = np.ones(10)
    X_com = np.ones(10)
    mask = np.random.random(10) > 0.5
    error = average_difference(X_com, X_ref, mask)
    assert np.isclose(error, 0.0, atol=1.0e-12)
