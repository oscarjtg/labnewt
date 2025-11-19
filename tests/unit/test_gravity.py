"""Units tests for Gravity class."""

import numpy as np

from labnewt import Gravity


def test_gravity_default_initialisation():
    """Checks the default initialisation."""
    gravity = Gravity()
    assert np.isclose(gravity.gx, 0.0, atol=1.0e-12)
    assert np.isclose(gravity.gy, -9.81, atol=1.0e-12)


def test_gravity_custom_initialisation():
    """Checks initialising Gravity with given gx, gy."""
    gx, gy = 0.1, -0.1
    gravity = Gravity(gx, gy)
    assert np.isclose(gravity.gx, gx, atol=1.0e-12)
    assert np.isclose(gravity.gy, gy, atol=1.0e-12)


def test_gravity_set_magnitude():
    """Checks that set_gravity() works with given magnitude."""
    gravity = Gravity()
    gravity.set_gravity(magnitude=1.0)
    assert np.isclose(gravity.gx, 0.0, atol=1.0e-12)
    assert np.isclose(gravity.gy, -1.0, atol=1.0e-12)


def test_gravity_set_direction():
    """Checks that set_gravity() works with given direction."""
    gravity = Gravity()
    gravity.set_gravity(direction=(1.0, 0.0))
    assert np.isclose(gravity.gx, 9.81, atol=1.0e-12)
    assert np.isclose(gravity.gy, 0.0, atol=1.0e-12)


def test_set_gravity():
    """Checks set_gravity() with given magnitude and direction."""
    gravity = Gravity()
    gravity.set_gravity(magnitude=0.1, direction=(1.0, 0.0))
    assert np.isclose(gravity.gx, 0.1, atol=1.0e-12)
    assert np.isclose(gravity.gy, 0.0, atol=1.0e-12)


def test_set_gravity_complex():
    """A more complex example of test_set_gravity()."""
    rng = np.random.default_rng(42)
    n = 10
    magnitudes = rng.random(n)
    xs = rng.random(n)
    ys = rng.random(n)
    for mag, x, y in zip(magnitudes, xs, ys):
        gravity = Gravity()
        gravity.set_gravity(magnitude=mag, direction=(x, y))
        # Check magnitude.
        assert np.isclose(mag, np.sqrt(gravity.gx**2 + gravity.gy**2), atol=1.0e-12)
        # Check direction: ratio of components should be same as input.
        assert np.isclose(x / y, gravity.gx / gravity.gy, atol=1.0e-12)
