from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from labnewt import MacroscopicGuo, MacroscopicStandard, Model


def test_model_set_u_func():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return a * x + b * y

    a = 1.0
    b = 10.0
    model.set_u(func, a, b)

    u_expected = np.empty_like(model.u)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            u_expected[j, i] = func(x, y, a, b)

    assert np.allclose(model.u, u_expected, atol=1.0e-12)


def test_model_set_u_array():
    nx = 9
    ny = 9
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu)

    def func(x, y, a, b):
        return a * x + b * y

    a = 1.0
    b = 10.0

    u_expected = np.empty_like(model.u)
    for i in range(nx):
        for j in range(ny):
            x = dx * (0.5 + i)
            y = dx * (0.5 + j)
            u_expected[j, i] = func(x, y, a, b)

    model.set_u(u_expected)

    assert np.allclose(model.u, u_expected, atol=1.0e-12)


def test_model_plot_savefig(tmp_path):
    save_path = tmp_path / "plots" / "demoplot.png"
    model = Model(5, 5, 0.1, 0.1, 0.01)

    with patch("matplotlib.pyplot.savefig") as mock_save:
        model.plot_fields(path=str(save_path))
        mock_save.assert_called_once()
        called_path = mock_save.call_args[0][0]
        assert str(save_path) == called_path or called_path.endswith("demoplot.png")

    plt.close("all")


@pytest.mark.parametrize(
    "Macroscopic",
    [MacroscopicStandard, MacroscopicGuo],
    ids=["MacroscopicStandard", "MacroscopicGuo"],
)
def test_model_quiescent(Macroscopic):
    """
    Checks that quiescent fluid remains quiescent after 100 timesteps.
    Also (implicitly) tests that arrays are created with the correct shape.
    """
    nx = 8
    ny = 12
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu, macros=Macroscopic())
    shape = (ny, nx)
    model.set_r(np.ones(shape))
    model.set_u(np.zeros(shape))
    model.set_v(np.zeros(shape))
    model.initialise()

    # Check after one timestep.
    model.step()
    assert np.allclose(model.r, np.ones(shape), atol=1.0e-12)
    assert np.allclose(model.u, np.zeros(shape), atol=1.0e-12)
    assert np.allclose(model.v, np.zeros(shape), atol=1.0e-12)

    # Check again after 100 timesteps.
    for _ in range(100):
        model.step()
    assert np.allclose(model.r, np.ones(shape), atol=1.0e-12)
    assert np.allclose(model.u, np.zeros(shape), atol=1.0e-12)
    assert np.allclose(model.v, np.zeros(shape), atol=1.0e-12)
