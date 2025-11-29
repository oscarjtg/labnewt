"""Unit tests for LocalAverageRefiller class."""

import numpy as np
import numpy.testing as npt

from labnewt import FreeSurfaceModel
from labnewt.refiller import LocalAverageRefiller


def test_local_average_refiller_one_pad_all_fluid():
    nx = 4
    ny = 5
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    rng = np.random.default_rng(42)
    model.fi = rng.beta(1.0, 1.0, (model.stencil.nq, *model.shape))
    fi0 = np.copy(model.fi)
    fi_id = id(model.fi)

    mask = np.zeros(model.shape, dtype=bool)
    mask[3, 1] = True

    model.vof.F_mask[...] = True

    refiller = LocalAverageRefiller()
    refiller.fill(model, mask)

    expected = (
        model.fi[:, 2, 0]
        + model.fi[:, 3, 0]
        + model.fi[:, 4, 0]
        + model.fi[:, 4, 1]
        + model.fi[:, 4, 2]
        + model.fi[:, 3, 2]
        + model.fi[:, 2, 2]
        + model.fi[:, 2, 1]
    ) / 8

    # Check correct cell change.
    npt.assert_allclose(model.fi[:, mask].ravel(), expected)

    # Check no unintended changes.
    npt.assert_allclose(model.fi[:, ~mask], fi0[:, ~mask])
    assert fi_id == id(model.fi)


def test_local_average_refiller_one_pad_some_fluid():
    nx = 4
    ny = 5
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    rng = np.random.default_rng(42)
    model.fi = rng.beta(1.0, 1.0, (model.stencil.nq, *model.shape))
    fi0 = np.copy(model.fi)
    fi_id = id(model.fi)

    mask = np.zeros(model.shape, dtype=bool)
    mask[3, 1] = True

    model.vof.F_mask[:3, :] = True
    model.vof.I_mask[3, :] = True
    model.vof.G_mask[4:, :] = True

    refiller = LocalAverageRefiller()
    refiller.fill(model, mask)

    expected = (model.fi[:, 2, 0] + model.fi[:, 2, 2] + model.fi[:, 2, 1]) / 3

    # Check correct cell change.
    npt.assert_allclose(model.fi[:, mask].ravel(), expected)

    # Check no unintended changes.
    npt.assert_allclose(model.fi[:, ~mask], fi0[:, ~mask])
    assert fi_id == id(model.fi)


def test_local_average_refiller_one_pad_no_fluid():
    nx = 4
    ny = 5
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    rng = np.random.default_rng(42)
    model.fi = rng.beta(1.0, 1.0, (model.stencil.nq, *model.shape))
    fi0 = np.copy(model.fi)
    fi_id = id(model.fi)

    mask = np.zeros(model.shape, dtype=bool)
    mask[3, 1] = True

    model.vof.I_mask[0, :] = True
    model.vof.G_mask[1:, :] = True

    r0 = 1.1
    refiller = LocalAverageRefiller(density=r0)
    refiller.fill(model, mask)

    expected = r0 * model.stencil.w

    # Check correct cell change.
    npt.assert_allclose(model.fi[:, mask].ravel(), expected)

    # Check no unintended changes.
    npt.assert_allclose(model.fi[:, ~mask], fi0[:, ~mask])
    assert fi_id == id(model.fi)


def test_local_average_refiller_two_pad_all_fluid():
    nx = 5
    ny = 5
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    rng = np.random.default_rng(42)
    model.fi = rng.beta(1.0, 1.0, (model.stencil.nq, *model.shape))
    fi0 = np.copy(model.fi)
    fi_id = id(model.fi)

    mask = np.zeros(model.shape, dtype=bool)
    mask[2, 2] = True

    model.vof.F_mask[...] = True

    refiller = LocalAverageRefiller(pad=2)
    refiller.fill(model, mask)

    expected = (np.sum(model.fi, axis=(1, 2)) - model.fi[:, 2, 2]) / 24

    # Check correct cell change.
    npt.assert_allclose(model.fi[:, mask].ravel(), expected)

    # Check no unintended changes.
    npt.assert_allclose(model.fi[:, ~mask], fi0[:, ~mask])
    assert fi_id == id(model.fi)


def test_local_average_refiller_two_pad_some_fluid():
    nx = 5
    ny = 6
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = FreeSurfaceModel(nx, ny, dx, dt, nu)

    rng = np.random.default_rng(42)
    model.fi = rng.beta(1.0, 1.0, (model.stencil.nq, *model.shape))
    fi0 = np.copy(model.fi)
    fi_id = id(model.fi)

    mask = np.zeros(model.shape, dtype=bool)
    mask[3, 2] = True

    model.vof.F_mask[:2, :] = True
    model.vof.I_mask[2, :] = True
    model.vof.G_mask[3:, :] = True

    refiller = LocalAverageRefiller(pad=2)
    refiller.fill(model, mask)

    expected = (
        model.fi[:, 1, 0]
        + model.fi[:, 1, 1]
        + model.fi[:, 1, 2]
        + model.fi[:, 1, 3]
        + model.fi[:, 1, 4]
    ) / 5

    # Check correct cell change.
    npt.assert_allclose(model.fi[:, mask].ravel(), expected)

    # Check no unintended changes.
    npt.assert_allclose(model.fi[:, ~mask], fi0[:, ~mask])
    assert fi_id == id(model.fi)
