"""Units tests for UniformRefiller class."""

import numpy as np
import numpy.testing as npt

from labnewt import Model
from labnewt.refiller import UniformRefiller


def test_uniform_refiller_basic_example():
    rho = 1.0
    vx = 0.0
    vy = 0.0
    refiller = UniformRefiller(rho, vx, vy)

    nx = 2
    ny = 3
    dx = 1.0
    dt = 1.0
    nu = 1.0
    model = Model(nx, ny, dx, dt, nu)

    r0 = 1.1
    u0 = 0.1
    v0 = 0.1
    model.set_r((lambda x, y: r0))
    model.set_u((lambda x, y: u0))
    model.set_v((lambda x, y: v0))
    model._set_f(lambda x, y: 0.0)

    r_orig = np.copy(model.r)
    u_orig = np.copy(model.u)
    v_orig = np.copy(model.v)
    fi_orig = np.copy(model.fi)

    r_id = id(model.r)
    u_id = id(model.u)
    v_id = id(model.v)
    fi_id = id(model.fi)

    needs_refilling = np.zeros((ny, nx), dtype=bool)
    needs_refilling[1, 0] = True
    needs_refilling[2, 1] = True

    refiller.fill(model, needs_refilling)

    # Check refilled cells.
    npt.assert_allclose(model.r[needs_refilling], np.array([rho, rho]))
    npt.assert_allclose(model.u[needs_refilling], np.array([vx, vx]))
    npt.assert_allclose(model.v[needs_refilling], np.array([vy, vy]))
    npt.assert_allclose(
        model.fi[:, needs_refilling],
        np.broadcast_to(model.stencil.w[:, np.newaxis], (9, 2)),
    )

    # Check cells that shouldn't have changed.
    npt.assert_allclose(model.r[~needs_refilling], r_orig[~needs_refilling])
    npt.assert_allclose(model.u[~needs_refilling], u_orig[~needs_refilling])
    npt.assert_allclose(model.v[~needs_refilling], v_orig[~needs_refilling])
    npt.assert_allclose(model.fi[:, ~needs_refilling], fi_orig[:, ~needs_refilling])

    # Check that arrays were modified in-place.
    assert r_id == id(model.r)
    assert u_id == id(model.u)
    assert v_id == id(model.v)
    assert fi_id == id(model.fi)
