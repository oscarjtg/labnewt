import math

import numpy as np
import numpy.testing as npt

from labnewt._vof import _normals, _normals_


def test_normals_inplace_constant_x_gradient():
    """
    phi varies only in x (linear ramp from 0..1). The unit normal should point
    in -x direction on interface cells (nx == -1, ny == 0). Non-interface cells
    must be NaN.
    """
    ny, nx = 5, 7
    # phi increases along x from 0..1
    phi = np.tile(np.linspace(0.0, 1.0, nx, dtype=float), (ny, 1))
    I_mask = (phi > 0.0) & (phi < 1.0)

    norm_x = np.full_like(phi, fill_value=-12345.0, dtype=float)
    norm_y = np.full_like(phi, fill_value=-9999.0, dtype=float)

    # keep copies of read-only arrays
    phi_copy = phi.copy()
    I_mask_copy = I_mask.copy()
    id_phi = id(phi)
    id_I = id(I_mask)

    ret = _normals_(norm_x, norm_y, phi, I_mask)  # default dx=dy=1.0

    # function returns None
    assert ret is None

    # interior interface cells (I_mask True) should have nx == -1, ny == 0
    assert np.allclose(norm_x[I_mask], -1.0, rtol=1e-10, atol=1e-10)
    assert np.allclose(norm_y[I_mask], 0.0, rtol=1e-10, atol=1e-10)

    # outside interface should be NaN
    assert np.all(np.isnan(norm_x[~I_mask]))
    assert np.all(np.isnan(norm_y[~I_mask]))

    # phi and I_mask must not be modified (content and identity)
    npt.assert_array_equal(phi, phi_copy)
    npt.assert_array_equal(I_mask, I_mask_copy)
    assert id(phi) == id_phi
    assert id(I_mask) == id_I


def test_normals_constant_x_gradient():
    """
    phi varies only in x (linear ramp from 0..1). The unit normal should point
    in -x direction on interface cells (nx == -1, ny == 0). Non-interface cells
    must be NaN.
    """
    ny, nx = 5, 7
    # phi increases along x from 0..1
    phi = np.tile(np.linspace(0.0, 1.0, nx, dtype=float), (ny, 1))
    I_mask = (phi > 0.0) & (phi < 1.0)

    # keep copies of read-only arrays
    phi_copy = phi.copy()
    I_mask_copy = I_mask.copy()
    id_phi = id(phi)
    id_I = id(I_mask)

    norm_x, norm_y = _normals(phi, I_mask)  # default dx=dy=1.0

    # interior interface cells (I_mask True) should have nx == -1, ny == 0
    assert np.allclose(norm_x[I_mask], -1.0, rtol=1e-10, atol=1e-10)
    assert np.allclose(norm_y[I_mask], 0.0, rtol=1e-10, atol=1e-10)

    # outside interface should be NaN
    assert np.all(np.isnan(norm_x[~I_mask]))
    assert np.all(np.isnan(norm_y[~I_mask]))

    # phi and I_mask must not be modified (content and identity)
    npt.assert_array_equal(phi, phi_copy)
    npt.assert_array_equal(I_mask, I_mask_copy)
    assert id(phi) == id_phi
    assert id(I_mask) == id_I


def test_normals_oblique_gradient_all_interface():
    """
    phi has a constant oblique gradient; produce phi values in a range
    that keeps I_mask True everywhere. Check that normalized vector equals
    -grad/|grad|.
    """
    ny, nx = 8, 9
    # choose small slopes so phi stays in (0,1) everywhere after offset
    ax = 0.01
    ay = 0.02
    # build coordinate grids
    ys = np.arange(ny)[:, None]
    xs = np.arange(nx)[None, :]
    phi_raw = ax * xs + ay * ys
    # normalize phi_raw to range (0.2, 0.8) to ensure I_mask True everywhere
    minv = phi_raw.min()
    maxv = phi_raw.max()
    phi = 0.2 + 0.6 * (phi_raw - minv) / (maxv - minv)
    assert np.all((phi > 0.0) & (phi < 1.0))  # ensure interface everywhere
    I_mask = np.ones_like(phi, dtype=bool)

    # compute expected gradients using central difference (dx=dy=1)
    # for this linear field, gradients are constant; compute analytically
    # Note: because we normalized phi, the true gradient is scaled:
    scale = 0.6 / (maxv - minv)
    grad_x = ax * scale
    grad_y = ay * scale

    expected_mag = math.hypot(grad_x, grad_y)
    expected_nx = -grad_x / expected_mag
    expected_ny = -grad_y / expected_mag

    norm_x, norm_y = _normals(phi, I_mask, dx=1.0, dy=1.0)

    # All entries are interface: compare entire arrays
    npt.assert_allclose(norm_x, expected_nx, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(norm_y, expected_ny, rtol=1e-10, atol=1e-10)


def test_only_outputs_modified_and_shapes_preserved():
    """
    Ensure that only the norm_x and norm_y arrays are modified and the other inputs
    (phi, I_mask) are not changed or replaced.
    """
    ny, nx = 6, 6
    # create random phi but keep values so some are exactly 0/1 and others in (0,1)
    rng = np.random.default_rng(42)
    # build phi in (0,1) then force border to 0 or 1 to create non-interface cells
    phi = rng.random((ny, nx))
    phi[0, :] = 0.0
    phi[-1, :] = 1.0
    I_mask = (phi > 0.0) & (phi < 1.0)

    norm_x = np.full_like(phi, fill_value=np.nan, dtype=float)
    norm_y = np.full_like(phi, fill_value=np.nan, dtype=float)

    # copies for mutation checks
    phi_copy = phi.copy()
    I_copy = I_mask.copy()
    id_phi = id(phi)
    id_I = id(I_mask)

    # sentinel values in outputs to ensure they get overwritten where appropriate
    norm_x.fill(-1.0)
    norm_y.fill(-2.0)

    _normals_(norm_x, norm_y, phi, I_mask)

    # shapes preserved
    assert norm_x.shape == (ny, nx)
    assert norm_y.shape == (ny, nx)

    # phi and I_mask unchanged in content and identity
    npt.assert_array_equal(phi, phi_copy)
    npt.assert_array_equal(I_mask, I_copy)
    assert id(phi) == id_phi
    assert id(I_mask) == id_I

    # outputs: where I_mask False -> NaN, where True -> finite numbers and normalized
    assert np.all(np.isnan(norm_x[~I_mask]))
    assert np.all(np.isnan(norm_y[~I_mask]))

    # For interface cells, check normalization: sqrt(nx^2 + ny^2) == 1, within tolerance
    mags = np.sqrt(norm_x[I_mask] ** 2 + norm_y[I_mask] ** 2)
    npt.assert_allclose(mags, np.ones_like(mags), rtol=1e-10, atol=1e-10)
