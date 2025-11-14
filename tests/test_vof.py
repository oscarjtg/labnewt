import numpy as np

from labnewt._vof import _dMq, _dMq_


class Stencil:
    """Minimal stencil object matching the function's expected attributes."""

    def __init__(self, ex, ey, q_rev):
        self.ex = np.asarray(ex, dtype=int)
        self.ey = np.asarray(ey, dtype=int)
        self.q_rev = np.asarray(q_rev, dtype=int)
        self.nq = len(self.ex)


def _compute_expected(dMq_shape, fo, phi, F_mask, I_mask, G_mask, s):
    """Reference implementation (simple Python loops) to compute expected dMq."""
    nq, ny, nx = dMq_shape
    expected = np.zeros((nq, ny, nx), dtype=fo.dtype)

    for q in range(nq):
        for y in range(ny):
            for x in range(nx):
                y2 = (y + s.ey[q]) % ny
                x2 = (x + s.ex[q]) % nx
                # A logic
                if not I_mask[y, x]:
                    A = 0.0
                elif G_mask[y2, x2]:
                    A = 0.0
                elif I_mask[y2, x2]:
                    A = 0.5 * (phi[y, x] + phi[y2, x2])
                elif F_mask[y2, x2]:
                    A = 1.0
                else:
                    # This should not happen per your assumptions; set to 0 for safety.
                    A = 0.0

                f_here = fo[q, y, x]
                f_nei = fo[s.q_rev[q], y2, x2]
                expected[q, y, x] = A * (f_nei - f_here)

    return expected


def make_masks(ny, nx, pattern="stripes"):
    """
    Helper to build mutually exclusive G/I/F masks satisfying the invariant:
    exactly one of G_mask, I_mask, F_mask is True at each cell.
    - 'stripes' alternates along y
    - 'checker' makes a checkerboard of the three masks
    """
    G_mask = np.zeros((ny, nx), dtype=bool)
    I_mask = np.zeros((ny, nx), dtype=bool)
    F_mask = np.zeros((ny, nx), dtype=bool)

    if pattern == "stripes":
        for y in range(ny):
            if y % 3 == 0:
                G_mask[y, :] = True
            elif y % 3 == 1:
                I_mask[y, :] = True
            else:
                F_mask[y, :] = True
    elif pattern == "checker":
        for y in range(ny):
            for x in range(nx):
                idx = (y + 2 * x) % 3
                if idx == 0:
                    G_mask[y, x] = True
                elif idx == 1:
                    I_mask[y, x] = True
                else:
                    F_mask[y, x] = True
    else:
        raise ValueError("unknown pattern")

    return F_mask, I_mask, G_mask


def test_dMq_basic_values():
    """Basic deterministic test: small grid with known values."""
    nq = 4
    ny = 5
    nx = 6

    # simple stencil: e.g. 4 directions: right, left, down, up
    ex = np.array([1, -1, 0, 0], dtype=int)
    ey = np.array([0, 0, 1, -1], dtype=int)
    q_rev = np.array(
        [1, 0, 3, 2], dtype=int
    )  # pairwise reverse mapping for this stencil

    s = Stencil(ex, ey, q_rev)

    # fo: shape (nq, ny, nx) with simple values increasing for easy hand-check
    fo = np.arange(nq * ny * nx, dtype=float).reshape((nq, ny, nx)) * 0.1

    # phi: some small float pattern
    phi = np.linspace(0.0, 1.0, ny * nx).reshape((ny, nx))

    # masks (mutually exclusive)
    F_mask, I_mask, G_mask = make_masks(ny, nx, pattern="stripes")

    # Keep copies of all read-only arrays for later mutation checks
    fo_copy = fo.copy()
    phi_copy = phi.copy()
    F_copy = F_mask.copy()
    I_copy = I_mask.copy()
    G_copy = G_mask.copy()

    # Run function
    dMq = _dMq(fo, phi, F_mask, I_mask, G_mask, s)

    # compute expected via reference loop
    expected = _compute_expected(dMq.shape, fo, phi, F_mask, I_mask, G_mask, s)

    # Value correctness
    np.testing.assert_allclose(dMq, expected, rtol=1e-12, atol=1e-12)

    # Ensure other arrays were not modified
    np.testing.assert_array_equal(fo, fo_copy)
    np.testing.assert_array_equal(phi, phi_copy)
    np.testing.assert_array_equal(F_mask, F_copy)
    np.testing.assert_array_equal(I_mask, I_copy)
    np.testing.assert_array_equal(G_mask, G_copy)


def test_only_dMq_modified_and_shape_preserved():
    """
    Ensure only dMq is modified by _dMq_.

    Other arrays keep their original identity and content.
    """
    nq = 3
    ny = 4
    nx = 5

    ex = np.array([0, 1, -1], dtype=int)
    ey = np.array([1, 0, 0], dtype=int)
    q_rev = np.array([2, 0, 1], dtype=int)
    s = Stencil(ex, ey, q_rev)

    rng = np.random.default_rng(12345)
    fo = rng.random((nq, ny, nx)).astype(float)
    phi = rng.random((ny, nx)).astype(float)
    F_mask, I_mask, G_mask = make_masks(ny, nx, pattern="checker")

    dMq = np.zeros((nq, ny, nx), dtype=float)

    # make deep copies to check for mutation
    fo_copy = fo.copy()
    phi_copy = phi.copy()
    F_copy = F_mask.copy()
    I_copy = I_mask.copy()
    G_copy = G_mask.copy()

    # Also keep object ids for arrays (they should stay the same objects)
    id_fo = id(fo)
    id_phi = id(phi)
    id_F = id(F_mask)
    id_I = id(I_mask)
    id_G = id(G_mask)

    # Call
    _dMq_(dMq, fo, phi, F_mask, I_mask, G_mask, s)

    # dMq should now be non-zero in general (so changed from initial zeros)
    assert not np.all(dMq == 0.0)

    # The read-only arrays must be equal to their copies (no mutation)
    np.testing.assert_array_equal(fo, fo_copy)
    np.testing.assert_array_equal(phi, phi_copy)
    np.testing.assert_array_equal(F_mask, F_copy)
    np.testing.assert_array_equal(I_mask, I_copy)
    np.testing.assert_array_equal(G_mask, G_copy)

    # IDs (objects) should also be unchanged
    assert id(fo) == id_fo
    assert id(phi) == id_phi
    assert id(F_mask) == id_F
    assert id(I_mask) == id_I
    assert id(G_mask) == id_G


def test_dMq_against_loop_randomized():
    """Randomized test comparing the NumPy implementation to a Python-loop reference."""
    nq = 5
    ny = 7
    nx = 6

    rng = np.random.default_rng(2021)
    ex = rng.integers(-2, 3, size=nq)
    ey = rng.integers(-2, 3, size=nq)

    # produce a random permutation for q_rev
    q_rev = np.arange(nq)
    rng.shuffle(q_rev)

    s = Stencil(ex, ey, q_rev)

    fo = rng.normal(size=(nq, ny, nx)).astype(float)
    phi = rng.random((ny, nx)).astype(float)

    # create masks ensuring exclusivity; use stripes to be deterministic
    F_mask, I_mask, G_mask = make_masks(ny, nx, pattern="stripes")

    # copy read-only arrays
    fo_copy = fo.copy()
    phi_copy = phi.copy()
    F_copy = F_mask.copy()
    I_copy = I_mask.copy()
    G_copy = G_mask.copy()

    # run function
    dMq = _dMq(fo, phi, F_mask, I_mask, G_mask, s)

    # compute expected
    expected = _compute_expected(dMq.shape, fo, phi, F_mask, I_mask, G_mask, s)

    # compare
    np.testing.assert_allclose(dMq, expected, rtol=1e-12, atol=1e-12)

    # verify read-only arrays unchanged
    np.testing.assert_array_equal(fo, fo_copy)
    np.testing.assert_array_equal(phi, phi_copy)
    np.testing.assert_array_equal(F_mask, F_copy)
    np.testing.assert_array_equal(I_mask, I_copy)
    np.testing.assert_array_equal(G_mask, G_copy)
