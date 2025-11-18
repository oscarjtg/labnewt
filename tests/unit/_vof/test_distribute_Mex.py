import numpy as np
import numpy.testing as npt

from labnewt import StencilD2Q9
from labnewt._vof import _distribute_Mex


def distribute_M_reference(Mstar, rho, norm_x, norm_y, I_mask, s):
    ny, nx = Mstar.shape
    M = Mstar.astype(float).copy()

    q = len(s.ex)
    for y in range(ny):
        for x in range(nx):
            if not I_mask[y, x]:
                continue
            mstar = Mstar[y, x]
            # negative case
            if mstar < 0:
                M_ex = -mstar
                recipients = []
                for i in range(q):
                    dot = s.ex[i] * norm_x[y, x] + s.ey[i] * norm_y[y, x]
                    if dot < 0:
                        yy = y + s.ey[i]
                        xx = x + s.ex[i]
                        if 0 <= yy < ny and 0 <= xx < nx:
                            recipients.append((yy, xx))
                if len(recipients) == 0:
                    continue
                share = M_ex / len(recipients)
                for yy, xx in recipients:
                    M[yy, xx] -= share
                M[y, x] = mstar + M_ex
            # positive case
            elif mstar > rho[y, x]:
                M_ex = mstar - rho[y, x]
                recipients = []
                for i in range(q):
                    dot = s.ex[i] * norm_x[y, x] + s.ey[i] * norm_y[y, x]
                    if dot > 0:
                        yy = y + s.ey[i]
                        xx = x + s.ex[i]
                        if 0 <= yy < ny and 0 <= xx < nx:
                            recipients.append((yy, xx))
                if len(recipients) == 0:
                    continue
                share = M_ex / len(recipients)
                for yy, xx in recipients:
                    M[yy, xx] += share
                M[y, x] = mstar - M_ex
            else:
                continue
    return M


def test_simple_negative_case():
    Mstar = np.zeros((3, 3), dtype=float)
    rho = np.ones_like(Mstar) * 0.5
    Mstar[1, 1] = -2.0
    norm_x = np.zeros_like(Mstar)
    norm_y = np.zeros_like(Mstar)
    norm_x[1, 1] = -1.0
    s = StencilD2Q9()
    I_mask = np.zeros_like(Mstar, dtype=bool)
    I_mask[1, 1] = True

    M = _distribute_Mex(Mstar, rho, norm_x, norm_y, I_mask, s)
    M_ref = distribute_M_reference(Mstar, rho, norm_x, norm_y, I_mask, s)
    # Check against reference implementation
    npt.assert_allclose(M, M_ref)
    # Check that mass was conserved
    npt.assert_almost_equal(np.sum(Mstar), np.sum(M))


def test_simple_positive_case():
    Mstar = np.zeros((3, 3), dtype=float)
    rho = np.ones_like(Mstar) * 1.0
    Mstar[1, 1] = 5.0
    norm_x = np.zeros_like(Mstar)
    norm_y = np.zeros_like(Mstar)
    norm_x[1, 1] = 1.0 / np.sqrt(2)
    norm_y[1, 1] = 1.0 / np.sqrt(2)
    s = StencilD2Q9()
    I_mask = np.zeros_like(Mstar, dtype=bool)
    I_mask[1, 1] = True

    M = _distribute_Mex(Mstar, rho, norm_x, norm_y, I_mask, s)
    M_ref = distribute_M_reference(Mstar, rho, norm_x, norm_y, I_mask, s)
    # Check against reference implementation
    npt.assert_allclose(M, M_ref)
    # Check that mass was conserved
    npt.assert_almost_equal(np.sum(Mstar), np.sum(M))


def test_inputs_not_modified():
    Mstar = np.zeros((3, 3), dtype=float)
    rho = np.ones_like(Mstar) * 1.0
    Mstar[1, 1] = 5.0
    norm_x = np.zeros_like(Mstar)
    norm_y = np.zeros_like(Mstar)
    norm_x[1, 1] = 1.0
    norm_y[1, 1] = 1.0
    s = StencilD2Q9()
    I_mask = np.zeros_like(Mstar, dtype=bool)
    I_mask[1, 1] = True

    copies = [arr.copy() for arr in (Mstar, rho, norm_x, norm_y, I_mask)]
    _ = _distribute_Mex(Mstar, rho, norm_x, norm_y, I_mask, s)
    for orig, copy in zip((Mstar, rho, norm_x, norm_y, I_mask), copies):
        assert np.array_equal(orig, copy), "An input array was modified by distribute_M"
