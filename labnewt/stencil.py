import numpy as np


class Stencil:
    pass


class StencilD2Q9(Stencil):
    nq = 9
    ex = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1], dtype=np.int64)
    ey = np.array([0, 0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int64)
    cx = ex.astype(np.float64)
    cy = ey.astype(np.float64)
    w = np.array(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ],
        dtype=np.float64,
    )
    q_rev = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=np.int64)
    q_left = np.array([2, 6, 7], dtype=np.int64)
    q_right = np.array([1, 5, 8], dtype=np.int64)
    q_down = np.array([4, 6, 8], dtype=np.int64)
    q_up = np.array([3, 5, 7], dtype=np.int64)


for arr in (
    StencilD2Q9.ex,
    StencilD2Q9.ey,
    StencilD2Q9.cx,
    StencilD2Q9.cy,
    StencilD2Q9.w,
    StencilD2Q9.q_rev,
    StencilD2Q9.q_left,
    StencilD2Q9.q_right,
    StencilD2Q9.q_down,
    StencilD2Q9.q_up,
):
    arr.flags.writeable = False
