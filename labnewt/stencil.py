import numpy as np


class StencilD2Q9:
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


for arr in (
    StencilD2Q9.ex,
    StencilD2Q9.ey,
    StencilD2Q9.cx,
    StencilD2Q9.cy,
    StencilD2Q9.w,
):
    arr.flags.writeable = False
