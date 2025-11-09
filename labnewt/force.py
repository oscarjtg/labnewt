"""Force classes."""

import numpy as np


class ConstantGravityForce:
    def __init__(self, dx, dt, g_magnitude=9.81, ex=0, ey=-1):
        self.dx = dx
        self.dt = dt
        self._set_g_star(g_magnitude)
        self._set_ex_ey(ex, ey)
        self._set_Fx_Fy()

    def _set_g_star(self, g_magnitude):
        self.g_star = np.abs(g_magnitude * self.dt**2 / self.dx)

    def _set_ex_ey(self, ex, ey):
        self.ex = ex / np.sqrt(ex**2 + ey**2)
        self.ey = ey / np.sqrt(ex**2 + ey**2)

    def _set_Fx_Fy(self):
        self.Fx = self.g_star * self.ex
        self.Fy = self.g_star * self.ey

    def set_gravity_magnitude(self, g_magnitude):
        self._set_g_star(g_magnitude)
        self._set_Fx_Fy()

    def set_gravity_direction(self, ex, ey):
        self._set_ex_ey(ex, ey)
        self._set_Fx_Fy()

    def apply_to_distribution(self, f, s):
        f[:] += (
            3.0
            * s.w[:, None, None]
            * (s.cx[:, None, None] * self.Fx + s.cy[:, None, None] * self.Fy)
        ) * self.dt
