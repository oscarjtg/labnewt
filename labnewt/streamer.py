import numpy as np

from ._shift import periodic_shift


class Streamer:
    def stream(self, f, s):
        for q in range(s.nq):
            f[q, :, :] = periodic_shift(f[q, :, :], s, q)
        return f
