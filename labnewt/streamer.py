import numpy as np


class Streamer:
    def stream(self, f, s):
        for q in range(s.nq):
            f[q, :, :] = np.roll(f[q, :, :], s.ey[q], axis=0)
            f[q, :, :] = np.roll(f[q, :, :], s.ex[q], axis=1)
        return f
