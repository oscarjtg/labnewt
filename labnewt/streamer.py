from ._shift import periodic_shift


class Streamer:
    def stream(self, fi, fo, s):
        """
        Streams populations in array fi by directions s.ex and s.ey to array fo.

            fo[q, y, x] = fi[q, y - s.ey[q], x - s.ex[q]]

        Modifies fo in-place. Does not change fi.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array containing f_in[q, y, x]

        fo : np.ndarray
            Three-dimensional numpy array containing f_out[q, y, x]

        s : Stencil
            Lattice stencil
        """
        for q in range(s.nq):
            fo[q, :, :] = periodic_shift(fi[q, :, :], s, q)
