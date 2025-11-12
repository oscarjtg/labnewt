from ._shift import periodic_shift


class Streamer:
    def stream(self, fi, fo, s):
        """
        Streams populations in array fo by directions s.ex and s.ey to array fi.

            fi[q, y, x] = fo[q, y - s.ey[q], x - s.ex[q]]

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

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
            fi[q, :, :] = periodic_shift(fo[q, :, :], s, q)
