from ._moments import _m0, _mx, _my


class Macroscopic:
    def density(self, r, f):
        """
        Calculates density from distribution functions.

        Modifies `r` in place. Does not change `f`.

        Parameters
        ----------
        r : np.ndarray
            Two dimensional numpy array of rho[y, x].
        f : np.ndarray
            Three dimensional numpy array of f[q, y, x].
        """
        r[:] = _m0(f)

    def velocity_x(self, u, r, f, s):
        """
        Calculates velocity x-component from distribution functions.

        Modifies `u` in place. Does not change `r`, `f` or `s`.

        Parameters
        ----------
        u : np.ndarray
            Two dimensional numpy array of u[y, x].
        r : np.ndarray
            Two dimensional numpy array of r[y, x].
        f : np.ndarray
            Three dimensional numpy array of f[q, y, x].
        s : Stencil
            Lattice stencil.
        """
        u[:] = _mx(f, s) / r

    def velocity_y(self, v, r, f, s):
        """
        Calculates velocity y-component from distribution functions.

        Modifies `v` in place. Does not change `f` or `s`.

        Parameters
        ----------
        u : np.ndarray
            Two dimensional numpy array of u[y, x].
        r : np.ndarray
            Two dimensional numpy array of r[y, x].
        f : np.ndarray
            Three dimensional numpy array of f[q, y, x].
        s : Stencil
            Lattice stencil.
        """
        v[:] = _my(f, s) / r

    def forcing(self, f, Fx, Fy, s):
        """
        Adds forcing terms F_q to distributions f.

            f[q, y, x] += F_q

        Modifies `f` in place. DOes not change `Fx`, `Fy` or `s`.

        Parameters
        ----------
        f : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        Fx : float or np.ndarray
            Float, or two-dimensional numpy array of shape (ny, nx).
            Contains x-component of the applied body force.
        Fy : float or np.ndarray
            Float, or two-dimensional numpy array of shape (ny, nx).
            Contains y-component of the applied body force.
        s : Stencil
            Lattice stencil.
        """
        f[:] += (
            3.0
            * s.w[:, None, None]
            * (s.cx[:, None, None] * Fx + s.cy[:, None, None] * Fy)
        )
