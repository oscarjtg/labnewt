from ._moments import _m0, _mx, _my


class Macroscopic:
    def density(self, r, f):
        """
        Calculates density from distribution functions.

        Modifies r in place. Does not change f.

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

        Modifies u in place. Does not change r, f or s.

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

        Modifies v in place. Does not change f or s.

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

    def force_distribution_array(self, f, Fx, Fy, s):
        """
        Adds forcing terms F_q to distributions f.

            f[q, y, x] += F_q[y, x]

        Modifies `f` in place. `Fx` and `Fy` are read-only and remain unchanged.

        Parameters
        ----------
        f : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        Fx : np.ndarray
            Two-dimensional numpy array of shape (ny, nx).
            Not modified.
        Fy : np.ndarray
            Two-dimensional numpy array of shape (ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        """
        f[:] += (
            3.0
            * s.w[:, None, None]
            * (
                s.cx[:, None, None] * Fx[None, :, :]
                + s.cy[:, None, None] * Fy[None, :, :]
            )
        )

    def force_distribution_constant(self, f, Fx, Fy, s):
        """
        Adds forcing terms F_q to distributions f.

            f[q, y, x] += F_q

        Modifies `f` in place.

        Parameters
        ----------
        f : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        Fx : float
            X-component of force.
        Fy : float
            Y-component of force.
        s : Stencil
            Lattice stencil.
        """
        f[:] += (
            3.0
            * s.w[:, None, None]
            * (s.cx[:, None, None] * Fx + s.cy[:, None, None] * Fy)
        )
