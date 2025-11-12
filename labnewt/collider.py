from ._equilibrium import _feq2


class ColliderSRT:
    def __init__(self, nu, dx, dt):
        tau_star = 0.5 + 3.0 * dt * nu / (dx**2)
        self.omega = 1.0 / tau_star

    def collide(self, fo, fi, r, u, v, s):
        """
        Apples the single relaxation time collision algorithm.

        Modifies fo in place. Does not change fi, r, u, v, and s.

        Parameters:
        ----------
        fo : np.ndarray
            Three-dimensional numpy array containing f_out[q, y, x].
        fi : np.ndarray
            Three-dimensional numpy array containing f_in[q, y, x].
        r : np.ndarray
            Two dimensional numpy array containing rho[y, x].
        u : np.ndarray
            Two dimenaional numpy array containing u[y, x].
        v : np.ndarray
            Two dimensional numpy array containing v[y, x].
        s : Stencil
            A stencil instance.
        """
        fo[:] = self.omega * _feq2(r, u, v, s) + (1 - self.omega) * fi
