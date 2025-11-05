from ._equilibrium import feq2


class ColliderSRT:
    def __init__(self, nu, dx, dt):
        tau_star = 0.5 + 3.0 * dt * nu / (dx**2)
        self.omega = 1.0 / tau_star

    def collide(self, f, r, u, v, s):
        """
        Apples the single relaxation time collision algorithm.

        Parameters:
        ----------
        f : np.ndarray
            Three-dimensional numpy array of floats
            containing particle distribution values.

        s : Stencil
            A stencil instance.

        Returns:
        --------
        np.ndarray:
            Three-dimensional numpy array of floats
            containing the particle distribution values
            after collision.
        """
        return self.omega * feq2(r, u, v, s) + (1 - self.omega) * f
