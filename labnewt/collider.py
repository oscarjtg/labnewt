from ._equilibrium import _feq2


class Collider:
    def collide(self, model):
        raise NotImplementedError


class ColliderSRT(Collider):
    def __init__(self, nu, dx, dt):
        tau_star = 0.5 + 3.0 * dt * nu / (dx**2)
        self.omega = 1.0 / tau_star

    def collide(self, model):
        """
        Apples the single relaxation time collision algorithm.

        Modifies `model.fo` in-place.

        Parameters:
        ----------
        model : Model
            A model instance.
        """
        model.fo[:] = (
            self.omega * _feq2(model.r, model.uc, model.vc, model.stencil)
            + (1 - self.omega) * model.fi
        )
