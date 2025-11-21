"""Force classes."""

from .gravity import Gravity


class Force:
    def apply(self, model):
        """Calculate force components from model arrays."""
        raise NotImplementedError


class ConstantGravityForce(Force):
    def __init__(self, dx, dt, g_magnitude=9.81, direction=(0.0, -1.0)):
        """
        Initialises ConstantGravityForce class.

        Note that conversion factor Cg needed to convert
        physical units to lattice units.

        Parameters
        ----------
        dx : float
            Float giving grid spacing.
        dt : float
            Float giving time step.
        g_magnitude : float, optional
            Float giving magnitude of gravitational acceleration,
            in physical units.
        direction : tuple, optional
            Tuple of floats giving direction of gravitational force.
        """
        self.Cg = dt**2 / dx
        self.gravity = Gravity()
        self.gravity.set_gravity(magnitude=g_magnitude, direction=direction)
        self.Fx = self.Cg * self.gravity.gx
        self.Fy = self.Cg * self.gravity.gy

    def _set_force_components(self):
        self.Fx = self.Cg * self.gravity.gx
        self.Fy = self.Cg * self.gravity.gy

    def set_gravity_magnitude(self, magnitude):
        """Sets the magnitude of the gravity vector."""
        self.gravity.set_gravity(magnitude=magnitude)
        self._set_force_components()

    def set_gravity_direction(self, direction):
        """Sets the direction of the gravity vector."""
        self.gravity.set_gravity(direction=direction)
        self._set_force_components()

    def apply(self, model):
        """
        Adds constant, uniform force to `model`.

        Modifies `model.Fx` and `model.Fy` arrays in-place.

        Parameters
        ----------
        model : Model
            A Model object, or an object that inherits from Model.

        Returns
        -------
        None
        """
        model.Fx += self.Fx
        model.Fy += self.Fy


class GravityForce(ConstantGravityForce):
    def apply(self, model):
        """
        Adds gravitational force weighted by density and fill fraction.

        Modifies `model.Fx` and `model.Fy` arrays in-place.

        Parameters
        ----------
        model : Model or FreeSurfaceModel
            A Model object, or an object that inherits from Model.

        Returns
        -------
        None
        """
        self.Fx = self.Cg * self.gravity.gx * model.r
        self.Fy = self.Cg * self.gravity.gy * model.r
        if hasattr(model, "vof"):
            self.Fx *= model.vof.phi
            self.Fy *= model.vof.phi
        model.Fx += self.Fx
        model.Fy += self.Fy
