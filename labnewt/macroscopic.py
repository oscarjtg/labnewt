from ._moments import _m0, _mx, _my


class Macroscopic:
    def density(self, model):
        raise NotImplementedError

    def velocity_x(self, model):
        raise NotImplementedError

    def velocity_y(self, model):
        raise NotImplementedError

    def forcing(self, model):
        raise NotImplementedError


class MacroscopicStandard(Macroscopic):
    def density(self, model):
        """
        Calculates density from distribution functions.

        Modifies `model.r` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.r[:] = _m0(model.fi)

    def velocity_x(self, model):
        """
        Calculates velocity x-component from distribution functions.

        Modifies `model.u` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.u[:] = _mx(model.fi, model.stencil) / model.r

    def velocity_y(self, model):
        """
        Calculates velocity y-component from distribution functions.

        Modifies `model.v` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.v[:] = _my(model.fi, model.stencil) / model.r

    def forcing(self, model):
        """
        Adds forcing terms F_q to distributions f.

            f[q, y, x] += F_q

        Modifies `model.fo` in place. Does not change `Fx`, `Fy` or `s`.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.fo[:] += (
            3.0
            * model.stencil.w[:, None, None]
            * (
                model.stencil.cx[:, None, None] * model.Fx
                + model.stencil.cy[:, None, None] * model.Fy
            )
        )
