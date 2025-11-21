import numpy as np

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
        Calculates fluid velocity x-component from distribution functions.

        Modifies `model.u` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.u[:] = (_mx(model.fi, model.stencil) + 0.5 * model.Fx) / model.r

    def velocity_y(self, model):
        """
        Calculates fluid velocity y-component from distribution functions.

        Modifies `model.v` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.v[:] = (_my(model.fi, model.stencil) + 0.5 * model.Fy) / model.r

    def velocity_x_coll(self, model):
        """
        Calculates collision step velocity x-component from distribution functions.

        Modifies `model.u` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.uc[:] = _mx(model.fi, model.stencil) / model.r

    def velocity_y_coll(self, model):
        """
        Calculates collision step velocity y-component from distribution functions.

        Modifies `model.v` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.vc[:] = _my(model.fi, model.stencil) / model.r

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
            * model.stencil.w[:, np.newaxis, np.newaxis]
            * (
                model.stencil.cx[:, np.newaxis, np.newaxis] * model.Fx
                + model.stencil.cy[:, np.newaxis, np.newaxis] * model.Fy
            )
        )


class MacroscopicGuo(Macroscopic):
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
        Calculates fluid velocity x-component from distribution functions.

        Modifies `model.u` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.u[:] = (_mx(model.fi, model.stencil) + 0.5 * model.Fx) / model.r

    def velocity_y(self, model):
        """
        Calculates fluid velocity y-component from distribution functions.

        Modifies `model.v` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.v[:] = (_my(model.fi, model.stencil) + 0.5 * model.Fy) / model.r

    def velocity_x_coll(self, model):
        """
        Calculates collision step velocity x-component from distribution functions.

        Modifies `model.uc` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.uc[:] = (_mx(model.fi, model.stencil) + 0.5 * model.Fx) / model.r

    def velocity_y_coll(self, model):
        """
        Calculates collision step velocity y-component from distribution functions.

        Modifies `model.vc` in place.

        Parameters
        ----------
        model : Model
            A Model object.
        """
        model.vc[:] = (_my(model.fi, model.stencil) + 0.5 * model.Fy) / model.r

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
        cF = (
            model.stencil.cx[:, np.newaxis, np.newaxis] * model.Fx
            + model.stencil.cy[:, np.newaxis, np.newaxis] * model.Fy
        )
        cu = (
            model.stencil.cx[:, np.newaxis, np.newaxis] * model.u
            + model.stencil.cy[:, np.newaxis, np.newaxis] * model.v
        )
        uF = model.u * model.Fx + model.v * model.Fy
        model.fo[:] += (
            (1.0 - 0.5 * model.collider.omega)
            * model.stencil.w[:, np.newaxis, np.newaxis]
            * (3 * cF + 9 * cu * cF - 3 * uF)
        )
