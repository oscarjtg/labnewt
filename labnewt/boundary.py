"""
Boundary condition classes.
These are passed into Model through the add_boundary_condition() method.
"""


class NoSlip:
    def bounce_back(self, fi, fo, qi, qo, x, y):
        """
        Applies bounce back lattice Boltzmann boundary rule.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        qi : int
            Lattice index of incoming particles.
        qo : int
            Lattice index of outgoing particles.
        x : int
            Spatial index of x-coordinate of grid cell.
        y : int
            Spatial index of y-coordinate of grid cell.
        """
        fi[qi, y, x] = fo[qo, y, x]


class LeftWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to left wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        """
        qi = s.q_right
        qo = s.q_rev[qi]
        x = 0
        y = slice(None)
        self.bounce_back(fi, fo, qi, qo, x, y)


class RightWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to right wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        """
        qi = s.q_left
        qo = s.q_rev[qi]
        x = -1
        y = slice(None)
        self.bounce_back(fi, fo, qi, qo, x, y)


class BottomWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to bottom wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        """
        qi = s.q_up
        qo = s.q_rev[qi]
        x = slice(None)
        y = 0
        self.bounce_back(fi, fo, qi, qo, x, y)


class TopWallNoSlip(NoSlip):
    def apply(self, fi, fo, s):
        """
        Applies no slip BC to top wall, which is stationary.

        Modifies `fi` in place. `fo` is read-only and remains unchanged.

        Parameters
        ----------
        fi : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Modified in place.
        fo : np.ndarray
            Three-dimensional numpy array of shape (nq, ny, nx).
            Not modified.
        s : Stencil
            Lattice stencil.
        """
        qi = s.q_down
        qo = s.q_rev[qi]
        x = slice(None)
        y = -1
        self.bounce_back(fi, fo, qi, qo, x, y)
