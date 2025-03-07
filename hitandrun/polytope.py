"""
A class to hold polytopes in H-representation.

Francesc Font-Clos
Oct 2018

Modified by Aaron Mishkin to support equality constraints.
"""

import numpy as np
import scipy


class Polytope(object):
    """A polytope in H-representation."""

    def __init__(self, A=None, b=None, C=None, d=None):
        """
        Create a polytope in H-representation.

        The polytope is defined as the set of
        points x in Rn such that

        A x <= b
        C x = d

        """
        # dimensionality verifications
        assert A is not None and b is not None
        assert len(b.shape) == 1
        assert len(A.shape) == 2
        assert A.shape[0] == len(b)
        # store data
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.dim = A.shape[1]
        self.nplanes = A.shape[0]
        self._find_auxiliar_points_in_planes()

        # basis for Null(C)
        self.N = None

        # compute span of Null(C) if necessary
        if self.C is not None:
            n, m = C.shape

            # Compute the Null space of C
            U, s, Vh = scipy.linalg.svd(C, full_matrices=True)
            zeros = np.argwhere(np.isclose(s, 0))
            if len(zeros) == 0:
                self.N = Vh[n:].T
            else:
                self.N = Vh[np.min(zeros) :].T

            if self.N.size == 0:
                raise ValueError("Cx = d has a unique solution!")

    def check_inside(self, point):
        """Check if a point is inside the polytope."""
        checks = self.A @ point <= self.b
        check = np.all(checks)

        check = check and np.allclose(self.C @ point, self.d)

        return check

    def _find_auxiliar_points_in_planes(self):
        """Find an auxiliar point for each plane."""
        aux_points = [
            self._find_auxiliar_point(self.A[i], self.b[i])
            for i in range(self.nplanes)
        ]
        self.auxiliar_points = aux_points

    def _find_auxiliar_point(self, Ai, bi):
        """Find an auxiliar point for one plane."""
        p = np.zeros(self.dim)
        j = np.argmax(Ai != 0)
        p[j] = bi / Ai[j]
        return p
