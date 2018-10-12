"""Basic tests for polytopes."""

import unittest
import time
import numpy as np
from hitandrun.hitandrun import HitAndRun
from hitandrun.polytope import Polytope


class TestHitAndRun(unittest.TestCase):
    """Basic tests for hit and run."""

    def test_hitandrun_instantiate(self):
        """Test if HitAndRun object can be created."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        x0 = np.array([-.5, -.5], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        hitandrun = HitAndRun(polytope=polytope,
                              starting_point=x0,
                              thin=1.0,
                              n_samples=100
                              )
        self.assertTrue(isinstance(hitandrun, HitAndRun))

    def test_hitandrun_sampling(self):
        """Test if we can get samples."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        x0 = np.array([-.5, -.5], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        hitandrun = HitAndRun(polytope=polytope, starting_point=x0)
        samples = hitandrun.get_samples(n_samples=100)
        checks = samples @ polytope.A.T - b
        self.assertTrue(np.alltrue(checks < 0))

    def test_hitandrun_timing(self):
        """Test if we can get samples."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        x0 = np.array([-.5, -.5], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        hitandrun = HitAndRun(polytope=polytope, starting_point=x0,
                              n_samples=100, thin=1)
        start = time.time()
        hitandrun.get_samples(n_samples=10000)
        end = time.time()
        print("Running time = %s" % (end - start))
        assert False