import numpy as np
import unittest
from mot.common.hypothesis_reduction import Hypothesisreduction

TOL = 1e-4


class Test_Hypothesisreduction(unittest.TestCase):
    def test_prune(self):
        num_of_hypotheses = 100
        test_multihypotheses = [None for i in range(num_of_hypotheses)]
        assert NotImplementedError

    def test_cap(self):
        assert NotImplementedError

    def test_merge(self):
        assert NotImplementedError
