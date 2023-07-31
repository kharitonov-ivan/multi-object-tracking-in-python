import unittest

import numpy as np
import pytest

from src.common.state import Gaussian
from src.configs import GroundTruthConfig, Object


TOL = 1e-4


class Test_GroundTruthConfig(unittest.TestCase):
    def test_ground_truth_config_single_object(self):
        test_total_time = 10
        test_objects = [
            Object(
                initial=Gaussian(x=np.array([0.0, 1.0]), P=np.eye(2)),
                t_birth=0,
                t_death=10,
            )
        ]

        got_ground_truth_config = GroundTruthConfig(  # noqa F841
            object_configs=test_objects,
            total_time=test_total_time,
        )

    def test_ground_truth_config_type(self):
        test_total_time = "10"
        test_objects = [
            Object(
                initial=Gaussian(x=np.array([0.0, 1.0]), P=np.eye(2)),
                t_birth=0,
                t_death=10,
            )
        ]
        with pytest.raises(Exception):
            got_ground_truth_config = GroundTruthConfig(  # noqa F841
                object_configs=test_objects,
                total_time=test_total_time,
            )
